import os
import re
import faiss
import openai
from dash import Dash, html, Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server  # Reference the underlying flask app

# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

class QueryBot:
    def __init__(self):
        self.index = None
        self.query_engine = None
        self.is_index_initialized = False

    def initialize_index(self, api_key):
        if self.is_index_initialized:
            return
        openai.api_key = api_key
        Settings.llm = OpenAI(temperature=0.1, model="gpt-4")
        self.index = run_indexing_process()
        self.query_engine = self.index.as_query_engine()
        self.is_index_initialized = True

    def query(self, input_text):
        if not self.is_index_initialized:
            raise Exception("Index is not initialized")
        response = self.query_engine.query(input_text)
        return response

# Instantiate the QueryBot
query_bot = QueryBot()

app.layout = dbc.Container([
    html.Br(),
    html.H1('LLM Query Assistant'),
    html.H2("Kaplan and Sadock's Comprehensive Textbook of Psychiatry"),
    html.Br(),
    html.Div([
        dbc.Label('Enter your OpenAI API Key:'),
        dbc.Input(id='api-key', type='text', placeholder='Enter your API key here...')
    ]),
    html.Br(),
    html.Div([
        dbc.Button('Initialize Index', id='init-index-btn', n_clicks=0)
    ]),
    html.Br(),
    dbc.Spinner([
        dbc.Textarea(id='my-input', size="lg", placeholder="Enter your text", disabled=True),
    ], id='spinner', color="dark", type="border", fullscreen=False),
    html.Br(),
    dbc.Button("Query", id="gen-button", className="me-2", n_clicks=0, disabled=True),
    html.Br(),
    html.H3('Output:'),
    dbc.Spinner([
        html.Div(id='my-output')
    ], id='spinner_output', color="dark", type="border", fullscreen=False),
    html.Br()
])

@app.callback(
    [Output('init-index-btn', 'disabled'),
     Output('my-input', 'disabled'),
     Output('gen-button', 'disabled')],
    [Input('init-index-btn', 'n_clicks')],
    [State('api-key', 'value')]
)
def initialize_index(n_clicks, api_key):
    if n_clicks > 0 and api_key:
        query_bot.initialize_index(api_key)
    return query_bot.is_index_initialized, not query_bot.is_index_initialized, not query_bot.is_index_initialized

@app.callback(
    Output('my-output', 'children'),
    [Input('gen-button', 'n_clicks')],
    [State('my-input', 'value')]
)
def update_output_div(n_clicks, input_value):
    if n_clicks == 0 or not input_value:
        raise PreventUpdate
    try:
        response = query_bot.query(input_value)
        output = [f"Input Text: {input_value}"]
        if hasattr(response, 'metadata'):
            document_info = str(response.metadata)
            find = re.findall(r"'page_label': '[^']*', 'file_name': '[^']*'", document_info)
            output.append(str(response))
            output.append(f"Context Information: {find}")
        return html.Div([html.P(line) for line in output])
    except Exception as e:
        return f"Error: {str(e)}"

def run_indexing_process():
    PERSIST_DIR = "./vector_storage"
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context=storage_context)
    return index

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
