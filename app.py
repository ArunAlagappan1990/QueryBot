import os
import re
import faiss
import openai
from threading import Thread
import dash_bootstrap_components as dbc
from dash import Dash, html, Input, Output, State
from dash.exceptions import PreventUpdate
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

# Reference the underlying flask app (Used by gunicorn webserver in Heroku production deployment)
server = app.server 

# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)


app.layout = dbc.Container([
    html.Br(),
    html.H1('LLM Query Assistant'),
    html.H2("Kaplan and Sadock's Comprehensive Textbook of Psychiartry"),
    html.Br(),
    html.Div(
        [
            dbc.Label('Enter your OpenAI API Key:'),
            dbc.Input(id='api-key', type='text', placeholder='Enter your API key here...')
        ]
    ),
    html.Br(),
    html.Div(
        [
            dbc.Button('Initialize Index', id='init-index-btn', n_clicks=0)
        ]
    ),
    html.Br(),
    dbc.Spinner(
        children=[
            dbc.Textarea(id='my-input', size="lg", placeholder="Enter your text", disabled=True),
            ], 
        id='spinner', 
        color="dark", 
        type="border", 
        fullscreen=False
        ),
    html.Br(),
    dbc.Button("Query", id="gen-button", className="me-2", n_clicks=0, disabled=True),
    html.Br(),
    html.H3('Output:'),
    dbc.Spinner(
        children=[html.Div(id='my-output')],
        id='spinner_output', 
        color="dark", 
        type="border", 
        fullscreen=False
        ),
    html.Br()

])


@app.callback(
    [Output('my-input', 'disabled'),
     Output('gen-button', 'disabled')],
    [Input('init-index-btn', 'n_clicks')],
    [State('api-key', 'value')]
)
def initialize_index(n_clicks, api_key):
    global global_index, global_query_engine

    if n_clicks == 0 or not api_key:
        raise PreventUpdate

    openai.api_key = api_key
    Settings.llm = OpenAI(temperature=0.1, model="gpt-4")

    # Check if the index is already loaded
    try:
        global_index 
    except NameError:
        # Start the indexing process in a separate thread
        indexing_thread = Thread(target=run_indexing_process)
        indexing_thread.start()

        # Wait for the indexing process to finish
        indexing_thread.join()

    return False, False  # Enable input, button, and log interval

@app.callback(
    Output('my-output', 'children'),
    [Input('gen-button', 'n_clicks')],
    [State('api-key', 'value'),
     State('my-input', 'value')]
)
def update_output_div(gen, api_key, input_value):
    if gen == 0 or not api_key or not input_value:
        return 'Enter your API key, and enter your text, then click "Generate Text".'

    response = global_query_engine.query(input_value)

    output = [f"API Key: {api_key}", f"Input Text: {input_value}"]
    if hasattr(response, 'metadata'):
        document_info = str(response.metadata)
        find = re.findall(r"'page_label': '[^']*', 'file_name': '[^']*'", document_info)
        output.append(str(response))
        output.append(f"Context Information: {find}")

    return html.Div([html.P(line) for line in output])

def run_indexing_process():
    global global_index, global_query_engine, log_messages

    PERSIST_DIR = "./vector_storage"

    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, show_progress=True
            )
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
        storage_context = StorageContext.from_defaults(
            vector_store = vector_store, persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context=storage_context)

    global_index = index
    global_query_engine = index.as_query_engine()


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
