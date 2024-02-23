import os
import re
import faiss
import openai
import base64
from threading import Thread
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
    Settings
)
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

app = Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

# Global variables to store the index, query engine, and log messages
global_index = None
global_query_engine = None
log_messages = []

# dimensions of text-ada-embedding-002
d = 1536
faiss_index = faiss.IndexFlatL2(d)

app.layout = dbc.Container([
    html.Br(),
    html.H1('LLM Query Assistant'),
    html.Br(),
    html.Div(
        [
            dbc.Label('Enter your OpenAI API Key:'),
            dbc.Input(id='api-key', type='text', placeholder='Enter your API key here...')
        ]
    ),
    html.Div(
        [
            dcc.Upload(
                id='upload-pdf',
                children=html.Div(['Drag and Drop or ', html.A('Select a PDF File')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                multiple=False,
                accept='.pdf',
            )
        ]
    ),
    dbc.Spinner(html.Div(id='upload-status')),
    html.Br(),
    html.Div(
        [
            dbc.Button('Initialize Index', id='init-index-btn', n_clicks=0)
        ]
    ),
    html.Br(),
    html.H5('Indexing Log:'),
    html.Ul(id='indexing-log'),
    dcc.Interval(id='log-interval', interval=2000, n_intervals=0, disabled=True),  # Update every 2 seconds
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
    Output('upload-status', 'children'),
    [Input('upload-pdf', 'contents')],
    [State('upload-pdf', 'filename')]
)
def save_uploaded_file(contents, filename):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if not os.path.exists('data'):
        os.makedirs('data')

    file_path = os.path.join('data', filename)
    with open(file_path, 'wb') as f:
        f.write(decoded)

    return f'File {filename} uploaded successfully!'

@app.callback(
    [Output('my-input', 'disabled'),
     Output('gen-button', 'disabled'),
     Output('log-interval', 'disabled')],
    [Input('init-index-btn', 'n_clicks')],
    [State('api-key', 'value')]
)
def initialize_index(n_clicks, api_key):
    global global_index, global_query_engine

    if n_clicks == 0 or not api_key:
        raise PreventUpdate

    openai.api_key = api_key
    Settings.llm = OpenAI(temperature=0.1, model="gpt-4")


    # Start the indexing process in a separate thread
    indexing_thread = Thread(target=run_indexing_process)
    indexing_thread.start()

    # Wait for the indexing process to finish
    indexing_thread.join()

    return False, False, False  # Enable input, button, and log interval

@app.callback(
    Output('indexing-log', 'children'),
    [Input('log-interval', 'n_intervals')]
)
def update_indexing_log(n_intervals):
    return [html.Li(msg) for msg in log_messages]

@app.callback(
    Output('my-output', 'children'),
    [Input('gen-button', 'n_clicks')],
    [State('api-key', 'value'),
     State('my-input', 'value')]
)
def update_output_div(gen, api_key, input_value):
    if gen == 0 or not api_key or not input_value:
        return 'Enter your API key, upload a PDF file, and enter your text, then click "Generate Text".'

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

    PERSIST_DIR = "./storage"

    log_messages.append('Chunking & loading the uploaded files...')
    documents = SimpleDirectoryReader("data").load_data()
    log_messages.append(f'Loaded {len(documents)} documents.')

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    log_messages.append(f'Indexing...')
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, show_progress=True
        )
    log_messages.append('Indexing completed. Persisting index...')
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    log_messages.append('Index persisted successfully.')

    global_index = index
    global_query_engine = index.as_query_engine()

if __name__ == '__main__':
    app.run_server(debug=True)
