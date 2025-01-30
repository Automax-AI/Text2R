from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import re
import base64
import io
from dotenv import find_dotenv, load_dotenv
import os

# Load the API key
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  

# Choose the model
model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Extract the Plotly fig object from the code generated in the callback
def get_fig_from_code(code):
    local_variables = {}
    exec(code, {}, local_variables)
    return local_variables['fig']

# Upload data
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            df = pd.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
        
    return html.Div([
        html.H5(filename),
        dag.AgGrid(
            rowData=df.to_dict('records'),
            columnDefs = [{"field": i} for i in df.columns],
            defaultColDef={"filter": True, "sortable": True, "floatingFilter": True}),
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        dcc.Store(id='stored-file-name', data=filename),
        html.Hr()
    ])

app = Dash(suppress_callback_exceptions=True)

app.layout = [
    html.H1("Automax AI Market Conditions Tool"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
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
        multiple=True
    ),
    html.Div(id="output-grid"),
    dcc.Textarea(id='user-request', style={'width': '50%', 'height': 50, 'margin-top': 20}),
    html.Br(),
    html.Button('Submit', id='my-button'),
    dcc.Loading(
        [
            html.Div(id='my-figure', children=''),
            dcc.Markdown(id='content', children='')
        ],
        type='cube'
    ),
]

@callback(
    Output('output-grid', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)
        ]
        return children

@callback(
    Output('my-figure', 'children'),
    Output('content', 'children'),
    Input('my-button', 'n_clicks'),
    State('user-request', 'value'),
    State('stored-data', 'data'),
    State('stored-file-name', 'data'),
    prevent_initial_call=True
)
def create_graph(_, user_input, file_data, file_name):
    df = pd.DataFrame(file_data)
    df_5_rows = df.head()
    csv_string = df_5_rows.to_string(index=False)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You're a data visualization expert and you use your favourite graphing library Plotly only. "
                       "Suppose that the data is provided as a {name_of_file} file. Here are the first 5 rows of the data set: {data}. "
                       "Follow the user's indications when creating the graph. Use scikit-learn if you need to do any data analysis."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "data": csv_string,
            "name_of_file": file_name,
        }
    )
    result_output = response.content
    print(result_output)
    
    code_block_match = re.search(r'```(?:python)?\n(.*?)```', result_output, re.DOTALL)
    if code_block_match:
        code_block = code_block_match.group(1).strip()
        cleaned_code = re.sub(r'(?m)^\s*fig\.show\(\)\s*$', '', code_block)
        fig = get_fig_from_code(cleaned_code)
        return dcc.Graph(figure=fig), result_output
    else:
        return "", result_output

if __name__ == '__main__':
    app.run_server(debug=False, port=8008)
