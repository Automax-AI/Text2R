from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.load import loads
from langchain_openai import ChatOpenAI
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_ag_grid as dag
import pandas as pd
import re
import base64
import io
from dotenv import find_dotenv, load_dotenv
import os
import tempfile
import shutil
import rpy2.robjects as ro
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import default_converter, pandas2ri, conversion
from rpy2.rinterface_lib.embedded import RRuntimeError

pandas2ri.activate()
utils = importr('utils')
ggplot2 = importr('ggplot2')

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


model = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

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

# Add these CSS styles at the top of your file
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
]

app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([
        # Header section
        html.Div([
            html.Img(
                src='assets/automax-logo.png', 
                style={
                    'height': '60px',
                    'marginRight': '20px'
                }
            ),
            html.H1(
                "Automax AI Market Conditions Tool",
                style={
                    'color': '#2c3e50',
                    'fontFamily': 'Roboto, sans-serif',
                    'margin': '0'
                }
            )
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '20px',
            'backgroundColor': 'white',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        }),

        # Main content
        html.Div([
            dcc.Store(id='chat-history', data=[]),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt", style={'marginRight': '10px'}),
                    'Drag and Drop or ',
                    html.A('Select Files', style={'color': '#3498db'})
                ]),
                style={
                    'width': '100%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'textAlign': 'center',
                    'margin': '20px 0',
                    'backgroundColor': '#f8f9fa',
                    'transition': 'border .24s ease-in-out'
                },
                multiple=True
            ),
            
            html.Div(id="output-grid", style={'margin': '20px 0'}),
            
            # Input section
            html.Div([
                dcc.Textarea(
                    id='user-request',
                    placeholder='Enter your request here...',
                    style={
                        'width': '100%',
                        'height': '100px',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'border': '1px solid #ddd',
                        'marginBottom': '10px',
                        'fontFamily': 'Roboto, sans-serif'
                    }
                ),
                html.Button(
                    'Submit',
                    id='my-button',
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'padding': '10px 20px',
                        'border': 'none',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'fontSize': '16px',
                        'transition': 'background-color 0.3s'
                    }
                )
            ], style={'width': '100%', 'maxWidth': '800px', 'margin': '0 auto'}),
            
            # Results section
            dcc.Loading(
                children=[
                    html.Div(
                        id='my-figure',
                        style={'margin': '20px 0'}
                    ),
                    html.Div([
                        dcc.Markdown(
                            id='content',
                            style={
                                'padding': '20px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '5px',
                                'marginTop': '20px'
                            }
                        )
                    ])
                ],
                type='cube',
                color='#3498db'
            )
        ], style={
            'padding': '20px',
            'maxWidth': '1200px',
            'margin': '0 auto'
        })
    ], style={
        'fontFamily': 'Roboto, sans-serif',
        'backgroundColor': '#f5f6fa',
        'minHeight': '100vh'
    })
])


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
    Output('chat-history', 'data'),
    Input('my-button', 'n_clicks'),
    State('user-request', 'value'),
    State('stored-data', 'data'),
    State('stored-file-name', 'data'),
    State('chat-history', 'data'),
    prevent_initial_call=True
)
def create_graph(_, user_input, file_data, file_name, chat_history):
    if not chat_history:
        chat_history = []
        
    df = pd.DataFrame(file_data)
    df_5_rows = df.head()
    csv_string = df_5_rows.to_string(index=False)
    
    formatted_history = []
    for message in chat_history:
        if isinstance(message, dict):
            if message["role"] == "human":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You're a data visualization expert using R and ggplot2. "
         "The data is available in a dataframe called 'df'. "
         "Here are the first 5 rows: {data}. "
         "Write R code that processes the data and creates the visualization using ggplot2."
         "Always use backticks around column names with spaces like `Close Date`. "
         "Never use bare column names with spaces. "
         "Always explicitly load all required packages with library() calls, even when using :: notation."
         "Include all necessary data transformations."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | model
    response = chain.invoke(
        {
            "input": user_input,  
            "data": csv_string,
            "chat_history": formatted_history,  
        }
    )
    result_output = response.content
    print(result_output)
    
    chat_history.append({"role": "human", "content": user_input})
    chat_history.append({"role": "assistant", "content": result_output})
    
    # Extract R code block
    code_block_match = re.search(
        r'```(?:r|R)\s*\n(.*?)```',  
        result_output, 
        re.DOTALL
    )

    if not code_block_match:
        print("No code block found with improved regex") 
        return html.Div("No valid R code found."), result_output

    code_block = code_block_match.group(1).strip()
    
    print("Extracted Code Block:\n", code_block)
    
    try:
        # Convert pandas DataFrame to R dataframe
        with conversion.localconverter(default_converter + pandas2ri.converter):
            r_df = pandas2ri.py2rpy(df)
            ro.globalenv['df'] = r_df


        # Install missing packages
        package_matches = re.findall(
            r'library\(["\']?([^"\')]+)["\']?\)|require\(["\']?([^"\')]+)["\']?\)', 
            code_block
        )
        required_packages = list(set(
            [pkg for match in package_matches for pkg in match if pkg]
        ))
        print("Required packages:", required_packages)
        # Install missing packages
        for pkg in required_packages:
            if not isinstalled(pkg):
                utils.install_packages(pkg, repos="https://cran.r-project.org")

        # Create temporary directory for plot
        temp_dir = tempfile.mkdtemp()
        plot_path = os.path.join(temp_dir, 'plot.png')

        # Execute R code
        ro.r(f'''
        {code_block}
        ggsave("{plot_path}", width=10, height=6)
        ''')


        with open(plot_path, 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            img_src = f'data:image/png;base64,{encoded_image}'


        shutil.rmtree(temp_dir)
        return html.Img(
            src=img_src, 
            style={
                "width": "50%",  
                "height": "auto",  
                "max-height": "500px",
                "max-width": "750px",  
                "display": "block",
                "margin": "0 auto"  
            }
        ), result_output, chat_history


    except RRuntimeError as e:
        error_message = f"Error executing R code:\n{str(e)}"
        return html.Div(error_message), result_output
    except Exception as e:
        error_message = f"General error:\n{str(e)}"
        return html.Div(error_message), result_output

if __name__ == '__main__':
    app.run_server(debug=False, port=8008)