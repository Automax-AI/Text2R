from langchain_core.messages import HumanMessage, AIMessage
from langchain.load import loads
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import streamlit as st
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
import threading
from rpy2.robjects.conversion import localconverter
import warnings
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from langchain.output_parsers import PydanticOutputParser
import json
import hashlib
from supabase import create_client, Client
from datetime import datetime
import uuid
import requests
import time

# Suppress R initialization warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Pydantic model for structured output
class VisualizationResponse(BaseModel):
    r_code: str = Field(description="The complete R code block that creates the visualization")
    explanation: str = Field(description="Detailed explanation of what the visualization shows and how to interpret it")
    title: Optional[str] = Field(description="A title for the visualization", default=None)

    @classmethod
    def get_json_schema(cls):
        schema = {
            "title": "Visualization Response",
            "description": "A structured response containing R code and explanation for data visualization",
            "type": "object",
            "properties": {
                "r_code": {
                    "title": "R Code",
                    "description": "The complete R code block that creates the visualization",
                    "type": "string"
                },
                "explanation": {
                    "title": "Explanation",
                    "description": "Detailed explanation of what the visualization shows and how to interpret it",
                    "type": "string"
                },
                "title": {
                    "title": "Title",
                    "description": "A title for the visualization",
                    "type": "string"
                }
            },
            "required": ["r_code", "explanation"]
        }
        return schema

# Supabase setup
def get_supabase_client() -> Client:
    """Create and return a Supabase client instance"""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    
    if not url or not key:
        st.error("Supabase credentials not found. Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        st.stop()
        
    return create_client(url, key)

# Initialize R in the main thread
def init_r():
    pandas2ri.activate()    
    
    if not hasattr(st.session_state, 'r_initialized'):
        try:
            # Store these in session state
            with localconverter(default_converter + pandas2ri.converter):
                st.session_state.utils = importr('utils')
                st.session_state.ggplot2 = importr('ggplot2')
            
            st.session_state.r_initialized = True
            
            # Pre-install common R packages for visualization
            common_packages = [
                "ggplot2", "dplyr", "tidyr", "lubridate", "scales", "RColorBrewer",
                "viridis", "plotly", "ggthemes", "gridExtra", "reshape2", "forcats",
                "stringr", "purrr", "readr", "readxl", "knitr", "broom", "modelr",
                "tibble", "magrittr", "data.table"
            ]
            
            for pkg in common_packages:
                if not isinstalled(pkg):
                    try:
                        with localconverter(default_converter + pandas2ri.converter):
                            st.session_state.utils.install_packages(pkg, repos="https://cran.r-project.org")
                    except Exception as e:
                        st.warning(f"Couldn't install package {pkg}: {str(e)}")
        except Exception as e:
            st.error(f"Error initializing R: {str(e)}")
            import traceback
            st.error(traceback.format_exc())


dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

model = ChatOpenAI(model="o3-mini", api_key=os.getenv("OPENAI_API_KEY"))

def process_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error('Please upload a CSV or Excel file.')
            return None
        return df
    except Exception as e:
        st.error(f'Error processing file: {str(e)}')
        return None

def create_visualization(df, user_input, chat_history):
    if not chat_history:
        chat_history = []

    df_5_rows = df.head()
    csv_string = df_5_rows.to_string(index=False)

    formatted_history = []
    for message in chat_history:
        if isinstance(message, dict):
            if message["role"] == "human":
                formatted_history.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant":
                formatted_history.append(AIMessage(content=message["content"]))

    parser = PydanticOutputParser(pydantic_object=VisualizationResponse)

    format_instructions = """Return a JSON object with the following structure:
{{
    "r_code": "Your complete R code that creates the visualization",
    "explanation": "Your detailed explanation of the visualization",
    "title": "Optional title for the visualization"
}}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a data visualization expert using R and ggplot2. "
         "The data is available in a dataframe called 'df'. "
         "Here are the first 5 rows: {data}. "
         "\nYour task is to create a visualization based on the user's request. "
         "Follow these rules:"
         "\n1. Always use backticks around column names with spaces"
         "\n2. Never use bare column names with spaces"
         "\n3. Always explicitly load all required packages with library() calls"
         "\n4. Include all necessary data transformations"
         "\n5. DO NOT include any print() statements - the plot will be saved directly"
         "\n6. Store your final plot in a variable called 'p'"
         f"\n\n{format_instructions}"
         "\n\nIMPORTANT: Always format your response in the required structure"
         ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | model
    response = chain.invoke({
        "input": user_input,
        "data": csv_string,
        "chat_history": formatted_history,
    })
    
    try:

        parsed_response = parser.parse(response.content)
        
        result_output = f"### {parsed_response.title if parsed_response.title else 'Visualization'}\n\n"
        result_output += f"{parsed_response.explanation}\n\n"
        result_output += f"```r\n{parsed_response.r_code}\n```"
        
        chat_history.append({"role": "human", "content": user_input})
        chat_history.append({"role": "assistant", "content": result_output})

        try:
            with localconverter(default_converter + pandas2ri.converter) as cv:
                r_df = pandas2ri.py2rpy(df)
                ro.globalenv['df'] = r_df
                
                package_matches = re.findall(
                    r'library\(["\']?([^"\')]+)["\']?\)|require\(["\']?([^"\')]+)["\']?\)',
                    parsed_response.r_code
                )
                required_packages = list(set([pkg for match in package_matches for pkg in match if pkg]))
                
                for pkg in required_packages:
                    if not isinstalled(pkg):
                        st.session_state.utils.install_packages(pkg, repos="https://cran.r-project.org")
    
                # Create temporary directory for plot
                temp_dir = tempfile.mkdtemp()
                plot_path = os.path.join(temp_dir, 'plot.png')
                
                grdevices = importr('grDevices')
                
                # Simple approach: Just run the R code and then save the plot
                # First run the code to create the plot object 'p'
                ro.r(parsed_response.r_code)
                
                # Then explicitly save it as PNG
                ro.r(f'''
                # Save the plot directly to a PNG file
                ggsave("{plot_path}", plot = p, width = 6, height = 4, dpi = 300)
                ''')
                
                # Verify the file was created
                if not os.path.exists(plot_path) or os.path.getsize(plot_path) == 0:
                    st.error("Failed to generate plot file")
                    raise Exception("Plot file was not created successfully")

            with open(plot_path, 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                
            shutil.rmtree(temp_dir)
            return encoded_image, result_output, chat_history, "image/png"

        except RRuntimeError as e:
            st.error(f"Error executing R code:\n{str(e)}")
            return None, result_output, chat_history, None
        except Exception as e:
            st.error(f"General error:\n{str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None, result_output, chat_history, None
            
    except Exception as e:
        st.error(f"Error parsing LLM response: {str(e)}")
        return None, response.content, chat_history, None

def save_file_to_db(user_id, filename, content, content_type):
    """Save an uploaded file to Supabase for a user"""
    try:
        supabase = get_supabase_client()
        
        # Delete any existing files for this user
        supabase.table('user_files').delete().eq('user_id', user_id).execute()
        
        file_id = str(uuid.uuid4())
        uploaded_at = datetime.now().isoformat()
        
        # Convert content to base64 string if it's binary
        if isinstance(content, bytes):
            content = base64.b64encode(content).decode('utf-8')
        
        # Insert the new file
        response = supabase.table('user_files').insert({
            'id': file_id,
            'user_id': user_id,
            'filename': filename,
            'file_data': content,
            'content_type': content_type,
            'uploaded_at': uploaded_at
        }).execute()
        
        if not response.data:
            return None
            
        return file_id
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def get_user_file(user_id):
    """Retrieve the user's stored file data from Supabase"""
    try:
        supabase = get_supabase_client()
        
        response = supabase.table('user_files').select('id, filename, file_data, content_type').eq('user_id', user_id).execute()
        
        if not response.data:
            return None
        
        file_data = response.data[0]
        
        # Convert base64 back to binary if needed
        data = file_data['file_data']
        if isinstance(data, str) and not data.startswith('{'):
            try:
                data = base64.b64decode(data)
            except Exception:
                pass
        
        return {
            "id": file_data['id'],
            "filename": file_data['filename'],
            "data": data,
            "content_type": file_data['content_type']
        }
    except Exception as e:
        st.error(f"Error retrieving file: {str(e)}")
        return None

def save_chat_history(user_id, messages):
    """Save chat history for a user to Supabase"""
    try:
        supabase = get_supabase_client()
        
        # Check if history exists for this user
        response = supabase.table('chat_history').select('id').eq('user_id', user_id).execute()
        
        updated_at = datetime.now().isoformat()
        messages_json = json.dumps(messages)
        
        if response.data:
            # Update existing history
            supabase.table('chat_history').update({
                'messages': messages_json,
                'updated_at': updated_at
            }).eq('user_id', user_id).execute()
        else:
            # Create new history entry
            history_id = str(uuid.uuid4())
            supabase.table('chat_history').insert({
                'id': history_id,
                'user_id': user_id,
                'messages': messages_json,
                'updated_at': updated_at
            }).execute()
    except Exception as e:
        st.error(f"Error saving chat history: {str(e)}")

def get_chat_history(user_id):
    """Retrieve chat history for a user from Supabase"""
    try:
        supabase = get_supabase_client()
        
        response = supabase.table('chat_history').select('messages').eq('user_id', user_id).execute()
        
        if not response.data:
            return []
        
        return json.loads(response.data[0]['messages'])
    except Exception as e:
        st.error(f"Error retrieving chat history: {str(e)}")
        return []

def clear_user_data(user_id, clear_file=True, clear_history=True):
    """Clear user's file and/or chat history from Supabase"""
    try:
        supabase = get_supabase_client()
        
        if clear_file:
            supabase.table('user_files').delete().eq('user_id', user_id).execute()
        
        if clear_history:
            supabase.table('chat_history').delete().eq('user_id', user_id).execute()
    except Exception as e:
        st.error(f"Error clearing user data: {str(e)}")

# Supabase Authentication Functions
def register_with_supabase(name, email, password):
    """Register a new user using Supabase Auth"""
    try:
        supabase = get_supabase_client()
        
        # Register user with Supabase Auth
        auth_response = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "data": {
                    "name": name
                }
            }
        })
        
        if not auth_response.user:
            return False, "Registration failed"
        
        # Wait for the user to be created in the database
        time.sleep(1)
        
        return True, auth_response.user
        
    except Exception as e:
        error_msg = str(e)
        if "User already registered" in error_msg:
            return False, "Email already registered"
        return False, f"Error registering user: {error_msg}"

def login_with_supabase(email, password):
    """Login user using Supabase Auth"""
    try:
        supabase = get_supabase_client()
        
        # Login user with Supabase Auth
        auth_response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if not auth_response.user:
            return False, "Login failed"
        
        # Get user metadata
        user = auth_response.user
        user_data = {
            "id": user.id,
            "name": user.user_metadata.get("name", "User"),
            "email": user.email
        }
        
        return True, user_data
        
    except Exception as e:
        error_msg = str(e)
        if "Invalid login credentials" in error_msg:
            return False, "Invalid email or password"
        return False, f"Error logging in: {error_msg}"

def logout_from_supabase():
    """Logout user from Supabase Auth"""
    try:
        supabase = get_supabase_client()
        supabase.auth.sign_out()
        return True
    except Exception as e:
        st.error(f"Error logging out: {str(e)}")
        return False

def setup_authentication():
    """Set up the authentication pages (login/register) using Supabase Auth"""
    if 'user' not in st.session_state:
        st.session_state.user = None
        
    if st.session_state.user is not None:
        return True
        
    auth_page = st.sidebar.radio("Authentication", ["Login", "Register"])
    
    if auth_page == "Login":
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if email and password:
                with st.spinner("Logging in..."):
                    success, result = login_with_supabase(email, password)
                    if success:
                        st.session_state.user = result
                        st.success(f"Welcome back, {result['name']}!")
                        time.sleep(1) 
                        st.rerun()
                    else:
                        st.error(result)
            else:
                st.warning("Please enter both email and password")
                
    else:  # Register page
        st.subheader("Register")
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm Password", type="password")
        
        if st.button("Register"):
            if name and email and password and password2:
                if password != password2:
                    st.error("Passwords do not match")
                else:
                    with st.spinner("Creating your account..."):
                        success, result = register_with_supabase(name, email, password)
                        if success:
                            st.success("Account created successfully! Please login.")
                        else:
                            st.error(result)
            else:
                st.warning("Please fill out all fields")
    
    return False

def initialize_supabase_tables():
    """Initialize the Supabase database with necessary tables if they don't exist"""
    # This function only needs to verify tables exist by querying them
    try:
        supabase = get_supabase_client()
        
        # Check if tables exist by querying them - will throw an error if they don't
        try:
            # Just check if we can query the tables
            supabase.table('user_files').select('id').limit(1).execute()
            supabase.table('chat_history').select('id').limit(1).execute()
        except Exception as e:
            # Tables may not exist
            st.error(f"Database tables not properly set up: {str(e)}")
            st.info("Please refer to the README.md file for instructions on setting up the database tables.")
    except Exception as e:
        st.error(f"Error connecting to Supabase: {str(e)}")

def main():
    st.set_page_config(page_title="Automax AI Market Conditions Tool", layout="wide")
    
    initialize_supabase_tables()
    
    init_r()
    
    is_authenticated = setup_authentication()
    
    if not is_authenticated:
        st.title("Welcome to Automax AI Market Conditions Tool")
        st.write("""
        This powerful tool helps you analyze market conditions using AI and R visualization.
        Please login or register to get started.
        """)
        return
    
    user = st.session_state.user
    
    # Header
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.image('assets/automax-logo.png', width=100)
    with col2:
        st.title("Automax AI Market Conditions Tool")
    with col3:
        st.write(f"Welcome, {user['name']}")
        if st.button("Logout"):
            logout_from_supabase()
            st.session_state.user = None
            st.rerun()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Controls")
        if st.button("New Conversation"):
            # First clear data from database
            clear_user_data(user['id'], clear_file=True, clear_history=True)
            
            # Reset ALL relevant session state variables
            if 'chat_history' in st.session_state:
                st.session_state.pop('chat_history')
            # Force df to be None to reset the file display
            if 'df' in st.session_state:
                st.session_state.pop('df')
            # Clear any other cached data
            if 'user_file' in st.session_state:
                st.session_state.pop('user_file')
            
            st.success("Started a new conversation! Please upload a new file.")
            time.sleep(1)  # Brief pause to show success message
            st.rerun()
            
        if st.button("Clear History Only"):
            # Clear just the history from database
            clear_user_data(user['id'], clear_file=False, clear_history=True)
            
            # Reset only the chat history in session state
            if 'chat_history' in st.session_state:
                st.session_state.pop('chat_history')
            
            st.success("Chat history cleared!")
            time.sleep(1)  # Brief pause to show success message 
            st.rerun()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        # Try to load from database
        chat_history = get_chat_history(user['id'])
        st.session_state.chat_history = chat_history
    
    # Store df in session state to maintain consistency
    if 'df' not in st.session_state:
        st.session_state.df = None

    # Check if user has an existing file
    existing_file = get_user_file(user['id'])

    if existing_file:
        st.info(f"Using your existing file: {existing_file['filename']}")
        
        # Process the stored file
        content_type = existing_file['content_type']
        file_data = existing_file['data']
        
        try:
            if 'csv' in existing_file['filename'].lower():
                if isinstance(file_data, bytes):
                    st.session_state.df = pd.read_csv(io.BytesIO(file_data))
                else:
                    # Handle base64-encoded data
                    decoded_data = base64.b64decode(file_data)
                    st.session_state.df = pd.read_csv(io.BytesIO(decoded_data))
            elif 'xls' in existing_file['filename'].lower():
                if isinstance(file_data, bytes):
                    st.session_state.df = pd.read_excel(io.BytesIO(file_data))
                else:
                    # Handle base64-encoded data
                    decoded_data = base64.b64decode(file_data)
                    st.session_state.df = pd.read_excel(io.BytesIO(decoded_data))
        except Exception as e:
            st.error(f"Error processing stored file: {str(e)}")
            st.session_state.df = None

    # Use df from session state
    df = st.session_state.df

    if df is None:
        # File upload if no existing file or processing failed
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Drag and drop your file here or click to select"
        )
        
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file)
            
            if df is not None:
                # Save the file to the database
                file_content = uploaded_file.getvalue()
                save_file_to_db(
                    user['id'], 
                    uploaded_file.name, 
                    file_content, 
                    uploaded_file.type
                )
                st.session_state.df = df
                st.success(f"File {uploaded_file.name} saved to your account")
    
    if df is not None:
        st.write("### Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        # User input
        user_input = st.text_area(
            "Enter your request here...",
            height=100,
            placeholder="What would you like to visualize?"
        )
        
        if st.button("Generate Visualization"):
            with st.spinner("Generating visualization..."):
                image_data, result_output, chat_history, content_type = create_visualization(
                    df,
                    user_input,
                    st.session_state.chat_history
                )
                
                if image_data:
                    if content_type == "image/png":
                        # Create a container with constrained width for the image
                        col1, col2, col3 = st.columns([1, 3, 1])
                        with col2:
                            # Display the image in the middle column (2/4 of the page width)
                            st.image(
                                f"data:image/png;base64,{image_data}",
                                use_container_width=True,  # Still true but for a narrower container
                                output_format="PNG"
                            )
                    elif content_type == "application/pdf":
                        st.download_button(
                            label="Download PDF",
                            data=image_data,
                            file_name="plot.pdf",
                            mime="application/pdf"
                        )
                
                st.markdown("### Analysis")
                st.markdown(result_output)
                
                # Update chat history in session state and database
                st.session_state.chat_history = chat_history
                save_chat_history(user['id'], chat_history)
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for msg in st.session_state.chat_history:
            if msg["role"] == "human":
                st.info(f"**You**: {msg['content']}")
            else:
                st.success(f"**Assistant**: {msg['content']}")

if __name__ == '__main__':
    main()
