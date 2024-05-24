from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Depends,
    Body,
)
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import os
import json
import requests
import openai
from dotenv import load_dotenv
import backoff
from starlette.middleware.sessions import SessionMiddleware
import logging
from contextlib import asynccontextmanager
from uuid import uuid4
from fastapi.staticfiles import StaticFiles

# Module Docker
from .openai_assistant import assistant_setup, create_political_conversation, create_casual_conversation, chatbot_completion
from .post_data import ChatInput

import sys
sys.path.append('/home/mo/code/deliberation_chatbot/app')
from .log_config import setup_logging  # Ensures logging is configured
import logging

# This retrieves the root logger which was configured in log_config.py
setup_logging()
logger = logging.getLogger("machma_logger")


# Load the .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

assistant_dict = {}

# On Startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This happens just after starting the server
    # We initialize the Assistants API here
    casual_assistant, political_assistant, vector_store = await assistant_setup() 
    assistant_dict['casual_assistant'] = casual_assistant.id
    assistant_dict['political_assistant'] = political_assistant.id
    assistant_dict['vector_store'] = vector_store.id
    logger.info(f"""Created Casual Assistant with ID: {assistant_dict['casual_assistant']}, 
                Political Assistant with ID: {assistant_dict['political_assistant']} 
                & Vector Store with ID: {assistant_dict['vector_store']}""")
    yield
    # This happens just before shutting down the server
    logger.info(f"Shutting down the server")

app = FastAPI(lifespan=lifespan)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"Hello": "World"}

# # Local File run
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# Docker File run
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/")
async def index(request: Request):
    """
    Serves the home page of the application.

    This function provides the user with an initial interface or landing page.

    Args:
        request (Request): The HTTP request object.

    Returns:
        TemplateResponse: The response containing the rendered "index.html" template.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# Create a hello get endpoint that returns a simple json key value pair
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/hello")
async def hello():
    """
    Simple get endpoint that returns a key value pair.

    Returns:
        JSONResponse: The response containing the key value pair.
    """
    return JSONResponse(content={"message": "Hello, World!"})

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/about", summary="Renders the about page.")
async def about(request: Request):
    """
    Renders the about page.

    This function renders the "about.html" template.

    Args:
        request (Request): The request object containing information about the HTTP request.

    Returns:
        TemplateResponse: A TemplateResponse object that renders the "about.html" template.
    """
    return templates.TemplateResponse("about.html", {"request": request})

# Configure Session Middleware
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")
# In-memory session storage
sessions = {}


def get_session_id(request: Request):
    print("Session Details: ", request.session)
    print("Query Params: ", request.query_params)
    if "session_id" in request.query_params:
        return request.query_params["session_id"]
    else:
        if "session_id" not in request.session:
            request.session["session_id"] = str(uuid4())
        return request.session["session_id"]

@backoff.on_exception(backoff.expo, Exception, max_tries=5)    
@app.get("/chat_pol", summary="Initialize or continue a political chat session")
async def get_chat(
    request: Request,
    gender: str = 'male',
    birth_year: int = '1995',
    school_education: str = 'Abitur oder erweiterte Oberschule mit Abschluss 12. Klasse (Hochschulreife)',
    vocational_education: str = 'Hochschulabschluss',
    occupation: str = 'Student/in',
    interest_in_politics: str = 'stark',
    political_concern: str = 'Klimawandel',
    initial_reasoning: str = 'Der Klimawandel gefährdet die Zukunft der Menschheit. Wir sollten zunnächst die co2 emissionen der reichsten vermindern.',
    session_id: str = Depends(get_session_id),
):    

    logger.info("assistant id: ", assistant_dict['political_assistant'])
    logger.info("vector store id: ", assistant_dict['vector_store'])
        
    thread, run, first_message = await create_political_conversation(assistant_dict['political_assistant'], assistant_dict['vector_store'], 
                                                      gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      occupation, 
                                                      interest_in_politics, 
                                                      political_concern, 
                                                      initial_reasoning)  
    
    sessions[session_id] = {
        "chat_history": {"user": [], "bot": [first_message]},
        "thread_id": thread.id,
        "gender": gender,
        "birth_year": birth_year,
        "school_education": school_education,
        "vocational_education": vocational_education,
        "occupation": occupation,
        "interest_in_politics": interest_in_politics,
        "political_concern": political_concern,
        "initial_reasoning": initial_reasoning,
        "is_political": True,
    }
    
    logger.info(f"""Created Assistant with ID: {assistant_dict['political_assistant']} 
                    & Vector Store with ID: {assistant_dict['vector_store']}
                    & Thread with ID: {thread.id}
                    & Run with ID: {run.id}
                    First Message: {first_message}""")
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": sessions[session_id],
            "session_id": session_id,
            "first_message": first_message,
        },
    )

@backoff.on_exception(backoff.expo, Exception, max_tries=5)    
@app.get("/chat_cas", summary="Initialize or continue a casual chat session")
async def get_chat(request: Request,
                   session_id: str = Depends(get_session_id)):
    
    logger.info("casual assistant id: ", assistant_dict['casual_assistant'])

    
    thread, run, first_message = await create_casual_conversation(assistant_dict['casual_assistant'])
    
    sessions[session_id] = {
        "chat_history": {"user": [], "bot": [first_message]},
        "thread_id": thread.id,
        "is_political": False,
    }
    
    logger.info(f"""Created Assistant with ID: {assistant_dict['casual_assistant']} 
                    & Thread with ID: {thread.id}
                    & Run with ID: {run.id}
                    First Message: {first_message}""")
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": sessions[session_id],
            "session_id": session_id,
            "first_message": first_message,
        },
    )

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.post("/chat", summary="Processes user input in a political chat session")
async def post_chat(chat_input: ChatInput = Body(...)):
    """
    Processes and responds to user input in a chat session. Handles storing the user's message, 
    generating a bot response, and updating the chat history.

    ### Parameters:
    - `chat_input`: A model that includes the user's input message and the session ID.
        - `user_input`: The message input by the user.
        - `session_id`: Identifier for the current chat session.

    ### Returns:
    - `JSONResponse`: Contains the updated chat history, including both user and bot messages.

    ### Raises:
    - `HTTPException`: In case of errors during the chatbot completion process.
    """
    logger.debug("chat_input: ", chat_input)
    logger.debug("chat_input.user_input: ", chat_input.user_input)
    logger.debug("Session ID: ", chat_input.session_id)
    logger.debug("Sessions: ", sessions)
    logger.debug("particular Sessions: ", sessions[chat_input.session_id])
    user_input = chat_input.user_input
    session_id = chat_input.session_id

    if session_id not in sessions:
        logger.error("Session ID not in sessions: ", session_id)
        return JSONResponse(status_code=404, content={"message": "Session ID not found."})

    session_data = sessions[session_id]
    chat_history = session_data["chat_history"]
    # Append user message
    chat_history["user"].append(user_input)

    try:
        logger.debug("Calling Chatbot Completion now...")
        bot_response = await chatbot_completion(
            chat_history["user"][-1],
            assistant_dict['political_assistant'],
            session_data["thread_id"],
        )
        logger.debug("Bot Response: ", bot_response)
        logger.info("Bot response: ", bot_response)
        
        # Append bot response
        chat_history["bot"].append(bot_response)

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})

    return JSONResponse(content={"chat_history": chat_history})

# Function to retrieve data from JSON file
async def get_data():
    with open("data/data.json") as f:
        data = json.load(f)
    return data


# Function to send API request to external API
async def send_api_request():
    url = os.getenv("EXTERNAL_API_URL", "https://api.example.com/data")
    response = requests.get(url)
    return response.json()


# Function to process data and return response
async def process_data():
    data = await get_data()
    api_data = await send_api_request()
    # Process data and create response
    response = {"data": data, "api_data": api_data}
    return response


# Route to call functions and return response
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/process_data")
async def get_processed_data():
    try:
        response = await process_data()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
