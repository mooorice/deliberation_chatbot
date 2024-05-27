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
import re

# Module Docker
from .openai_assistant import assistant_setup, create_political_conversation, create_casual_conversation, create_question_thread, chatbot_completion
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

# Initialize the Assistants Dictionary to store the assistant IDs
assistant_dict = {}

# On Startup
# We initialize our Assistants and Vector store here
@asynccontextmanager
async def lifespan(app: FastAPI):
    casual_assistant, political_assistant, question_assistant, vector_store = await assistant_setup() 
    question_thread = await create_question_thread(question_assistant.id)
    assistant_dict['casual_assistant'] = casual_assistant.id
    assistant_dict['political_assistant'] = political_assistant.id
    assistant_dict['question_assistant'] = question_assistant.id
    assistant_dict['questions_thread_id'] = question_thread.id
    assistant_dict['vector_store'] = vector_store.id
    logger.info(f"""
                Created Political Assistant with ID: {assistant_dict['political_assistant']}, 
                Casual Assistant with ID: {assistant_dict['casual_assistant']}, 
                Question Assistant with ID: {assistant_dict['question_assistant']},
                & Vector Store with ID: {assistant_dict['vector_store']}
                """)
    yield
    # This happens just before shutting down the server
    logger.info(f"Shutting down the server")

app = FastAPI(lifespan=lifespan)

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"Hello": "World"}

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

# Helper function to get the session ID from the query parameters or create a new session ID
def get_session_id(request: Request):
    logger.debug("Session Details: %s", request.session)
    logger.debug("Query Params: %s", request.query_params)
    if "session_id" in request.query_params:
        return request.query_params["session_id"]
    else:
        if "session_id" not in request.session:
            request.session["session_id"] = str(uuid4())
        return request.session["session_id"]

# Create a chat endpoint that initializes a chat session
@backoff.on_exception(backoff.expo, Exception, max_tries=5)    
@app.get("/chat", summary="Initialize or continue a political chat session")
async def get_chat(
    request: Request,
    session_id: str = Depends(get_session_id),
    treatment: bool = True,
    gender: str = 'keine angabe',
    birth_year: int = 'keine angabe',
    school_education: str = 'keine angabe',
    vocational_education: str = 'keine angabe',
    interest_in_politics: str = 'keine angabe',
    political_concern: str = 'keine angabe',):    
    """
    Takes input from the user and initializes a chat session. The function creates a new chat session.
    The chat session is created based on the treatment type. If the treatment type is set to True, the function
    creates a political chat session. If the treatment type is set to False, the function creates a casual chat session.
    
    ### Parameters:
    - `request`: The HTTP request object.
    - `session_id`: The unique identifier for the chat session.
    - `treatment`: A boolean value that determines the type of chat session to create.
    - socio-demographic and political information from the survey
    
    ### Returns:
    - `TemplateResponse`: The response containing the rendered "chat.html" template.
    """
    
    # Create a new chat session according to the treatment type
    if not treatment:
        thread, first_message = await create_casual_conversation(assistant_dict['casual_assistant'])
    if treatment:
        thread, first_message = await create_political_conversation(assistant_dict['political_assistant'], assistant_dict['vector_store'], 
                                                        gender, 
                                                        birth_year, 
                                                        school_education, 
                                                        vocational_education, 
                                                        interest_in_politics, 
                                                        political_concern 
                                                        )  
        
    logger.info("Chat session created on thread %s", thread.id)
    
    # Store the chat session data
    sessions[session_id] = {
        "chat_history": {"user": [], "bot": [first_message]},
        "treatment": treatment,
        "thread_id": thread.id,
        "gender": gender,
        "birth_year": birth_year,
        "school_education": school_education,
        "vocational_education": vocational_education,
        "interest_in_politics": interest_in_politics,
        "political_concern": political_concern,
        }
    
    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "chat_history": sessions[session_id],
            "session_id": session_id,
            "first_message": first_message,
        },
    )

# Create a chat endpoint that continues a chat session
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
    logger.debug("post endpoint reached")
    logger.debug("chat_input: %s", chat_input)
    logger.debug("chat_input.user_input: %s", chat_input.user_input)
    logger.debug("Session ID: %s", chat_input.session_id)
    logger.debug("Sessions: %s", sessions)
    
    user_input = chat_input.user_input
    session_id = chat_input.session_id

    if session_id not in sessions:
        logger.error("Session ID not in sessions: %s", session_id)
        return JSONResponse(status_code=404, content={"message": "Session ID not found."})

    session_data = sessions[session_id]
    chat_history = session_data["chat_history"]
    
    # Append user message
    chat_history["user"].append(user_input)

    try:        
        # Determine which assistant to use based on session_data["treatment"]
        assistant_type = assistant_dict['political_assistant'] if session_data["treatment"] else assistant_dict['casual_assistant']
        
        logger.info("Message input: %s", chat_history["user"][-1])
        # Get response
        bot_response = await chatbot_completion(
            chat_history["user"][-1],
            assistant_type,
            session_data["thread_id"],
        )
        logger.info("Bot response: %s", bot_response)
        
        # short_response = await chatbot_completion(
        #     bot_response,
        #     assistant_dict["question_assistant"],
        #     assistant_dict["questions_thread_id"],
        # )
        
        logger.info("short response: %s", bot_response)
        # Remove all citation markings and the text in between from bot_response
        bot_response_cleaned = re.sub(r'【[^】]*】', '', bot_response)

        # Check if this is the 5th bot message and add thank you message
        if len(chat_history['bot']) == 5: 
            thank_you_message = "<p>Vielen Dank für diese spannende Unterhaltung! Sie können nun mit der Umfrage fortfahren. Wenn Sie möchten, können wir aber auch gerne noch weiter diskutieren."
            bot_response_cleaned += " " + thank_you_message
        
        # Append bot response
        chat_history["bot"].append(bot_response_cleaned)

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"message": e.detail})

    return JSONResponse(content={"chat_history": chat_history})
