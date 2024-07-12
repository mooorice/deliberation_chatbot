import asyncio
from openai import OpenAI
import backoff
from dotenv import load_dotenv
import os
from fastapi import HTTPException
import sys
sys.path.append('/home/mo/code/deliberation_chatbot/app')

from .log_config import setup_logging  # Ensures logging is configured
import logging

# This retrieves the root logger which was configured in log_config.py
setup_logging()
logger = logging.getLogger("machma_logger")

# Load the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Helper function to create a political conversation
def get_political_conversation(gender, birth_year, school_education, vocational_education, interest_in_politics, political_concern):
    """
    Takes sociodemographic data and political concern as input.
   
    Inserts the provided sociodemographic data and political concern into a conversation starter.
    
    Returns a conversation starter for a political discussion.
    """
    
    logger.info("Creating political conversation...")

    system_prompt = f"""
    Your discussion partner self identifies as
    Gender: {gender}
    Born in: {birth_year}
    Their highest level of general school education ist: {school_education}
    Their highest level of vocational training is: {vocational_education}
    Their interest in politics is: {interest_in_politics}
    Their main political concern is: {political_concern}
    
    Use this information to tailor your conversation style and arguments, but do not mention these characteristics explicitly. 

    Be friendly and introduce yourself as an AI Assistant designed to help discuss political issues.
    
    Then, refer to their stated main political concern and start asking critical questions that probe the logic and evidence behind their views.
    Focus on the underlying values, assumptions and perspectives that inform their political concern. Ask why they hold their views.
    Always refer to political party positions from your knowledge base to provide context and counter arguments.
    Avoid getting stuck on one aspect; cover a broad range of aspects.
    
    Given the context of the upcoming European elections, refer to the functions and competences of the European Parliament when relevant. 
    
    Be concise. Limit yourself to one question or statement at a time.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start

# Helper function to create a casual conversation
def get_casual_conversation():
    """
    Returns a conversation starter for a casual discussion.
    """
    
    logger.info("Creating casual conversation...")
    
    system_prompt = f"""
    Be friendly, introduce yourself, and to use all your knowledge of non-political subjects to stimulate and engage your discussion partner. 
    Remember to engage and stimulate your conversation partner. Speak in German using a clear language adapted to your discussion partner, 
    gracefully avoid and redirect any political issue. Keep your messages moderately concise.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start

# Helper function to create a question conversation
def get_question_conversation():
    """
    Returns a conversation starter for a question discussion.
    """
    
    logger.info("Creating question conversation...")
    
    system_prompt = f"""You're an AI designed to check the quality of the answers provided by another assistant.
                            Your only job is to check that there is at most two questions in every message. 
                            If there is more than two questions, pick the most critical questions or combine the questions to maximize stimulation of critical thinking.
                            That is, to question the assumptions, evidence, and logic behind the question.
                            Do not change the rest of the message. Never remove information that is not a question.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start

# Helper function to create a question conversation
def get_question_conversation():
    """
    Returns a conversation starter for a question discussion.
    """
    
    logger.info("Creating question conversation...")
    
    system_prompt = f"""You're an AI designed to check the quality of the answers provided by another assistant.
                            Your only job is to check that there is at most two questions in every message. 
                            If there is more than two questions, pick the most critical questions or combine the questions to maximize stimulation of critical thinking.
                            That is, to question the assumptions, evidence, and logic behind the question.
                            Do not change the rest of the message. Never remove information that is not a question.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start


# Helper function to create an assistant, if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_assistant(type):
    """
    Takes a boolean flag to determine the type of assistant to create.
    
    Creates a new assistant based on the provided type.
    
    Returns the created assistant.
    """
    
    
    if type == "political":
        name = "Political Assistant"
        instructions = """You're an AI political guide designed to engage in a Socratic maieutic dialogue. 
                            Your goal is to help your discussion partner explore and reflect on their political opinions without nudging them in any direction. 

                            Ask critical questions that probe the logic and evidence behind the user's views. 
                            Always add counterpoints and perspectives from your knowledge base of political party programs. 
                            Avoid getting stuck on one aspect; cover a broad range of aspects.

                            Conduct the discussion in German, using the formal "Sie" form, unless prompted otherwise. 
                            Keep the conversation friendly and respectful. 
                            """
    elif type == "casual":
        name = "Casual Assistant"
        instructions = """You're an AI designed to engage in exciting, stimulating, and informative, non-political conversations with your discussion partner. 
                            Your goal is to keep your discussion partner engaged while avoiding any political topic. Make sure to be friendly, introduce yourself, 
                            and to ask them how their day has been. Use all your knowledge of non-political subjects to stimulate and engage your discussion partner. 
                            For instance you may: 
                            1. Start asking your discussion partner about their interests and then talk about things they care about, 
                            or propose topics yourself if they do not want to mention personal hobbies or interests; 
                            2. Create interesting and unexpected connections and provide intriguing facts on their interests and 
                            varied subjects like science, food, traveling, technology, health, sports, arts, entertainment...; 
                            3. Let the discussion partner dream suggesting thought experiments on their interest and general interest topics to create a spark of enthusiasm. 
                            Important things to consider are: 
                            1. You have to speak German in the conversation, unless prompted otherwise by your discussion partner; 
                            2. You have to make sure not to overwhelm your discussion partner with too many questions or too much information, 
                            limiting yourself to 1 question per message; 
                            3. Avoid all political discussions and maintain a neutral stance on any issue; 
                            4. Use simple and clear language that resonates to the average German person; 
                            5. Craft each message to be effective and concise.    
                            """    
    elif type == "question":
        name = "Question Assistant"
        instructions = """You're an AI designed to check the quality of the answers provided by another assistant.
                            Your only job is to check that there is at most two questions in every message. 
                            If there is more than two questions, pick the most critical questions or combine the questions to maximize stimulation of critical thinking.
                            That is, to question the assumptions, evidence, and logic behind the question.
                            Do not change the rest of the message. Never remove information that is not a question.
                            """
    try:
        # Making a call to create an assistant
        assistant = client.beta.assistants.create(
            name=name,
            instructions=instructions,
            model="gpt-4o",
            tools=[{"type": "file_search"}],
        )
        logger.info(f"Assistant created with ID: {assistant.id}")
        return assistant
    except Exception as e:
        logger.info("Failed to create assistant:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")

# Helper function to get an assistant, returns the ID of the Political Analyst Assistant if found 
# Creates a new Political Assistant if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_political_assistant():
    """
    Checks if a political assistant exists, and returns it if found.
    If not, creates a new political assistant.
    
    Returns the political assistant.
    """
    
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Political Assistant":
                    logger.info(f"Political Assistant found with ID: {assistant.id}")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Political Assistant found in the list of assistants, creating a new one...")
            assistant = await create_assistant(type="political")
            return assistant
        else:
            logger.info("No assistants available or failed to fetch data, creating a new assistant...")
            assistant = await create_assistant(type="political")
            return assistant
    except Exception as e:
        logger.info("Failed to fetch assistants:", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch assistants: {str(e)}")
    
# Helper function to get an assistant, returns the ID of the Casual Assistant if found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_casual_assistant():
    """
    Checks if a casual assistant exists, and returns it if found.
    If not, creates a new casual assistant.
    
    Returns the casual assistant.
    """
    
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Casual Assistant":
                    logger.info(f"Casual Assistant found with ID: {assistant.id}")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Casual Assistant found in the list of assistants, creating a new one...")
            assistant = await create_assistant(type="casual")
            return assistant
        else:
            logger.info("No assistants available or failed to fetch data, creating a new assistant...")
            assistant = await create_assistant(type="casual")
            return assistant
    except Exception as e:
        logger.info("Failed to fetch assistants:", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch assistants: {str(e)}")
    
# Helper function to get an assistant, returns the ID of the Question Assistant if found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_question_assistant():
    """
    Checks if a question assistant exists, and returns it if found.
    If not, creates a new question assistant.
    
    Returns the question assistant.
    """
    
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Question Assistant":
                    logger.info(f"Question Assistant found with ID: {assistant.id}")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Question Assistant found in the list of assistants, creating a new one...")
            assistant = await create_assistant(type="question")
            return assistant
        else:
            logger.info("No assistants available or failed to fetch data, creating a new assistant...")
            assistant = await create_assistant(type="question")
            return assistant
    except Exception as e:
        logger.info("Failed to fetch assistants:", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch assistants: {str(e)}")

# Helper function to ensure the vector store is attached to the assistant
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def ensure_vector_store(assistant):
    """
    Takes an assistant and ensures that the correct vector store is attached to it.
    
    Returns the vector store.     
    """
    try:
        # Check if the current assistant has the correct vector store attached
        vector_store_ids = assistant.tool_resources.file_search.vector_store_ids if assistant.tool_resources and assistant.tool_resources.file_search else []
        logger.debug("Vector Store IDs: %s", vector_store_ids)
        if vector_store_ids:
            # Fetch the vector store details
            vector_store = client.beta.vector_stores.retrieve(vector_store_id=vector_store_ids[0])
            if vector_store and vector_store.name == "Party Programs":
                logger.info(f"Correct vector store already attached with ID: {vector_store.id}")
                return vector_store
            else:
                logger.info("Incorrect vector store attached, looking for correct vector store.")
        else:
            logger.info("No vector store attached, looking for correct vector store.")

        # If the correct vector store is not attached, check if such a store exists
        all_stores = client.beta.vector_stores.list()
        party_programs_store = next((store for store in all_stores.data if store.name == "Party Programs"), None)
        
        if party_programs_store:
            vector_store = party_programs_store
            logger.info(f"Vector store 'Party Programs' found with ID: {vector_store.id}")
        else:
            # Create a new vector store if not found
            logger.info("Creating new vector store 'Party Programs'")
            vector_store = client.beta.vector_stores.create(name="Party Programs")

            # Upload files to the vector store
            directory = 'app/data'
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory {directory} does not exist")
            file_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pdf')]
            if not file_paths:
                raise FileNotFoundError(f"No PDF files found in directory {directory}")
            
            # Open file streams
            file_streams = [open(path, "rb") for path in file_paths]
            try:
                file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store.id, files=file_streams
                )
                logger.info(f"Files uploaded to vector store: {file_batch.status}")
            finally:
                # Ensure all file streams are closed
                for file_stream in file_streams:
                    file_stream.close()

        # Attach the vector store to the assistant
        client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
        )
        logger.info(f"Vector store '{vector_store.id}' attached to assistant.")
        return vector_store

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=f"File not found: {e}")
    except Exception as e:
        logger.error(f"Failed to ensure vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ensure vector store: {str(e)}")
    
# Helper Function to create a new political conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_political_conversation(assistant, vector_store, gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      interest_in_politics, 
                                                      political_concern 
                                                      ):
    """
    Takes assistant, vector store, and sociodemographic data as input.
    
    Creates a new political conversation thread with the provided data.
    
    Returns the created thread and first message.
    """
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=get_political_conversation(gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      interest_in_politics, 
                                                      political_concern 
                                                      ),
                            tool_resources={
                                "file_search": {
                                    "vector_store_ids": [vector_store]
                                }
                            }
                        )
        logger.info(f"Political conversation thread created with ID: {thread.id}")
        
        # Create and poll the run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant
        )
        logger.debug(f"Political conversation run created with ID: {run.id}")
        if run.status == 'completed': 
            logger.debug("Run status is completed") 
            response = client.beta.threads.messages.list(
                thread_id=thread.id)
        else:
            logger.info(f"Run status is not completed: {run.status}")
            raise HTTPException(status_code=500, detail="Run did not complete in the expected time frame.")
        # response = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        logger.debug("response: %s", response)
        first_message = response.data[0].content[0].text.value
        logger.debug("first_message: %s", first_message)
        return thread, first_message
    except Exception as e:
        logger.error("Failed to create political conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create political conversation: {str(e)}")
    
# Helper Function to create a new casual conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_casual_conversation(assistant):
    """
    Takes assistant as input.
    
    Creates a new casual conversation thread.
    
    Returns the created thread and first message.
    """
    
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=get_casual_conversation(),
                            )
        logger.info(f"Casual conversation thread created with ID: {thread.id}")
        
        # Create and poll the run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant
        )
        logger.debug(f"Casual conversation run created with ID: {run.id}")
        if run.status == 'completed': 
            logger.debug("Run status is completed") 
            response = client.beta.threads.messages.list(
                thread_id=thread.id)
        else:
            logger.info(f"Run status is not completed: {run.status}")
            raise HTTPException(status_code=500, detail="Run did not complete in the expected time frame.")
        # response = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        logger.debug("response: %s", response)
        first_message = response.data[0].content[0].text.value
        logger.debug("first_message: %s", first_message)
        return thread, first_message
    except Exception as e:
        logger.error("Failed to create casual conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create casual conversation: {str(e)}")
    
# Helper Function to create a new question conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_question_thread(assistant):
    """
    Takes assistant as input.
    
    Creates a new question conversation thread.
    
    Returns the created thread.
    """
    
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=get_question_conversation(),
                            )
        logger.info(f"Question conversation thread created with ID: {thread.id}")
        logger.debug("Question thread created with ID: %s", thread.id)
        logger.debug("Question Assistant ID: %s", assistant)
        # Create and poll the run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant
        )
        logger.debug("run created with id: %s", run.id)
        client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        logger.info(f"Question conversation run created with ID: {run.id}")
        return thread
    except Exception as e:
        logger.error("Failed to create question conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create question conversation: {str(e)}")
    
# Main function to run the setup
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def assistant_setup():
    """
    Main function to set up the assistants and vector store.
    """
    try:
        # Step 1: Get or create the casual assistant
        casual_assistant = await get_casual_assistant()
        
        # Step 2: Get or create the political assistant
        political_assistant = await get_political_assistant()
        
        # Step 3: Get or create the question assistant
        question_assistant = await get_question_assistant()

        # Step 4: Ensure the assistant has the correct vector store attached
        vector_store = await ensure_vector_store(political_assistant)
        
        return casual_assistant, political_assistant, question_assistant, vector_store

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main Chatbot Completion Function for assistants API
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def chatbot_completion(
    user_message,
    assistant,
    thread,
    ):
    """
    Takes user message, assistant ID, and thread ID as input.
    
    Gets chatbot answer based on the provided input.
    
    Returns the bot response.
    """
    logger.debug(user_message)
    logger.debug(assistant)
    logger.debug(thread)
    try:
        # Create a message to append to our thread
        bot_message = client.beta.threads.messages.create(
            thread_id=thread, role='user', content=user_message)
        logger.info(f"Bot message received: {bot_message}")
        # Execute our run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread,
            assistant_id=assistant,
        )
        response = client.beta.threads.messages.list(thread_id=thread, run_id=run.id)
        return response.data[0].content[0].text.value
    except Exception as e:
        error_message = f"Error occurred in chatbot_completion: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
    
    