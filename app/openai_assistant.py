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
def get_political_conversation(gender, birth_year, school_education, vocational_education, occupation, interest_in_politics, political_concern):
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
    Their occupation is: {occupation}

    Their interest in politics is: {interest_in_politics}
    Their main political concern is: {political_concern}
    
    Use this information to tailor your conversation style and arguments, but do not mention these characteristics explicitly. 

    Be friendly and introduce yourself as an AI Assistant designed to help discuss political issues.
    Then, refer to their stated main political concern and ask questions that probe the logic and evidence behind their views.
    Introduce counterpoints and allow them to reflect on and reformulate their opinion.
    Avoid getting stuck on one aspect; cover a broad range of aspects.
    Follow socratic maieutic dialogue principles to help them explore and reflect on their political opinions without nudging them in any direction.
    
    Use the vector store data on party positions frequently to provide context, examples and party positions on the topic to your discussion partner.
    Given the context of the upcoming European elections, refer to the functions and competences of the European Parliament when relevant. 
    
    Conduct the discussion in German, using the formal "Sie" form, unless prompted otherwise. 
    Keep the conversation friendly and respectful. Avoid overwhelming your partner with too many questions or too much information. 
    Limit yourself to one question or statement at a time.
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
    Be friendly and introduce yourself. Have nice and fun conversation with your conversation partner. Ask about their day and interests. 
    Share some fun facts or jokes and ask about their opinion on various topics.
    Do not talk about politics, instead politely change the topic if it comes up.
    Make sure to discuss in german and make sure not to overwhelm your discussion partner with too many questions or too much information. 
    Limit yourself to at most 3 questions or statements at a time, preferably only 1 or 2.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start


# Helper function to create an assistant, if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_assistant(is_political):
    """
    Takes a boolean flag to determine the type of assistant to create.
    
    Creates a new assistant based on the provided type.
    
    Returns the created assistant.
    """
    
    
    if is_political:
        name = "Political Assistant"
        instructions = """You're an AI political guide designed to engage in a Socratic maieutic dialogue. 
                            Your goal is to help your discussion partner explore and reflect on their political opinions without nudging them in any direction. 
                            Be curious and assist them in understanding their beliefs and the reasons behind them.

                            You'll receive initial data about your discussion partner's sociodemographic background. 
                            Use this information to tailor your conversation style and arguments, but do not mention these characteristics explicitly. 
                            You'll also be provided with the political topic of greatest concern to your partner.

                            Start the conversation by introducing yourself as a political AI assistant and mentioning the provided topic. 
                            Ask follow-up questions that probe the logic and evidence behind their views. 
                            Introduce counterpoints and perspectives from political parties to facilitate reflection and reformulation of their opinions. 
                            Avoid getting stuck on one aspect; cover a broad range of aspects.

                            Given the context of the upcoming European elections, refer to the functions and competences of the European Parliament when relevant. 
                            Utilize a vector store containing information about party positions to provide context, examples, and party viewpoints.

                            Conduct the discussion in German, using the formal "Sie" form, unless prompted otherwise. 
                            Keep the conversation friendly and respectful. Avoid overwhelming your partner with too many questions or too much information. 
                            Limit yourself to one question or statement at a time.
                            """
    else:
        name = "Casual Assistant"
        instructions = """You're an AI designed to engage in casual conversation with your discussion partner. Your goal is to have a light-hearted 
                            and fun conversation with your partner. Make sure to be friendly and introduce yourself. Ask about their day and interests. 
                            Share some fun facts or jokes and ask about their opinion on various topics. Make sure to discuss in german and make sure 
                            not to overwhelm your discussion partner with too many questions or too much information. Limit yourself to at most 3 questions 
                            or statements at a time, preferably only 1 or 2.
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
            assistant = await create_assistant(is_political=True)
            return assistant
        else:
            logger.info("No assistants available or failed to fetch data, creating a new assistant...")
            assistant = await create_assistant(is_political=True)
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
            assistant = await create_assistant(is_political=False)
            return assistant
        else:
            logger.info("No assistants available or failed to fetch data, creating a new assistant...")
            assistant = await create_assistant(is_political=False)
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
            vector_store_id = party_programs_store.id
            logger.info(f"Vector store 'Party Programs' found with ID: {vector_store_id}")
        else:
            # Create a new vector store if not found
            logger.info("Creating new vector store 'Party Programs'")
            vector_store = client.beta.vector_stores.create(name="Party Programs")
            vector_store_id = vector_store.id

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
                vector_store_id=vector_store_id, files=file_streams
            )
            logger.info(f"Files uploaded to vector store: {file_batch.status}")
        finally:
            # Ensure all file streams are closed
            for file_stream in file_streams:
                file_stream.close()

        # Attach the vector store to the assistant
        client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        logger.info(f"Vector store '{vector_store_id}' attached to assistant.")
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
                                                      occupation, 
                                                      interest_in_politics, 
                                                      political_concern 
                                                      ):
    """
    Takes assistant, vector store, and sociodemographic data as input.
    
    Creates a new political conversation thread with the provided data.
    
    Returns the created thread, run, and first message.
    """
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=get_political_conversation(gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      occupation, 
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
        response = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        first_message = response.data[0].content[0].text.value
        logger.info(f"Political conversation run created with ID: {run.id}")
        return thread, run, first_message
    except Exception as e:
        logger.error("Failed to create political conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create political conversation: {str(e)}")
    
# Helper Function to create a new casual conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_casual_conversation(assistant):
    """
    Takes assistant as input.
    
    Creates a new casual conversation thread.
    
    Returns the created thread, run, and first message.
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
        response = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        first_message = response.data[0].content[0].text.value
        logger.info(f"Casual conversation run created with ID: {run.id}")
        return thread, run, first_message
    except Exception as e:
        logger.error("Failed to create casual conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create casual conversation: {str(e)}")
    
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

        # Step 2: Ensure the assistant has the correct vector store attached
        vector_store = await ensure_vector_store(political_assistant)
        
        return casual_assistant, political_assistant, vector_store

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
    
    