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


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_political_conversation(gender, birth_year, school_education, vocational_education, occupation, interest_in_politics, political_concern, initial_reasoning):
    logger.info("Political Concern:", political_concern)
    system_prompt = f"""
    Your discussion partner self identifies as
    Gender: {gender}
    Born in: {birth_year}
    Their highest level of general school education ist: {school_education}
    Their highest level of vocational training is: {vocational_education}
    Their occupation is: {occupation}

    Their interest in politics is: {interest_in_politics}
    Their main political concern is: {political_concern}
    And their initial reasoning on why he thinks it's important and how to solve it is: {initial_reasoning}

    Be friendly and introduce yourself.
    Then, refer to their stated political opinion and ask questions that probe the logic and evidence behind their views.
    Introduce counterpoints and allow them to reflect on and reformulate their opinion.
    Use the information about party positions frequently to provide context, examples and party positions on the topic to your discussion partner.
    Make sure to discuss in german.
    """

    conversation_start = [{
            "role": "user",
            "content": system_prompt,
        }]
    return conversation_start

def get_casual_conversation():
    logger.info("Creating casual conversation")
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
    if is_political:
        name = "Socratic Assistant"
        instructions = """You're an AI political guide designed to engage in a socratic maieutic dialogue. Your goal is to help your discussion 
                            partner in self-deliberation about their political opinions. Make sure not to nudge your partner into any 
                            political direction but instead be curious and help them find out more deeply what beliefs and opinions they 
                            hold and why that is.
                            You will be provided with some initial data about your discussion partner's sociodemographic background. You should astutely 
                            use this information to adjust your conversation style and arguments
                            to engage in a more personalized and effective discussion. However, you shall not mention explicitly any of 
                            those sociodemographic characteristics regarding your opponent.
                            You will also be provided with a the political topic that your discussion partner is most concerned about. 
                            And their initial reasoning on why they think it's important and how to solve it. You should start the conversation by
                            refering to this information and then ask follow up questions that probe the logic and evidence behind their views. 
                            Introduce counterpoints and standpoints from political partys and allow them to reflect on and reformulate their opinion.
                            Additionally you will have access to a vector store containing information about party positions on various topics.
                            You should use this information frequently to provide context, examples and party positions on the topic to your discussion partner. 
                            Do not mark your citations.
                            Make sure to discuss in german and make sure not to overwhelm your discussion partner with too many questions or too much information.
                            Limit yourself to at most 3 questions or statements at a time, preferably only 1 or 2.
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
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Socratic Assistant":
                    logger.info("Socratic Assistant found.")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Socratic Assistant found within, creating a new one...")
            assistant = await create_assistant(is_political=True)
            return assistant
        else:
            logger.info("No assistants or data available or failed to fetch data, attempting to create a new assistant.")
            assistant = await create_assistant(is_political=True)
            return assistant
    except Exception as e:
        logger.info("Failed to fetch assistants:", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch assistants: {str(e)}")
    
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_casual_assistant():
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Casual Assistant":
                    logger.info("Casual Assistant found.")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Casual Assistant found within, creating a new one...")
            assistant = await create_assistant(is_political=False)
            return assistant
        else:
            logger.info("No assistants or data available or failed to fetch data, attempting to create a new assistant.")
            assistant = await create_assistant(is_political=False)
            return assistant
    except Exception as e:
        logger.info("Failed to fetch assistants:", e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch assistants: {str(e)}")


# Helper function to ensure the vector store is attached to the assistant
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def ensure_vector_store(assistant):
    try:
        # Check if the current assistant has the correct vector store attached
        vector_store_ids = assistant.tool_resources.file_search.vector_store_ids if assistant.tool_resources and assistant.tool_resources.file_search else []

        if vector_store_ids:
            # Fetch the vector store details
            vector_store = client.beta.vector_stores.retrieve(vector_store_id=vector_store_ids[0])
            if vector_store and vector_store.name == "Party Positions":
                logger.info("Correct vector store already attached with ID: {vector_store.id}")
                return vector_store
            else:
                logger.info("Incorrect vector store attached, looking for correct vector store.")
        else:
            logger.info("No vector store attached, looking for correct vector store.")

        # If the correct vector store is not attached, check if such a store exists
        all_stores = client.beta.vector_stores.list()
        party_positions_store = next((store for store in all_stores.data if store.name == "Party Positions"), None)

        if party_positions_store:
            vector_store_id = party_positions_store.id
            logger.info(f"Vector store 'Party Positions' found with ID: {vector_store_id}")
        else:
            # Create a new vector store if not found
            logger.info("Creating new vector store 'Party Positions'")
            vector_store = client.beta.vector_stores.create(name="Party Positions")
            vector_store_id = vector_store.id

            # Upload files to the new vector store
            file_paths = ["data/party_positions.pdf"]
            file_streams = [open(path, "rb") for path in file_paths]
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id, files=file_streams
            )
            logger.info(f"Files uploaded to vector store: {file_batch.status}")

        # Attach the vector store to the assistant
        client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        logger.info(f"Vector store '{vector_store_id}' attached to assistant.")
        return vector_store

    except Exception as e:
        logger.error("Failed to ensure vector store:", e)
        raise HTTPException(status_code=500, detail=f"Failed to ensure vector store: {str(e)}")
    
# Helper Function to create a new conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_political_conversation(assistant, vector_store, gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      occupation, 
                                                      interest_in_politics, 
                                                      political_concern, 
                                                      initial_reasoning):
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=get_political_conversation(gender, 
                                                      birth_year, 
                                                      school_education, 
                                                      vocational_education, 
                                                      occupation, 
                                                      interest_in_politics, 
                                                      political_concern, 
                                                      initial_reasoning),
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
    
# Helper Function to create a new conversation thread
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_casual_conversation(assistant):
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
    logger.debug("political chatbot completion called")
    logger.debug(user_message)
    logger.debug(assistant)
    logger.debug(thread)
    try:
        logger.debug("political chatbot completion try block starting...")
        # Create a message to append to our thread
        bot_message = client.beta.threads.messages.create(
            thread_id=thread, role='user', content=user_message)
        logger.debug(bot_message)
        # Execute our run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread,
            assistant_id=assistant,
        )
        logger.debug(run)
        response = client.beta.threads.messages.list(thread_id=thread, run_id=run.id)
        logger.debug(response)
        logger.debug(response.data[0].content[0].text.value)
        return response.data[0].content[0].text.value
    except Exception as e:
        error_message = f"Error occurred in chatbot_completion: {str(e)}"
        raise HTTPException(status_code=500, detail=error_message)
    
    