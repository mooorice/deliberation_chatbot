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

system_prompt = """
You're an AI political guide designed to engage in Socratic dialogue. Your goal is to help your discussion 
partner in self-deliberation about their political opinions. Make sure not to nudge your partner into any 
political direction but instead be curious and help them find out more deeply what beliefs and opinions they 
hold and why that is.

Begin the conversation with a greeting and and invitation to the discussion partner to share the political 
topics that concern them the most. Prompt them to select one topic to delve into first.

Opening Inquiry: Start with an open-ended question to explore their initial thoughts about the chosen topic:
'What concerns you most about this topic and why do you think it's important?'

Deepening Understanding: Once they respond, guide them deeper into their reasoning. Ask questions that probe
the logic and evidence behind their views: 'What makes you believe that this is the best approach? Can you 
share examples or evidence that support your opinion?'

Introducing Challenges: After understanding their argument, introduce a counterpoint or challenge to their 
view to test the robustness of their reasoning: 'Have you considered [a specific contradiction or different 
perspective]? How does this aspect affect your viewpoint?'

Reformulating Opinion: Encourage them to reflect on the counterpoint and adjust their opinion if necessary: 
'Given this new information, how might you refine your perspective on [Topic]?'

Continuation or Change: Before moving on, ask if they want to delve deeper into the same topic or switch to 
another concern: 'Would you like to explore this topic further, or shall we discuss another one of your 
concerns?'

Repeat these steps for each topic they are concerned about. Once all topics are discussed, conclude by 
asking if there are any additional topics they wish to explore or if they have any final thoughts to share.

This approach ensures a thorough, reflective conversation, helping your discussion partner critically 
examine and potentially broaden and deepen their political perspectives.
"""


conversation_start = [{
        "role": "user",
        "content": system_prompt,
    }]

# Helper function to create an assistant, if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_assistant():
    try:
        # Making a call to create an assistant
        assistant = client.beta.assistants.create(
            name="Party Analyst Assistant",
            instructions="You are an expert political analyst. Use your knowledge base to answer questions about party positions.",
            model="gpt-4-turbo",
            tools=[{"type": "file_search"}],
        )
        logger.info(f"Assistant created with ID: {assistant.id}")
        return assistant
    except Exception as e:
        logger.info("Failed to create assistant:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")

# Helper function to get an assistant, returns the ID of the Political Analyst Assistant if found 
# Creates a new Political Analyst Assistant if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_assistant():
    try:
        # Making a call to list assistantsassistant
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Party Analyst Assistant":
                    logger.info("Party Analyst Assistant found.")
                    return assistant
            # If no assistant found in the loop, create a new one
            logger.info("No Party Analyst Assistant found within, creating a new one...")
            assistant = await create_assistant()
            return assistant
        else:
            logger.info("No assistants or data available or failed to fetch data, attempting to create a new assistant.")
            assistant = await create_assistant()
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

            # Upload files to the new vector store
            file_paths = ["data/party_positions.pdf"]
            file_streams = [open(path, "rb") for path in file_paths]
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store.id, files=file_streams
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
async def create_conversation(assistant, vector_store):
    try:
        # Create a new conversation thread
        thread = client.beta.threads.create(
                            messages=conversation_start,
                            tool_resources={
                                "file_search": {
                                    "vector_store_ids": [vector_store]
                                }
                            }
                        )
        logger.info(f"Conversation thread created with ID: {thread.id}")
        
        # Create and poll the run
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant
        )
        response = client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
        first_message = response.data[0].content[0].text.value
        logger.info(f"Conversation run created with ID: {run.id}")
        return thread, run, first_message
    except Exception as e:
        logger.error("Failed to create conversation:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create conversation: {str(e)}")
    
# Main function to run the setup
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def assistant_setup():
    try:
        # Step 1: Get or create the assistant
        assistant = await get_assistant()

        # Step 2: Ensure the assistant has the correct vector store attached
        vector_store = await ensure_vector_store(assistant)
        
        return assistant, vector_store

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Main Chatbot Completion Function for assistants API
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def chatbot_completion(
    user_message,
    assistant,
    thread,
    ):
    logger.debug("chatbot completion called")
    logger.debug(user_message)
    logger.debug(assistant)
    logger.debug(thread)
    try:
        logger.debug("chatbot completion try block starting...")
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
    
    