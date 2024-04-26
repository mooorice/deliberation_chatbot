from openai import OpenAI
import backoff
from dotenv import load_dotenv
import os
from fastapi import HTTPException

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        print(f"Assistant created with ID: {assistant.id}")
        return assistant
    except Exception as e:
        print("Failed to create assistant:", e)
        raise HTTPException(status_code=500, detail=f"Failed to create assistant: {str(e)}")

# Helper function to get an assistant, returns the ID of the Political Analyst Assistant if found and creates a new one if not found
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_assistant():
    try:
        # Making a call to list assistants
        response = client.beta.assistants.list()  # Fetching all the assistants
        if response and response.data:
            for assistant in response.data:
                if assistant.name == "Party Analyst Assistant":
                    print("Party Analyst Assistant found.")
                    return assistant
            # If no assistant found in the loop, create a new one
            print("No Party Analyst Assistant found within, creating a new one...")
            return create_assistant()
        else:
            print("No assistants or data available or failed to fetch data, attempting to create a new assistant.")
            return create_assistant()
    except Exception as e:
        print("Failed to fetch assistants:", e)
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
                print("Correct vector store already attached.")
                return vector_store_ids[0]
            else:
                print("Incorrect vector store attached, looking for correct vector store.")
        else:
            print("No vector store attached, looking for correct vector store.")

        # If the correct vector store is not attached, check if such a store exists
        all_stores = client.beta.vector_stores.list()
        party_positions_store = next((store for store in all_stores.data if store.name == "Party Positions"), None)

        if party_positions_store:
            vector_store_id = party_positions_store.id
            print(f"Vector store 'Party Positions' found with ID: {vector_store_id}")
        else:
            # Create a new vector store if not found
            print("Creating new vector store 'Party Positions'")
            vector_store = client.beta.vector_stores.create(name="Party Positions")
            vector_store_id = vector_store.id

            # Upload files to the new vector store
            file_paths = ["data/party_positions.pdf"]
            file_streams = [open(path, "rb") for path in file_paths]
            file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id, files=file_streams
            )
            print(f"Files uploaded to vector store: {file_batch.status}")

        # Attach the vector store to the assistant
        updated_assistant = client.beta.assistants.update(
            assistant_id=assistant.id,
            tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
        )
        print(f"Vector store '{vector_store_id}' attached to assistant.")
        return vector_store_id

    except Exception as e:
        print("Failed to ensure vector store:", e)
        raise HTTPException(status_code=500, detail=f"Failed to ensure vector store: {str(e)}")
    
# Main function to run the setup
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def assistant_setup():
    try:
        # Step 1: Get or create the assistant
        assistant = await get_assistant()
        print(f"Assistant retrieved or created with ID: {assistant.id}")

        # Step 2: Ensure the assistant has the correct vector store attached
        vector_store_id = await ensure_vector_store(assistant)
        print(f"Assistant is now correctly configured with vector store ID: {vector_store_id}")
        
        return assistant

    except Exception as e:
        print("An error occurred in the main setup:", e)
        raise HTTPException(status_code=500, detail=str(e))