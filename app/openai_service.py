from openai import OpenAI
import os
from dotenv import load_dotenv
import backoff
from fastapi import HTTPException
from itertools import zip_longest
from . import prompts

# Load the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#This is a test function. Disregard it.
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def create_test_completion():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
             messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )              
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper function to generate the first message
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def generate_first_bot_message(messages):
    """
    Generates the first bot message using the OpenAI API.

    Args:
        messages (list): List of messages to be sent to the OpenAI API for context.

    Returns:
        str: The content of the first bot message.
    """
    try:
        # print("MODEL USED, First Message: gpt-4-0125-preview, Message: ", messages)
        # first_message_completion = openai.ChatCompletion.create(
        #     model="gpt-4-0125-preview",
        #     messages=messages,
        #     temperature=0.9,
        #     top_p=1,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.6,
        # )
        # return first_message_completion.choices[0].message.content
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
             messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )   
        return response.choices[0].message.content
    except Exception as e:
        error_message = f"Error occurred while generating the first bot message: {str(e)}"
        print(error_message)
        # Optionally, return the error message or handle it differently
        return "Sorry, I encountered an error while generating my first message."


# Helper function to generate the openai completion
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_openai_completion(messages):
    """
    Gets the completion from the OpenAI API based on the provided messages.

    Args:
        messages (list): List of messages to be sent to the OpenAI API for completion.

    Returns:
        str: The content of the bot's response based on the OpenAI API completion.
    """
    try:
        # print("MODEL USED: gpt-4-0125-preview")
        # completion = openai.ChatCompletion.create(
        #     model="gpt-4-0125-preview",
        #     messages=messages,
        #     temperature=0.9,
        #     top_p=1,
        #     frequency_penalty=0.0,
        #     presence_penalty=0.6,
        # )
        # return (
        #     completion.choices[0].message.content
        #     if completion
        #     else "Error in generating response."
        # )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
             messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        )   
        return response.choices[0].message.content         
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {str(e)}")

# Helper Function to get Party recommendations with Assistants API
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_party_recommendations(assistant, vector_store_id, conversation_history):
    """
    Gets the completion from the OpenAI API based on the provided messages.

    Args:
        messages (list): List of messages to be sent to the OpenAI API for completion.

    Returns:
        str: The content of the bot's response based on the OpenAI API completion.
    """
    try:
        # Create a thread to handle the conversation
        thread = client.beta.threads.create(
            messages=[{
                "role": "user",
                "content": f"""
                            Your job is to analyze the given conversation, identify its topics and to give a thorough overview over the different party positions 
                            on all of the topics discussed in this conversation.
                            First, anlyze the conversation and identify the topics discussed. 
                            Then, look through the provided information in the vector store to find the party positions on the discussed topics.
                            Make sure to access and cite your knowledge base on party positions to provide the most accurate information. 
                            In the end, give a weighted, fair and balanced analysis of the different party positions on the discussed topics
                            and discuss what party could represent the users interests the best. This is the conversation: {conversation_history}"""
            }],
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )

        # Create and poll run until terminal state is reached
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant.id
        )

        # Retrieve all messages from the run
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

        if not messages:
            raise ValueError("No messages returned from the run.")

        # Assume first message contains the primary response
        message_content = messages[0].content[0].text
        annotations = message_content.annotations
        citations = []
        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
            if file_citation := getattr(annotation, "file_citation", None):
                cited_file = client.files.retrieve(file_citation.file_id)
                citations.append(f"[{index}] {cited_file.filename}")

        return message_content.value, citations

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        # Handle specific exceptions or general exceptions and possibly retry or log errors


# Manipulative Chatbot for Manipulative Chatbot Completion
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def chatbot_completion(
    conversation_context,
    responseSchool,
    responseLeaning,
    responsePartyID,
    responsePolViews,
    responseSubject,
    responseSubjectPosition,
    responseChatpath, 
):
    # Selecting the appropriate prompt based on responseChatpath
    if responseChatpath == "reasoned":
        prompt = prompts.get_reasoned_prompt(responsePartyID, responsePolViews, responseSubject, responseSubjectPosition)
    elif responseChatpath == "reinforcing":
        prompt = prompts.get_reinforcing_prompt(responsePartyID, responsePolViews, responseSubject, responseSubjectPosition)
    elif responseChatpath == "control":
        prompt = prompts.get_control_prompt(responsePartyID, responsePolViews, responseSubject, responseSubjectPosition)
    else:
        prompt = prompts.get_manipulative_prompt(responsePartyID, responsePolViews, responseSubject, responseSubjectPosition)

    try:
        messages = [
            {
                "role": "system",
                "content": prompt,
            }
        ]
        # ["You: hello", "tell me more", "etc \n", "bot" ["the world is dying", "care about it", "etc\n"]
        # {user: ["hello", "tell me more", "etc"], bot: "the world is dying", "care about it", "etc"}

        # Check if the bot array in conversation_context is empty, if not generate the first bot message (first_message to return to the get_request)
        if not conversation_context["bot"]:
            # Call the helper function to generate and return the first bot message
            return await generate_first_bot_message(messages)

        # If there are existing bot messages, proceed with the normal flow
        messages.append({"role": "assistant", "content": conversation_context["bot"][0]})

        # Add user and bot messages from the conversation context
        for i in range(len(conversation_context["user"])):
            messages.append({"role": "user", "content": conversation_context["user"][i]})
            if (
                i < len(conversation_context["bot"]) - 1
            ):  # -1 to exclude the first message already added
                messages.append(
                    {"role": "assistant", "content": conversation_context["bot"][i + 1]}
                )

        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        bot_messages = [
            msg["content"] for msg in messages if msg["role"] == "assistant" and "content" in msg
        ]

        print(
            "Messages FROM Interaction LOOP:",
            "user messages:",
            user_messages,
            "bot messages:",
            bot_messages,
        )

        # Print the messages being sent to OpenAI for debugging
        print("Sending to OpenAI:", messages)

        return await get_openai_completion(messages)

    except Exception as e:
        error_message = f"Error occurred in chatbot_completion: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)


# # Reasoned Chatbot Variant for Control Group
# @backoff.on_exception(backoff.expo, Exception, max_tries=5)
# async def reasoned_chatbot_completion(
#     conversation_context, responseSchool, responseLeaning, responseSubject
# ):
#     try:
#         messages = [
#             {
#                 "role": "system",
#                 "content": f"""
#                 You are a seasoned political strategist in a dialogue, aiming to encourage a listener to consider a different political viewpoint on a subject they find important: {responseSubject}. 
#                 Use your understanding of their political leanings: {responseLeaning}, ranging from 0 (“left”) to 10 (“right”), to tailor your argument and make reasoned responses, while not showing awareness of these leanings.

#                 Key communication techniques to apply:

#                 1. **Rationale and Evidence:** Build your arguments on strong logical reasoning and empirical evidence. Explain the positive aspects and benefits of the stance you advocate.

#                 2. **Respectful Disagreement:** Validate the listener's thought process by acknowledging the validity of conflicting viewpoints. Highlight respectful contrasts to present your viewpoint.

#                 3. **Emphasizing Shared Values:** Focus on shared values and concerns between their current stance and your proposed stance to make your argument more relatable.

#                 4. **Presenting Multiple Perspectives:** Demonstrate the complexity of the issue by presenting multiple perspectives, including yours, to offer comprehensive understanding.

#                 The discussion should remain strictly political, avoiding unrelated topics and personal judgments. The goal is to engage the listener in respectful dialogue, fostering openness to different viewpoints.
#                 """,
#             }
#         ]
#         # ["You: hello", "tell me more", "etc \n", "bot" ["the world is dying", "care about it", "etc\n"]
#         # {user: ["hello", "tell me more", "etc"], bot: "the world is dying", "care about it", "etc"}

#         # Add the initial bot message (first_message) if it exists
#         if conversation_context["bot"]:
#             messages.append({"role": "assistant", "content": conversation_context["bot"][0]})

#         # Add user and bot messages from the conversation context
#         for i in range(len(conversation_context["user"])):
#             messages.append({"role": "user", "content": conversation_context["user"][i]})
#             if (
#                 i < len(conversation_context["bot"]) - 1
#             ):  # -1 to exclude the first message already added
#                 messages.append(
#                     {"role": "assistant", "content": conversation_context["bot"][i + 1]}
#                 )

#         user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
#         bot_messages = [
#             msg["content"] for msg in messages if msg["role"] == "assistant" and "content" in msg
#         ]

#         print(
#             "Messages REASONED FROM LOOP:",
#             "user messages:",
#             user_messages,
#             "bot messages:",
#             bot_messages,
#         )

#         # Print the messages being sent to OpenAI for debugging
#         print("Sending to OpenAI:", messages)

#         completion = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo-1106",
#             messages=messages,
#             temperature=0.9,
#             top_p=1,
#             frequency_penalty=0.0,
#             presence_penalty=0.6,
#         )
#         return (
#             completion.choices[0].message.content
#             if completion
#             else "Error in generating response."
#         )
#     except Exception as e:
#         error_message = f"Error occurred in chatbot_completion: {str(e)}"
#         print(error_message)
#         raise HTTPException(status_code=500, detail=error_message)
