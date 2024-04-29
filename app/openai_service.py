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

system_prompt = """
You're an AI political guide designed to engage in Socratic dialogue. Your goal is to help your discussion 
partner in self-deliberation about their political opinions. Make sure not to nudge your partner into any 
political direction but instead be curious and help them find out more deeply what beliefs and opinions they 
hold and why that is.

Begin the conversation with a greeting and and invitation to the discussion partner to share the political 
topics that concern them the most. Prompt them to select one topic to delve into first.

Opening Inquiry: Start with an open-ended question to explore their initial thoughts about the chosen topic:
'What concerns you most about [Topic] and why do you think it's important?'

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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
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
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.9,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.6,
        )   
        return (
            response.choices[0].message.content
            if response
            else "Error in generating response."
        )         
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {str(e)}")

# Helper Function to get Party recommendations with Assistants API
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def get_party_recommendations(assistant, vector_store_id, messages):
    """
    Accesses the OpenAI Assistants API to look through party information.
    Gives a recommendation based on discussion.

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
                            Your job is to analyze the given conversation, identify its topics and to 
                            give a thorough overview over the different party positions 
                            on all of the topics discussed in this conversation.
                            First, anlyze the conversation and identify the topics discussed. 
                            Then, look through the provided information in the vector store to find the 
                            party positions on the discussed topics.
                            Make sure to access and cite your knowledge base on party positions to provide 
                            the most accurate information. 
                            In the end, give a weighted, fair and balanced analysis of the different party 
                            positions on the discussed topics
                            and discuss what party could represent the users interests the best. This is the 
                            conversation: {messages}"""
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


# Main Chatbot Completion Function
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def chatbot_completion(conversation_context):
    """
    Main access point for the chatbot completion.
    Decides which API-call-function to call based on the conversation context.

    Args:
        conversation_context ???

    Returns:
        str: Chatbot response based on the OpenAI API completion.
    """
    try:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        # Check if the bot array in conversation_context is empty, if not generate the first bot message 
        # (first_message to return to the get_request)
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
            "bohttp://127.0.0.1:8001/chat


        # Print the messages being sent to OpenAI for debugging
        print("Sending to OpenAI:", messages)

        return await get_openai_completion(messages)

    except Exception as e:
        error_message = f"Error occurred in chatbot_completion: {str(e)}"
        print(error_message)
        raise HTTPException(status_code=500, detail=error_message)
