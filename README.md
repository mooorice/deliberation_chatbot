# Qualtrics AI API

## Description
The Deliberation Chatbot is an conversational AI system built using FastAPI. It is designed to engage users in meaningful dialogues, whether for thoughtful political discussions or light-hearted casual conversations. By leveraging OpenAI's API, the chatbot dynamically adapts its responses based on the user's inputs and session context, providing a personalized and engaging chat experience. The political assistant is designed to support users in self deliberation about political topics.

## Features
- **Socratic Political Dialogue**: Engage users in thoughtful political discussions, helping them to reflect on and refine their opinions.
- **Casual Conversations**: Provide users with a friendly and engaging chat experience on non-political topics.
- **Dynamic Interaction**: Automatically adapts the conversation style based on user inputs and session data.
- **Logging**: Captures detailed logs for monitoring and debugging.
- **Scalable Architecture**: Modular design with routers to handle different parts of the application.

## Installation
Pull the latest version of the Deliberative Chatbot from the GitHub repository: https://github.com/mooorice/deliberation_chatbot

### Running the API
To run the app locally, type in the following command in your terminal:
```bash
uvicorn app.main:app --reload --port 8001 --timeout-keep-alive 20
```
Then access via http://127.0.0.1:8001

### Running the Docker Container

Use 'docker build' to build the Docker container.

Use 'docker run' to run the Docker container, type in the following command in your terminal:

## License
This project is licensed under the MIT License. See the LICENSE file for details.