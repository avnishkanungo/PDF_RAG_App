# PDF_RAG_App

## Introduction:
The appliaction here is an advanced Retrieval-Augmented Generation (RAG) application designed to answer quesions on the basis of a document. The application also has automatically triggered agents that can help to take notes and if needed also generate a set of multiple choice questions based on the content of the document.

In addition we are utilizing LLMs installed locally using the Ollama registery, which will need to run on our machine's background for us to utilize the same.

## Features:

1. Using locally installed LLM using Ollama
2. Vector DB querying process optimized, vector DB queried only when there the prompt is relevant to the Vector DB content. This is done by calculating the cosine similarity of the vector embedding of the prompt and comparing it with the average vector embedding value of the data stored in the vector DB.
3. The agentic workflow includes the following tools:
    - Question Answering Tool
    - Note Taker Tool
    - MCQ Question Generator Tool
4. Output of the API based setup also provides a base64 encoded .wav file, which is implemented using the Sarvam Text To Speech API.

## **Steps to run the code:**

1. Download Ollama registry to install the model locally from this [link](https://ollama.com/download/linux)
2. Once you have Ollama installed, you can enable the ollama server and install the required model using the below commands in the termina:

```
ollama serve
```

```
ollama pull llama3.1
```
3. After Ollama is running, you can configure a virtual env of your choice(conda or normal python) and install the libraries using the below command, in another tab in the bash terminal:

```
pip install -r requirements.txt
```
4. Then we can use the below command to trigger the RAG application:

```
python query_engine.py
```
5. Once the command runs successfully, you will be promted in the terminal to enter how you want to tun this app with the options being Terminal or API.

```
How do you want to implement this app(terminal/API):
```
6. If you choose terminal, this will work as a terminal application and you will be asked for a prompt for the application in the terminal. If you choose the API option, this will start a local uvicorn server which you can use by navigating to the local server with /docs (127.0.0.1:8000/docs). Here you will find the Swagger UI which you can use to test out agent using the API.

## Output Example:

1. Terminal based outputs:
    
    - Prompts relevant to the vector DB content:
    
    ![Terminal Output 1](./images/terminal_output4.png)

    - Prompts not relevant to the vector DB content:

    ![Terminal Output 2](./images/terminal_output1.png)

    - MCQ Test Paper Creation Agent:

    ![Terminal Output 3](./images/terminal_output3.png)

    - Note Taker Agent:

    ![Terminal Output 4](./images/terminal_output2.png)

2. API based outputs:

    - Prompts relevant to the vector DB content:
    
    ![API Output 1](./images/api_output3.png)

    - Prompts not relevant to the vector DB content:

    ![API Output 2](./images/api_output1.png)

    - MCQ Question Generation Agent:

    ![API Output 3](./images/api_output2.png)








