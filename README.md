# PDF_RAG_App

## Introduction:
The appliaction here is an advanced Retrieval-Augmented Generation (RAG) application designed to answer quesions on the basis of a document. The application also has automatically triggered agents that can help to take notes and if needed also generate a set of multiple choice questions based on the content of the document.

In addition we are utilizing LLMs installed locally using the Ollama registery, which will need to run on our machine's background for us to utilize the same.

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



