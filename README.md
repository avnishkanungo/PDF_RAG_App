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



