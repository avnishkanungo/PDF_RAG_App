import os
import numpy as np
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from test_generation_engine import test_paper_engine
from llama_index.core.agent import ReActAgent
from notes_engine import note_engine
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests

app = FastAPI()

class engine_implementation():
    def __init__(self,file_path,VDB_path,llm):
        self.file_path = file_path
        self.VDB_path = VDB_path
        self.llm = llm
    
    def save_embedding_to_disk( self, model_name, collection_name):
        pdf_file = PDFReader().load_data(file = self.file_path)

        avg_embedding = self.avg_embedding_calculation(pdf_file,model_name)

        embed_model = HuggingFaceEmbedding(model_name=model_name)

        db = chromadb.PersistentClient(path=self.VDB_path)
        chroma_collection = db.get_or_create_collection(collection_name)

        # save embedding to disk
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        og_index = VectorStoreIndex.from_documents(
            pdf_file, storage_context=storage_context, embed_model=embed_model)
        
        return og_index, embed_model, avg_embedding
    
    
    def avg_embedding_calculation(self, pdf_file, model_name):
        parser = SentenceSplitter()
        sentences = []
        for doc in pdf_file:
            nodes = parser.get_nodes_from_documents([doc])
            sentences.extend([node.text for node in nodes])
        model = SentenceTransformer(model_name)
        return np.mean(model.encode(sentences),axis=0)

    
    def read_embedding_from_disk(self,collection_name,embedding_model):
        db_read = chromadb.PersistentClient(path=self.VDB_path)
        chroma_collection = db_read.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embedding_model,
        )
        return index

    # def create_query_engine()
    
    def llm_call(self,index):
        query_engine = index.as_query_engine(llm=self.llm)
        return query_engine
        # model = SentenceTransformer("BAAI/bge-base-en-v1.5")
        # while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        #     similarity =  cosine_similarity(model.encode(prompt).reshape(1, -1), avg_embedding.reshape(1, -1))[0][0]
        #     print(similarity)
        #     if similarity > 0.7:
        #         result = query_engine.query(prompt)
        #     else:
        #         result = self.llm.complete(prompt)
        #     print(result)

def agent_init():
    file_path = os.getcwd() + "/data/sound.pdf"
    vdb_path = os.getcwd() + "/chromadb"
    embedding_model = "BAAI/bge-base-en-v1.5"

    llm = Ollama(model="llama3.1") # , request_timeout=420.0)

    q_engine = engine_implementation(file_path=file_path,VDB_path=vdb_path,llm=llm)

    all_file_index, embedding_model, average_embedding = q_engine.save_embedding_to_disk(model_name=embedding_model,collection_name="DB_Collection")

    print(average_embedding)

    specific_file_index = q_engine.read_embedding_from_disk(collection_name="DB_Collection",embedding_model=embedding_model)

    # q_engine.llm_call(specific_file_index,average_embedding,embedding_model)

    tools = [
        test_paper_engine,
        QueryEngineTool(
            query_engine=q_engine.llm_call(specific_file_index),
            metadata=ToolMetadata(
                name="llm_question_answer_function",
                description="this gives out information about the pdf file that is being used here",
            ),
        ),
        note_engine
             ]

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")
    agent = ReActAgent.from_tools(tools, llm=llm)

    return agent,llm, average_embedding, model

def agent_call():
    agent, llm, avg_embedding,model = agent_init()
    while (prompt := input("Enter a prompt (q to quit): ")) != "q":
        similarity =  cosine_similarity(model.encode(prompt).reshape(1, -1), avg_embedding.reshape(1, -1))[0][0]
        print(similarity)
        if similarity > 0.55:
            result = agent.query(prompt)
            # encoded_wavfile = tts(result)
        else:
            result = llm.complete(prompt)
            # encoded_wavfile = tts(result)
        print(result)

def tts(result):
    url = "https://api.sarvam.ai/text-to-speech"

    payload = {
        "inputs": ["Here is a 5 question MCQ test paper based on characteristics of sound, reverberation and ultrasound"],
        "target_language_code": "en-IN"
    }
    headers = {
        "api-subscription-key": "b54e7cff-e055-43da-a1a8-31512b3c85b0",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.json()["audios"]


class Query(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Welcome to the LlamaIndex Agent API. Use the /ask endpoint to ask questions."}

@app.post("/ask")
async def ask_question(query: Query):
    agent, llm, avg_embedding,model = agent_init()
    try:
        similarity =  cosine_similarity(model.encode(query.question).reshape(1, -1), avg_embedding.reshape(1, -1))[0][0]
        if similarity > 0.55:
            response = agent.query(query.question)
            encoded_wavfile = tts(response)
        else:
            response = llm.complete(query.question)
            encoded_wavfile = tts(response)
        output_dict = {
            "result":response,
            "encoded_wav":encoded_wavfile,
        }
        return {"answer": output_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


        
        
if __name__ == "__main__":
    option = input("How do you want to run this app(terminal/API):")
    if option.lower() == "terminal":
        agent_call()
    elif option.lower() == "api":
        uvicorn.run(app, host="127.0.0.1", port=8000)
    

# file_path = os.getcwd() + "/data/iesc111.pdf"

# pdf_file = PDFReader().load_data(file = file_path)

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# db = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db.get_or_create_collection("DB_Collection")

# # save embedding to disk
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# storage_context = StorageContext.from_defaults(vector_store=vector_store)

# index = VectorStoreIndex.from_documents(
#     pdf_file, storage_context=storage_context, embed_model=embed_model
# )

# # read from Vector DB
# db2 = chromadb.PersistentClient(path="./chroma_db")
# chroma_collection = db2.get_or_create_collection("DB_Collection")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
# index = VectorStoreIndex.from_vector_store(
#     vector_store,
#     embed_model=embed_model,
# )

# llm = Ollama(model="llama3.1", request_timeout=420.0)

# query_engine = index.as_query_engine(llm=llm)

# response = query_engine.query(input("Ask the question:"))
# print(response)




