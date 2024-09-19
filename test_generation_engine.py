import os
from typing import List, Dict
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader
from dotenv import load_dotenv
from llama_index.core.tools import FunctionTool



file_path_for_doc = os.getcwd()+"/data/sound.pdf"

def test_generation(num_questions,file_path_for_doc):

    prompt = f"""
    Based on the content of the document, generate a set of {num_questions} multiple-choice questions.
    For each question, provide:
    1. The question itself
    2. Four possible answers (A, B, C, D)
    3. The correct answer
    4. A brief explanation of why the correct answer is right
    5. Make sure that there are no repeated questions

    Format your response as a Python list of dictionaries, where each dictionary represents a question and follows this structure:
    {{
        "question": "The question text",
        "options": {{
            "A": "First option",
            "B": "Second option",
            "C": "Third option",
            "D": "Fourth option"
        }},
        "correct_answer": "The correct option letter (A, B, C, or D)",
        "explanation": "Explanation for the correct answer"
    }}
    """

    def clean_response(response: str) -> str:
        start = response.find('[')
        end = response.rfind(']') + 1
        if start == -1 or end == 0:
            raise ValueError("Could not find a valid list in the response")
        
        list_str = response[start:end]
        list_str = list_str.replace('```python', '').replace('```', '')
        
        return list_str.strip()
    
    def save_mcq_to_file(mcq_test: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, question in enumerate(mcq_test, 1):
                f.write(f"Question {i}:\n")
                f.write(f"{question['question']}\n")
                for option, text in question['options'].items():
                    f.write(f"{option}. {text}\n")
                f.write(f"\nCorrect Answer: {question['correct_answer']}\n")
                f.write(f"Explanation: {question['explanation']}\n")
                f.write("-" * 50 + "\n\n")

    documents = PDFReader().load_data(file_path_for_doc)
    llm = Ollama(model="llama3.1", temperature=0.7, request_timeout=420)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents,embed_model=embed_model)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(prompt)
    cleaned_response = clean_response(response.response)
    save_mcq_to_file(eval(cleaned_response), "/teamspace/studios/this_studio/mcq_test2.txt")
    
    print(f"MCQ test has been generated and saved and the questions can also be found below")

    for i, question in enumerate(eval(cleaned_response), 1):
        print(f"\nQuestion {i}:")
        print(question['question'])
        for option, text in question['options'].items():
            print(f"{option}. {text}")
        print(f"\nCorrect Answer: {question['correct_answer']}")
        print(f"Explanation: {question['explanation']}")
        print("-" * 50)    

    return "Test Created"


test_paper_engine = FunctionTool.from_defaults(
    fn=test_generation,
    name="mcq_test_generation",
    description="this tool generates required number of question mcq testpaper based on the pdf file present at the location /teamspace/studios/this_studio/data/sound.pdf ",
)


# if __name__ == "__main__":
#     test_generation(15,"/teamspace/studios/this_studio/data/iesc111.pdf")

