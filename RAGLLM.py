import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

# Function to create a retriever
def create_retriever(text, model):
    # Split text into smaller chunks (example splits by paragraph)
    splits = [chunk for chunk in text.split("\n\n") if chunk.strip()]

    # Create embeddings for the chunks
    embeddings = model.encode(splits, convert_to_tensor=True)

    # Initialize FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.cpu().numpy())

    return index, splits

# Function to answer questions using RAG
def answer_question(question, index, splits, model):
    # Create embedding for the question
    question_embedding = model.encode([question], convert_to_tensor=True)

    # Ensure the embedding is a 2D array
    question_embedding = question_embedding.cpu().numpy()
    if question_embedding.ndim == 1:
        question_embedding = question_embedding.reshape(1, -1)

    # Retrieve the most similar documents
    D, I = index.search(question_embedding, k=5)  # Try k=3 or k=1 for fewer results
    relevant_texts = [splits[i] for i in I[0]]

    # Display only the top result for a more concise answer
    context = relevant_texts[0]  # or "\n".join(relevant_texts[:3]) for more text
    return context

# Main function to run the chatbot
def main(pdf_path):
    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract text from the PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Create a retriever
    index, splits = create_retriever(text, model)
    
    # Enter a loop to answer questions
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        context = answer_question(question, index, splits, model)
        print(f"Context:\n{context}\n")

if __name__ == "__main__":
    pdf_path = input("Enter the path to the PDF file: ")
    main(pdf_path)













# from langchain_community.document_loaders.pdf import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from langchain_community.embeddings.ollama import OllamaEmbeddings

# from langchain_community.vectorstores.chroma import Chroma

# def loader():
#     doc_loader = PyPDFLoader("C:\\Users\\leahk\\GPTLLMfinetune\\00009.pdf")
#     x = doc_loader.load_and_split()
#     return x
    

# def splitter(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False
#     )
    
#     return text_splitter.split_documents(documents)

# # embedding is like a key to a database 
# # func returns an embed function because it is needed sumwehre else (creating db and querying)
# # diff embedding funcs 
# # ollama is a platform that manages and runs opensource LLMS locally

# def get_embed():
#     embeddings = OllamaEmbeddings(
#         model="nomic-embed-text"
#     )
#     return embeddings


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"

# def building_chromadb(chunks: list[Document]):
#     database = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embed())
#     database.add_documents(chunks, ids=chunks)
#     database.persist()
    
#     last_pageid = None
#     current_chunk_index = 0
    
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         if chunk.metadata["id"] not in existing_ids:
#             new_chunks.append(chunk)

#     if len(new_chunks):
#         print(f"Adding new documents: {len(new_chunks)}")
#         new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
#         database.add_documents(new_chunks, ids=new_chunk_ids)
#         database.persist()
#     else:
#         print("No new documents to add")
        
        
        
        

# import argparse
# import os
# import shutil
# from langchain.document_loaders.pdf import PyPDFDirectoryLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema.document import Document
# from langchain.vectorstores.chroma import Chroma


# CHROMA_PATH = "chroma"
# DATA_PATH = "data"



# def load_documents():
#     document_loader = PyPDFDirectoryLoader("C:\\Users\\leahk\\GPTLLMfinetune\\00009.pdf")
#     return document_loader.load()


# def split_documents(documents: list[Document]):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=800,
#         chunk_overlap=80,
#         length_function=len,
#         is_separator_regex=False,
#     )
#     return text_splitter.split_documents(documents)


# def add_to_chroma(chunks: list[Document]):
#     # Load the existing database.
#     db = Chroma(
#         persist_directory=CHROMA_PATH, embedding_function=get_embed()
#     )

#     # Calculate Page IDs.
#     chunks_with_ids = calculate_chunk_ids(chunks)


#     # Only add documents that don't exist in the DB.
#     new_chunks = []
#     for chunk in chunks_with_ids:
#         new_chunks.append(chunk)

   

# def calculate_chunk_ids(chunks):

#     # This will create IDs like "data/monopoly.pdf:6:2"
#     # Page Source : Page Number : Chunk Index

#     last_page_id = None
#     current_chunk_index = 0

#     for chunk in chunks:
#         source = chunk.metadata.get("source")
#         page = chunk.metadata.get("page")
#         current_page_id = f"{source}:{page}"

#         # If the page ID is the same as the last one, increment the index.
#         if current_page_id == last_page_id:
#             current_chunk_index += 1
#         else:
#             current_chunk_index = 0

#         # Calculate the chunk ID.
#         chunk_id = f"{current_page_id}:{current_chunk_index}"
#         last_page_id = current_page_id

#         # Add it to the page meta-data.
#         chunk.metadata["id"] = chunk_id

#     return chunks


# def clear_database():
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)


# import argparse

# from langchain.prompts import ChatPromptTemplate
# from langchain_community.llms.ollama import Ollama



# CHROMA_PATH = "chroma"

# PROMPT_TEMPLATE = """
# Answer the question based only on the following context:

# {context}

# ---

# Answer the question based on the above context: {question}
# """


# def main():
#     # Create CLI.
#     parser = argparse.ArgumentParser()
#     parser.add_argument("query_text", type=str, help="The query text.")
#     args = parser.parse_args()
#     query_text = args.query_text
#     query_rag(query_text)


# def query_rag(query_text: str):
#     # Prepare the DB.
#     embedding_function = get_embed()
#     db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

#     # Search the DB.
#     results = db.similarity_search_with_score(query_text, k=5)

#     context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
#     prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
#     prompt = prompt_template.format(context=context_text, question=query_text)
#     # print(prompt)

#     model = Ollama(model="mistral")
#     response_text = model.invoke(prompt)

#     sources = [doc.metadata.get("id", None) for doc, _score in results]
#     formatted_response = f"Response: {response_text}\nSources: {sources}"
#     print(formatted_response)
#     return response_text



# from langchain_community.llms.ollama import Ollama

# EVAL_PROMPT = """
# Expected Response: {expected_response}
# Actual Response: {actual_response}
# ---
# (Answer with 'true' or 'false') Does the actual response match the expected response? 
# """


# def test_monopoly_rules():
#     assert query_and_validate(
#         question="How much total money does a player start with in Monopoly? (Answer with the number only)",
#         expected_response="$1500",
#     )


# def test_ticket_to_ride_rules():
#     assert query_and_validate(
#         question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
#         expected_response="10 points",
#     )


# def query_and_validate(question: str, expected_response: str):
#     response_text = query_rag(question)
#     prompt = EVAL_PROMPT.format(
#         expected_response=expected_response, actual_response=response_text
#     )

#     model = Ollama(model="mistral")
#     evaluation_results_str = model.invoke(prompt)
#     evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

#     print(prompt)

#     if "true" in evaluation_results_str_cleaned:
#         # Print response in Green if it is correct.
#         print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
#         return True
#     elif "false" in evaluation_results_str_cleaned:
#         # Print response in Red if it is incorrect.
#         print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
#         return False
#     else:
#         raise ValueError(
#             f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
#         )
   
    



