import os
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import PyPDF2
import config

# Load embedding model
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path=config.CHROMA_CLIENT_PATH)
collection = chroma_client.get_or_create_collection(name=config.COLLECTION_NAME)

# Load the TinyLlama model for text generation
qa_pipeline = pipeline("text-generation", model=config.TEXT_GEN_MODEL)

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    if not os.path.exists(pdf_path): 
        raise FileNotFoundError(f"Error: The file '{pdf_path}' does not exist.")
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    if not text.strip():
        raise ValueError(f"Error: No text could be extracted from '{pdf_path}'.")
    return text

def get_pdf_hash(pdf_path):
    """Generate a unique hash for the PDF file based on its content."""
    sha256_hash = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def store_pdf_in_vector_db(pdf_paths):
    """Stores text from PDFs as embeddings in ChromaDB using upsert to avoid duplication."""
    for pdf_path in pdf_paths:
        pdf_hash = get_pdf_hash(pdf_path)
        #print("Hash_of_the_PDF_file :",pdf_hash)
        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        #print("Extracted_text_from_pdf_files  :",text)
        # Generate embedding
        embedding = embedding_model.encode([text])[0]

        # Upsert document to prevent duplicate entries
        collection.upsert(
            ids=[pdf_hash],  # Using PDF hash as a unique ID
            embeddings=[embedding.tolist()],
            documents=[text]
        )
        print(f"Stored content from '{os.path.basename(pdf_path)}' in ChromaDB!") 

    """else:
        print("No PDFs found to store in the vector database.")"""

def retrieve_first_n_lines(n=100):
    """Retrieve the first n lines of the stored PDF from the vector database."""
    result = collection.get() 
    #print ("Result from vector database:", result)
    if result and "documents" in result and result["documents"]:
        full_text = result["documents"][0]  # Get the first stored document
        lines = full_text.strip().split("\n")
        first_n_lines = "\n".join(lines[:n])
        print("\nRetrieved Text from Vector (First {} lines):\n".format(n))
        print(first_n_lines)
        return first_n_lines
    else:
        print("No stored documents found in the vector database.")
        return ""

def generate_questions(text):
    """Generate questions based on the given text using TinyLlama."""
    if not text:
        print("No text available for question generation.")
        return
    print("\nGenerating questions...\n")
    # Modify prompt for better question generation
    prompt = f"Read the following text and generate 5 questions:\n{text}\nQuestions:"
    # Use text-generation pipeline with max_new_tokens
    questions = qa_pipeline(prompt, max_new_tokens=500)
    # Print the generated questions
    print(questions[0]['generated_text'])

if __name__ == "__main__":
    retrieve_pdf_path = "D:/Data_Aces/Codes/interview_assis/data/retrieve_from_pdf/Wolpaw.pdf"  # Specify the PDF file
    #print("PDF file path:", retrieve_pdf_path)
    store_pdf_in_vector_db([retrieve_pdf_path])  # Pass as a list
    retrieved_text = retrieve_first_n_lines(100)
    # generate_questions(retrieved_text)