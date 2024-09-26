from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = 'chroma'
DATA_PATH = 'data'
DATA_PATH_PDFS = 'data/pdf'


def main():
    generate_data_source()


def generate_data_source():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    documents = []
    # Define a jq_schema to extract data from JSON files
    jq_schema = '.'  # Modify this based on your JSON structure

    for root, dirs, files in os.walk(DATA_PATH):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith('.json'):
                # Use JSONLoader for JSON files with the specified jq_schema
                loader = JSONLoader(file_path, jq_schema, text_content=False)
                json_docs = loader.load()
                for doc in json_docs:
                    # If page_content is a list, join it into a string
                    if isinstance(doc.page_content, list):
                        doc.page_content = ' '.join(doc.page_content)
                    documents.append(doc)
            elif filename.endswith('.pdf'):
                # Use DirectoryLoader for other file types like PDFs
                loader = DirectoryLoader(DATA_PATH_PDFS)
                documents.extend(loader.load())
            else:
                print(f"Unsupported file type: {filename}")

    print('TOTAL_DOCUMENTS:', len(documents))
    for doc in documents:
        print(f'Document source: {doc.metadata.get("source", "N/A")}')
        print(f'Document content preview: {doc.page_content[:200]}...')

    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')

    if len(chunks):
        for index, chunk in enumerate(chunks):
            # Access the chunk using the index
            document = chunks[index]  # Or simply use `chunk`

            # Print the content and metadata of the document
            print(f'Document {index + 1} Page Content:', document.page_content)
            print(f'Document {index + 1} Meta Data:', document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # clear out the database first
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)
    # Force to save deprecated
    # db.persist()
    print(f'Saved {len(chunks)} chunks to {CHROMA_PATH}.')


main()
