import argparse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = 'chroma'


def main():
    # Create CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('query_text', type=str, help="The Query Text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the Text
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=1)

    # if len(results) == 0 or results[0][1] < 0.7:
    #     print(f'\n\n Unable to find matching results.')
    #     return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)


main()
