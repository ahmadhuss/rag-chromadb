import argparse

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

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

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Use LLM Model
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.5,
                     max_tokens=4000,
                     top_p=1,
                     frequency_penalty=0,
                     presence_penalty=0,
                     max_retries=2)
    response_text = llm.invoke(prompt)
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"\n\n\n{response_text.content}\n\n\nSources: {sources}"
    print(formatted_response)


main()
