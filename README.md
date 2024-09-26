# Knowledge-Based CLI RAG Application

This repository contains a knowledge-based CLI-type RAG (Retrieval-Augmented Generation) application. RAG applications typically work with textual data. In this repository, we can pass the textutal data in two formats: `.pdf` and `.json`.

This CLI-based RAG application uses the Langchain framework along with various ecosystem packages, such as:

-   langchain-core
-   langchain-community
-   langchain-chroma
-   langchain-openai

The repository utilizes the OpenAI LLM model for query retrieval from the vector embeddings.

## Installation  
  
Ensure you have Python >=3.10 <=3.13 installed on your system.

### Create virtual environment:  

```bash
python3 -m venv env
```

After that, Activate the virtual environment.
  
### Windows:  
  
```bash  
env\Scripts\activate  
```  
  
### Unix:  
  
```bash  
source env/bin/activate  
```  

### Install Dependencies:  

```bash  
pip install -r requirements.txt
```  

### Create Vector Database - Embedding based on the provided data (PDF, JSON files):

```bash  
python3 create_database.py
```

### Direct Query from the Vector Database:

You can perform simple direct queries on the vector database with the following commands:

```bash  
python3 query_data.py "What is Lee Kim?"
```

```bash  
python3 query_data.py "Who is Mr. Ken?"
```

```bash  
python3 query_data.py "What company type does Test Company have?"
```

### LLM-Based Queries on Our Vector Database:

You can leverage a language model for querying the vector database with more complex requests:

```bash  
python3 quality_response.py "What is Lee Kim?"
```

```bash  
python3 quality_response.py "Who is Mr. Ken?"
```

```bash  
python3 quality_response.py "What company type does Test Company have?"
```

```bash  
python3 quality_response.py "Provide information about Test Company in tabular format."
```

```bash  
python3 quality_response.py "How many resources are working in Lee Kim? List them in tabular format."
```

### License:
MIT