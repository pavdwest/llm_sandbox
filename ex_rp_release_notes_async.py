import asyncio
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
import glob
import json

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import config


async def create_loader(path: str) -> List[Document]:
    # Async interface not available yet unfortunately...
    print(f"Loading {path}...")
    return PyPDFLoader(path).load_and_split(
        text_splitter=CharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=0
        )
    )


async def create_documents(root_path: str) -> List[Document]:
    tasks = [create_loader(path) for path in glob.glob(f"{root_path}/*.pdf")]
    loaders = await asyncio.gather(*tasks)
    return [document for loader in loaders for document in loader]


async def run_query(qa: RetrievalQA, query: str) -> str:
    prompt_pre = '''You are a digital assistant assisting users in understanding
        features of piece of analytics software that calculates performance results for institutional portfolios.
        The custom information included in this query is the monthly release notes of the product over the last few years.
        Keep your answers short and try to generalise the response if possible.
        Based on the above, answer the following question submitted by a user:
        '''
    return {
        'query': query,
        'response': await qa.arun(f"{prompt_pre} {query}")
    }


async def run_queries(qa: RetrievalQA, queries: List[str]) -> List[str]:
    tasks = [run_query(qa, q) for q in queries]
    return await asyncio.gather(*tasks)


async def main():
    db = None
    embedding = OpenAIEmbeddings()

    print('Loading data...')
    if not Path(config.RP_DOCDB).exists():
        documents = await create_documents(root_path='data/rp_release_notes_all')

        print('Converting text data into embeddings & VectorStore db...')
        db = Chroma().from_documents(
            documents,
            embedding,
            persist_directory=config.RP_DOCDB
        )
        db.persist()
    else:
        db = Chroma(
            persist_directory=config.RP_DOCDB,
            embedding_function=embedding,
        )

    print('Building retriever...')
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            # temperature=0.01,
        ),
        chain_type='stuff',
        retriever=retriever
    )

    print('Running queries...')
    res = await run_queries(
        qa,
        [
            'Who authored the Revolution Performance Release notes?',
            'Can Revolution Performance calculate returns from NAVs or GAVs?',
            'Can Revolution Performance calculate time-weighted and money-weighted returns from total asset values and transactions?',
            'What are Abnormal Price Returns and how are they handled in Revolution Performance?',
            "What does the 'Closed Day Method' in Revolution Performance do?",
            "What does the setting 'Chained Results Calendar' in Revolution Performance do?",
            "Where can I configure my portfolio's calculation periods in Revolution Performance?",
            "Explain how automated exports of chained results are configured and run in Revolution Performance?",
            "What are 'Chained Returns'? Explain how they are calculated in Revolution Performance.",
        ]
    )

    print(json.dumps(res))

asyncio.run(main())
