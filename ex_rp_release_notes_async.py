import os
import asyncio
from pathlib import Path
from typing import List, Type
import glob
import json

from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import (
    PyPDFLoader,
    PDFMinerLoader,
    PyPDFium2Loader,
    PDFMinerPDFasHTMLLoader,
    PDFPlumberLoader,
    UnstructuredHTMLLoader,
    TextLoader,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import config


concurrency_limit = asyncio.Semaphore(5)


# PDFLoader = PyPDFLoader
# PDFLoader = PDFMinerLoader
# PDFLoader = PyPDFium2Loader
# PDFLoader = PDFMinerPDFasHTMLLoader
# PDFLoader = PDFPlumberLoader


def get_loader_type(path: str) -> Type[BaseLoader]:
    match Path(path).suffix:
        case '.pdf':
            # return PyPDFLoader
            # return PDFMinerLoader
            return PyPDFium2Loader
            # return PDFMinerPDFasHTMLLoader
            # return PDFPlumberLoader
        case '.html':
            return UnstructuredHTMLLoader
        case _:
            return TextLoader


async def create_loader(path: str) -> List[Document]:
    # Async interface not available yet unfortunately...
    print(f"Loading '{path}'...")
    LoaderClass = get_loader_type(path)
    if LoaderClass is not None:
        return LoaderClass(path).load_and_split(
            text_splitter=CharacterTextSplitter(
                chunk_size=4000,
                chunk_overlap=0
            )
        )
    else:
        print(f"COULD NOT LOAD DATA: {path}")
        breakpoint()
        exit()


async def create_documents(root_path: str) -> List[Document]:
    tasks = [create_loader(path) for path in glob.glob(f"{root_path}/**/*.*", recursive=True)]
    loaders = await asyncio.gather(*tasks)
    return [document for loader in loaders for document in loader]


async def run_query(llm: OpenAI, retriever: VectorStoreRetriever, query: str) -> str:
    async with concurrency_limit:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever
        )
        prompt_pre = '''You are a digital assistant explaining the features of Revolution Performance, analytics software for institutional portfolios.
            Explain features at a high level but keep your answer concise, to the point and general and always include the version number. Never include any specific client names.
            Answer the following question submitted by a user: '''
        print(f"Running query '{query}'...")
        return {
            'query': query,
            'response': await qa.arun(f"{prompt_pre} {query}")
        }


async def run_queries(llm: OpenAI, retriever: VectorStoreRetriever, queries: List[str]) -> List[str]:
    tasks = [run_query(llm, retriever, q) for q in queries]
    return await asyncio.gather(*tasks)


async def main():
    doc_path = 'data/rp_docs'

    # Checking if source data is present
    if not Path(doc_path).exists():
        print(f"Expected release notes pdfs to exist in '{doc_path}'. Exiting.")
        exit(1)

    db = None
    embedding = OpenAIEmbeddings()

    print('Loading data...')
    if not Path(config.RP_DOCDB).exists():
        documents = await create_documents(root_path=doc_path)

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
    llm = OpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.0,
        # model='gpt-3.5-turbo',
        # model='text-davinci-003',
        # model='text-davinci-002',
    )

    print('Running queries...')
    res = await run_queries(
        llm,
        retriever,
        [
            # 'Who authored the Revolution Performance Release notes?',
            # "Explain 'Aggregate Cash per Currency' setting in Revolution Performance." ,
            # 'Can Revolution Performance calculate returns from NAVs or GAVs?',
            # 'Can Revolution Performance calculate time-weighted and money-weighted returns from total asset values and transactions?',
            # 'Explain Price and Trading Returns in Revolution Performance.',
            # 'What returns can Revolution Performance calculate?',
            # 'Can Revolution Performance calculate Turnover Ratios?',
            # "Explain what the 'Closed Day Method' is and what it does in Revolution Performance",
            # "What are Abnormal Returns and why are they useful in Revolution Performance?",
            # "What does the setting 'Chained Results Calendar' in Revolution Performance do?",
            # "Where can I configure my portfolio's calculation periods in Revolution Performance?",
            # "Explain how automated exports of chained results are configured and run in Revolution Performance?",
            # "What are 'Chained Returns'? Explain how they are calculated in Revolution Performance.",
            # "How does Revolution Performance calculate derivatives like Futures and Options?",
            # 'What fields are available in the Workflow Summary Export json? Include the release number in which the feature was last updated.',
            # "What does the 'Share Price Variance' control do in Revolution Performance?",
            # 'Tell me everything you know about "Chained Business Returns".',
            # 'In which releases are "Business Return", "Business Returns", "Chained Business Returns" or "Chained Business Return" mentioned?',
            # "In which release notes is 'Chained Results Calendar' mentioned?",
            # "In which release notes is 'Export Batch' mentioned?",
            # "What feature produces a single file per workflow group for an export configuration",
            "Is there a feature to help with too many files being exported to FTP each day?",
        ]
    )

    print(json.dumps(res, indent=2))

asyncio.run(main())
