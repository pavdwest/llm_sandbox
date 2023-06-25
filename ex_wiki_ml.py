from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import config


print('Loading data into VectorStore...')
docs = WikipediaLoader('Performance attribution').load()
docs += WikipediaLoader('Midjourney').load()


print('Chunking source data...')
text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

print('Metadata:')
print(f"Texts: {len(texts)}")
print(len(texts[1].page_content))

print('Converting text data into embeddings & VectorStore db...')
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

print('Creating retriever...')
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0.01,
    ),
    chain_type='stuff',
    retriever=retriever
)

print('Running query...')
res = qa.run('I want to use midjourney. How do I use midjourney?')
print(res)



# print(index.query_with_sources('who authored the theoretical paper behind stable diffusion?'))
# print(index.query_with_sources('what is a controlnet?'))
