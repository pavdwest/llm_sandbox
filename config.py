import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
RP_DOCDB = 'db/rp_doc_db.chromadb'
