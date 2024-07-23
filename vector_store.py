from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# define pages to load
URLs=[
    'https://www.promtior.ai/',
    'https://www.promtior.ai/service',
    'https://www.promtior.ai/use-cases',
    'https://www.promtior.ai/contacto'

]

# load pages from website
loader = WebBaseLoader(URLs)

# split the pages into chunks
data = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})

# create the vector store database
db = FAISS.from_documents(texts, embeddings)

# save the vector store
db.save_local("faiss")