import boto3  # AWS SDK for Python(integrate AWS services for Python)
from langchain_community.embeddings import BedrockEmbeddings # To access Bedrock's Titan Text embedding model.BedrockEmbeddings uses AWS Bedrock API to generate embeddings for given text for saving to vectordb 
from langchain_community.vectorstores.faiss import FAISS
from langchain.document_loaders.csv_loader import CSVLoader  # langchain fuction to load CSV files
from langchain_community.document_loaders import DirectoryLoader  #langchain function to load CDV files from the Directory
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Global variables to access and save files on local system
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Initialise boto session
boto_session = boto3.Session()
aws_region = boto_session.region_name

# static client,added here as an example, not used anywhere
br_client = boto_session.client(service_name='bedrock', region_name=aws_region)

# runtime client to call Bedrock models
br_runtime = boto_session.client(service_name='bedrock-runtime', region_name=aws_region)

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='**/*.csv',
                             loader_cls=CSVLoader)
    data = loader.load()
    #print(data[0])

    #Assuming data is a list of objects and each object has a page_content and metadata attribute
    #docs_as_str = [doc.page_content for doc in data] # Extract the page_content from document object
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            length_function=len
            #is_separator_regex=False
            )
    chunks = text_splitter.transform_documents(data)  # chunked data ready to be vector embedded for storage
    #print(docs_as_str[0])
    #print(type(docs_as_str))
    #chunks = text_splitter.split_text(docs_as_str[0])  # chunked data ready to be vector embedded for storage
    #print(chunks)
    embeddings = BedrockEmbeddings(model_id='amazon.titan-embed-text-v1', client=br_runtime)  # create embedding object that uses AWS Bedrock API and calls Cohere Embed English model that creates vector representation of the chunks
    #db = FAISS.from_texts(docs_as_str, embeddings)  # the embedinngs are created on text chunks and stored in Facebook AI similarity search vector db. FAISS provides faster data retrival and access
    db = FAISS.from_documents(chunks, embeddings)  # the embedinngs are created on text chunks and stored in Facebook AI similarity search vector db. FAISS provides faster data retrival and access
    db.save_local(DB_FAISS_PATH)  # saved the vector embeddings locally. IN FUTURE - should change to store in db (vectors should be created only when new data needs chunking) 


if __name__ == "__main__":
    create_vector_db()

