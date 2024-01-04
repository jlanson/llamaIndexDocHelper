from dotenv import load_dotenv
import os
import pinecone
from llama_index.vector_stores import PineconeVectorStore
from llama_index import (
    SimpleDirectoryReader,
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENVIRONMENT"],
)


if __name__ == "__main__":
    load_dotenv()
    print("Ingesting documentation")
    # A data connector that helps load in html files
    UnstructuredReader = download_loader("UnstructuredReader")
    simpleDirectoryReader = SimpleDirectoryReader(input_dir=".\\bts-docs", file_extractor={".html": UnstructuredReader()})

    try:
        docs = simpleDirectoryReader.load_data()
    except:
        print("error")
    
    #Chunking the data
    simpleNodeParser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # Line not needed because node parsing is automatically done 
    # nodes = simpleNodeParser.get_nodes_from_documents(documents=docs)

    gpt = OpenAI(model="gpt-3.5-turbo", temperature=0)
    embedModel = OpenAIEmbedding(model="text-embedding-ada-002", embed_batch_size=100)
    serviceContext = ServiceContext.from_defaults(llm=gpt, embed_model=embedModel, node_parser=simpleNodeParser)

    # api key is supplied within .env
    indexName = 'llama'
    pineconeIndex = pinecone.Index(index_name=indexName)
    pineconeVectorStore = PineconeVectorStore(pinecone_index=pineconeIndex)
    storageContext = StorageContext.from_defaults(vector_store=pineconeVectorStore)

    index = VectorStoreIndex.from_documents(documents=docs, service_context=serviceContext, storage_context=storageContext, show_progress=True)

    print("Done")



