from dotenv import load_dotenv
import os

if __name__ == '__main__':
    load_dotenv()
    print(f"{os.environ['PINECONE_API_KEY']}")
    print("Ingesting documentation")