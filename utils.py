import os
from dotenv import load_dotenv

def load_environment_variables():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Ensure the OpenAI API key is set
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
