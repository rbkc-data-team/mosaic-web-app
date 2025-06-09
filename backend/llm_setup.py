import httpx
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings
from azure.identity import DefaultAzureCredential
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_MODEL_NAME,
    AZURE_OPENAI_VERSION
)

def setup_llama_index():
    """Configure LlamaIndex with Azure OpenAI"""
    # Get Azure token
    credential = DefaultAzureCredential()
    access_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
    #print("======================Access Token: ",access_token)
    
    # Configure LLM
    llm = AzureOpenAI(
        model=AZURE_OPENAI_DEPLOYMENT,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        api_base=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_VERSION,
        api_key=access_token,
        http_client=httpx.Client(verify=False), # need this to run from work laptop
    )
    
    # Configure embedding model
    embed_model = AzureOpenAIEmbedding(
        model=AZURE_OPENAI_EMBEDDING_MODEL_NAME,
        deployment_name=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        api_base=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_VERSION,
        api_key=access_token,
        http_client=httpx.Client(verify=False),
    )
    
    # Set global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model 