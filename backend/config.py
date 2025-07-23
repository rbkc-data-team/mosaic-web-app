import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

load_dotenv()

# Azure Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")
AZURE_EMBED_API_VERSION = os.getenv("AZURE_EMBED_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
AZURE_OPENAI_EMBEDDING_MODEL_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
AZURE_BLOB_STORAGE = os.getenv("AZURE_BLOB_STORAGE")
AZURE_BLOB_STORAGE_CONTAINER = os.getenv("AZURE_BLOB_STORAGE_CONTAINER")

# SQL Server Configuration
SQL_CONNECTION_STRING = os.getenv("SQL_CONNECTION_STRING")
SQL_CONNECTION_LOCAL = os.getenv("SQL_CONNECTION_LOCAL")
USE_LOCAL = os.getenv("USE_LOCAL")

# Azure Key Vault Configuration (optional)
KEY_VAULT_NAME = os.getenv("KEY_VAULT_NAME")
KEY_VAULT_URI = f"https://{KEY_VAULT_NAME}.vault.azure.net/" if KEY_VAULT_NAME else None

def get_secret_from_keyvault(secret_name: str) -> str:
    """Retrieve a secret from Azure Key Vault"""
    if not KEY_VAULT_URI:
        return None
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=KEY_VAULT_URI, credential=credential)
        return client.get_secret(secret_name).value
    except Exception as e:
        print(f"Error retrieving secret from Key Vault: {e}")
        return None 