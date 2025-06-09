import os
import httpx
from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text
import pandas as pd
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import time

"""Run script to test connections to Azure OpenAI and SQL Server"""


def get_azure_token():
    """Get Azure token using DefaultAzureCredential"""
    try:
        credential = DefaultAzureCredential()
        # Get access token for Azure OpenAI scope
        access_token = credential.get_token("https://cognitiveservices.azure.com/.default")
        return access_token.token
    except Exception as e:
        print(f"Error getting Azure token: {str(e)}")
        return None

def test_sql_connection():
    """Test SQL connection using local database"""
    print("\n=== Testing SQL Connection ===")
    try:
        # Get connection string from .env
        connection_local = r"DRIVER={SQL Server Native Client 11.0};Server=KC-1014745\SQLEXPRESS;DATABASE=master;UID=RBKC\CGSIAHA;Trusted_Connection=yes;"
        connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": connection_local})
        engine = create_engine(connection_url)
        
        # Try to connect and run a simple query
        with engine.connect() as conn:
            print("✓ Successfully connected to SQL Server")
            
            # Test query to get database name
            result = conn.execute(text("SELECT DB_NAME() as database_name"))
            db_name = result.fetchone()[0]
            print(f"✓ Connected to database: {db_name}")
            
            # Test query to list tables
            result = conn.execute(text("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'"))
            tables = result.fetchall()
            print("\nAvailable tables:")
            for table in tables:
                print(f"- {table[0]}")
                
    except Exception as e:
        print(f"✗ SQL Connection Error: {str(e)}")
        return False
    
    return True

def test_azure_openai():
    """Test Azure OpenAI connection using DefaultAzureCredential"""
    print("\n=== Testing Azure OpenAI Connection ===")
    try:
        # Set up the credential
        credential = DefaultAzureCredential()
        access_token = credential.get_token("https://cognitiveservices.azure.com/.default").token
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_VERSION"),
            api_key=access_token,
            http_client=httpx.Client(verify=False), # need this to run fro work laptop
        )
        
        # Test completion
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        print(f"Using deployment: {deployment}")
        
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "Say 'Connection successful' if you can read this."}
            ],
            temperature=0
        )
        print(f"✓ Azure OpenAI Response: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"✗ Azure OpenAI Connection Error: {str(e)}")
        return False
    
    return True

def main():
    """Main test function"""
    print("Starting connection tests...")
    load_dotenv()  # Load environment variables
    
    # Print environment variables for debugging
    print("\nEnvironment Variables:")
    print(f"AZURE_OPENAI_ENDPOINT: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"AZURE_OPENAI_VERSION: {os.getenv('AZURE_OPENAI_VERSION')}")
    print(f"AZURE_OPENAI_DEPLOYMENT: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")
    
    sql_success = test_sql_connection()
    azure_success = test_azure_openai()
    
    print("\n=== Test Summary ===")
    print(f"SQL Connection: {'✓ PASSED' if sql_success else '✗ FAILED'}")
    print(f"Azure OpenAI: {'✓ PASSED' if azure_success else '✗ FAILED'}")

if __name__ == "__main__":
    main() 