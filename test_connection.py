import sys
from sqlalchemy import create_engine, URL
import pandas as pd
import urllib

def create_sql_connection(connection_string):
    try:
        # URL encode the connection string
        connection_string_encoded = urllib.parse.quote_plus(connection_string)

        connection_url = URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": connection_string_encoded}
        )
        print("\nConstructed SQLAlchemy connection URL:")  
        print(connection_url)
        
        engine = create_engine(connection_url)
        conn = engine.connect()
        print("Connection established!")  

        # Test the connection
        test_query = "SELECT @@VERSION"
        print(f"Running test query: {test_query}") 
        result = pd.read_sql(test_query, conn)
        print("Successfully connected to database")
        print(f"SQL Server version: {result.iloc[0, 0]}")
        
        return conn
    except Exception as e:
        print(f"Error creating database connection: {str(e)}")
        raise

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_connection.py \"<SQL_CONNECTION_STRING>\"")
        sys.exit(1)
    
    connection_string = sys.argv[1]
    
    try:
        conn = create_sql_connection(connection_string)
        conn.close()
    except Exception:
        print("Connection test failed.")

if __name__ == "__main__":
    main()
