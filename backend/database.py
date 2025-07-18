from sqlalchemy import create_engine, URL
import pandas as pd
from config import SQL_CONNECTION_STRING, SQL_CONNECTION_LOCAL

def create_sql_connection(use_local = False): # use local databse or WCC datawarehouse
    """Creates connection to SQL data warehouse"""
    try:
        # Use a more explicit connection string format
        if use_local:
            if not SQL_CONNECTION_LOCAL:
                raise ValueError("Local SQL connection string not found in config")
            connection_string = SQL_CONNECTION_LOCAL
            print("Using Local Connection: {}".format(connection_string))
        else:
            if not SQL_CONNECTION_STRING:
                raise ValueError("Production SQL connection string not found in config")
            connection_string = SQL_CONNECTION_STRING
        
        print(f"Using connection string: {connection_string}")
        
        # Create the connection URL
        connection_url = URL.create(
            "mssql+pyodbc",
            query={"odbc_connect": connection_string}
        )
        
        # Create and return the connection
        engine = create_engine(connection_url)
        conn = engine.connect()
        
        # Test the connection
        test_query = "SELECT @@VERSION"
        result = pd.read_sql(test_query, conn)
        print("Successfully connected to database")
        print(f"SQL Server version: {result.iloc[0, 0]}")
        
        return conn
    
    except Exception as e:
        print(f"Error creating database connection: {str(e)}")
        raise

def execute_query(conn, query: str, params: dict = None) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame"""
    try:
        if params:
            # Print the parameters being used
            print(f"Query parameters: {params}")
            # Format the query with parameters using named parameter
            formatted_query = query.replace("{}", str(params['person_id']))
            print(f"Executing query: {formatted_query}")
            return pd.read_sql(formatted_query, conn)
        else:
            print(f"Executing query without parameters: {query}")
            return pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        print(f"Query that failed: {query}")
        if params:
            print(f"Parameters: {params}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def close_connection(conn):
    """Close SQL connection"""
    try:
        conn.close()
    except Exception as e:
        print(f"Error closing connection: {e}") 
