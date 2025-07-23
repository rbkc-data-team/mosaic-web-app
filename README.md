# Mosaic Data Chat Application

A cloud-native web application for querying and chatting with data about vulnerable children, leveraging Azure OpenAI services and a vector store backend. The application is now split into separate frontend and backend components deployed as individual Azure Web Apps within the same Azure Resource Group.

---
## Features

- SQL data retrieval for vulnerable children records (Mosaic backend data)
- Vector storage using Qdrant (in-memory) - permanently deleted at end of user session
- RAG-based chat interface using Azure OpenAI GPT models
- FastAPI backend with Streamlit frontend
- Secure handling of sensitive data
- Restricted record detection and handling
- Enhanced citation/reference display with source linking to allow users to evalutate context sources
- Evalutation metrics to show quality of response

---

## Usage

1. Enter a Person ID/ Group ID in the sidebar
2. Click "Process Data" to load and process the data
3. The system will check for restricted records before processing
   - If a restricted record is found, you'll receive a clear error message
   - Processing will be stopped to prevent unauthorized access
4. If the record is not restricted, data will be processed and loaded
5. Each chat query retrieves relevant context from the vector index
6. Start chatting with the data using natural language queries
7. View source documents and citations in the sidebar
8. Click "Clear Chat" to start a new session
9. Use "Clean Up Vector Store" to clear all processed data

## Security Features

- Restricted record detection and blocking
- All sensitive data is processed in-memory
- Vector store is cleared after each session
- Ensure proper access controls on SQL Server
- Use appropriate Azure RBAC roles
- Automatic detection and prevention of restricted record processing

---

## Architecture Overview

- **Frontend**: Streamlit application (`app.py`) serving the user interface.  
  This lives in the **frontend** branch and is deployed to an Azure Web App dedicated to the frontend.
  This streamlit front end is currently used for testing and development.

- **Backend**: Python FastAPI application handling API requests, database connectivity, vector store management, and Azure OpenAI integrations.  
  This is maintained in the **backend** branch and deployed to a separate Azure Web App.

- Both apps communicate over REST API endpoints.  
- Both Web Apps exist within the same Azure Resource Group for consolidated management.

---
## Aplication Deployment

The code base is deployed through Azure DevOps repositories.  The Azure Web Apps are connected to thier respective frontend/backend branches and with each change in the repo (commit), the code feeding the web-app is updated.

---

## Repository Structure

- **frontend branch**  
  Contains the Streamlit frontend code including `app.py`, UI components, and environment configurations specific to the frontend.

- **backend branch**  
  Contains the FastAPI backend code that connects to databases, manages the vector store, and performs calls to Azure OpenAI. Includes connection strings, API implementations, and environment configs for backend services.

---

## Prerequisites

- Azure Subscription  
- Azure CLI installed and configured  
- Git, with access to Azure DevOps Repos for both branches  
- Python 3.8+  
- Azure OpenAI service with deployment keys and endpoints  
- SQL Server database with proper access credentials  
- Azure Web Apps created for both frontend and backend in the same resource group

---

## Environment Configuration

Both frontend and backend require environment variables. These should be configured in each Azure Web App's Application Settings or via Azure Key Vault.

Example variables include:

```env
# Common
AZURE_OPENAI_API_KEY=<your_api_key>
AZURE_OPENAI_ENDPOINT=<your_openai_endpoint>
AZURE_OPENAI_DEPLOYMENT=<deployment_name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding_deployment_name>

# Backend specific
SQL_CONNECTION_STRING=<your_sql_server_connection_string>
```

---

## Troubleshooting

### Common Issues

1. Application fails to start:
   - Check application logs in Azure Portal
   - Verify all environment variables are set correctly
   - Ensure Python version matches requirements
   - check startup.sh and ensure that this startup command is provided in the Azure Web App configuration menu

2. Database connection issues:
   - Verify SQL Server firewall rules
   - Check connection string format
   - Ensure proper network security group rules

3. Vector store cleanup issues:
   - Check application logs for cleanup errors
   - Verify Qdrant client configuration
   - Monitor memory usage in Azure Portal