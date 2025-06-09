# Mosaic Data Chat Application

A web application that allows users to query and chat with data about vulnerable children using Azure OpenAI and LlamaIndex.

## Features

- SQL data retrieval for vulnerable children records
- Vector storage using Qdrant (in-memory)
- RAG-based chat interface using Azure OpenAI
- FastAPI backend with Streamlit frontend
- Secure handling of sensitive data
- Restricted record detection and handling
- Enhanced citation display with source linking

## Prerequisites

- Python 3.8+
- SQL Server with appropriate access
- Azure OpenAI service
- Azure Key Vault (optional)
- Azure CLI (for deployment)
- Gunicorn (for production deployment)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_VERSION=2023-05-15
AZURE_OPENAI_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name

SQL_CONNECTION_STRING=your_sql_connection_string
SQL_CONNECTION_LOCAL=your_local_sql_connection_string

# Optional Key Vault configuration
KEY_VAULT_NAME=your_key_vault_name
```

## Running the Application Locally

1. Start the FastAPI backend:
```bash
# For development
uvicorn api:app --reload

# For production
 gunicorn api:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

2. Start the Streamlit frontend:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Deploying to Azure Web App

### 1. Prerequisites
- Azure subscription
- Azure CLI installed and configured
- Git installed
- Access to Azure OpenAI service
- SQL Server connection details

### 2. Azure Resource Setup

1. Create a Resource Group:
```bash
az group create --name <your-resource-group> --location <your-location>
```

2. Create an App Service Plan:
```bash
az appservice plan create --name <your-plan-name> --resource-group <your-resource-group> --sku B1 --is-linux
```

3. Create a Web App:
```bash
az webapp create --resource-group <your-resource-group> --plan <your-plan-name> --name <your-app-name> --runtime "PYTHON:3.9"
```

### 3. Configure Application Settings

1. Set up Azure Key Vault (recommended):
```bash
az keyvault create --name <your-keyvault-name> --resource-group <your-resource-group> --location <your-location>
```

2. Add secrets to Key Vault:
```bash
az keyvault secret set --vault-name <your-keyvault-name> --name "AZURE-OPENAI-API-KEY" --value <your-api-key>
az keyvault secret set --vault-name <your-keyvault-name> --name "SQL-CONNECTION-STRING" --value <your-connection-string>
```

3. Configure Web App settings in Azure Portal:
   - Navigate to your Web App
   - Go to Configuration > Application settings
   - Add the following settings:
     ```
     AZURE_OPENAI_API_KEY=@Microsoft.KeyVault(SecretUri=https://<your-keyvault-name>.vault.azure.net/secrets/AZURE-OPENAI-API-KEY)
     AZURE_OPENAI_ENDPOINT=your_endpoint
     AZURE_OPENAI_VERSION=2024-05-01-preview
     AZURE_OPENAI_DEPLOYMENT=your_deployment_name
     AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name
     SQL_CONNECTION_STRING=@Microsoft.KeyVault(SecretUri=https://<your-keyvault-name>.vault.azure.net/secrets/SQL-CONNECTION-STRING)
     API_URL=https://<your-app-name>.azurewebsites.net
     # Azure Web App specific settings
      HOST=0.0.0.0
      PORT=8000
     ```

### 4. Deploy the Application

1. Configure local Git deployment:
```bash
az webapp deployment source config-local-git --name <your-app-name> --resource-group <your-resource-group>
```

2. Add Azure remote to your Git repository:
```bash
git remote add azure <git-url-from-previous-command>
```

3. Deploy the application:
```bash
git push azure main
```

### 5. Verify Deployment

1. Check deployment status:
```bash
az webapp deployment list --name <your-app-name> --resource-group <your-resource-group>
```

2. View application logs:
```bash
az webapp log tail --name <your-app-name> --resource-group <your-resource-group>
```

3. Access your application at `https://<your-app-name>.azurewebsites.net`

## Usage

1. Enter a Person ID in the sidebar
2. Click "Process Person Data" to load and process the data
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
- Azure Key Vault integration for secure credential storage
- Ensure proper access controls on SQL Server
- Use appropriate Azure RBAC roles
- Automatic detection and prevention of restricted record processing

## Recent Updates

- Added restricted record detection and handling
- Enhanced citation display with source linking
- Added automatic processing on Enter key
- Improved error messages for restricted records
- Added recent query history for Person and Group IDs
- Added cloud deployment configuration
- Implemented automated vector store cleanup

## Troubleshooting

### Common Issues

1. Application fails to start:
   - Check application logs in Azure Portal
   - Verify all environment variables are set correctly
   - Ensure Python version matches requirements

2. Database connection issues:
   - Verify SQL Server firewall rules
   - Check connection string format
   - Ensure proper network security group rules

3. Vector store cleanup issues:
   - Check application logs for cleanup errors
   - Verify Qdrant client configuration
   - Monitor memory usage in Azure Portal

## License

This project is proprietary and confidential. 