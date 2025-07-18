import streamlit as st
import requests
import json
import re
import time
import time
import os
from dotenv import load_dotenv
from datetime import date  
 
# Load environment variables
load_dotenv()

# Get the API URL from environment variable or use default
#API_URL = os.getenv("API_URL", "http://localhost:8000")
API_URL = "https://mosaic-assist.azurewebsites.net"

# Configure the page
st.set_page_config(
    page_title="Mosaic AI Assistant PoC",
    page_icon="🤖",
    layout="wide"
)

# Add a check for API availability
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.status_code == 200
    except:
        return False

# Show a warning if API is not available
if not check_api_health():
    st.error("⚠️ The API service is currently unavailable. Some features may not work.")
    st.stop()

# Initialize session state variables
if "recent_person_queries" not in st.session_state:
    st.session_state.recent_person_queries = []
if "recent_group_queries" not in st.session_state:
    st.session_state.recent_group_queries = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Person Search"
if "sources" not in st.session_state:
    st.session_state.sources = {}  # Store sources for citations
if "current_citations" not in st.session_state:
    st.session_state.current_citations = None
if "active_citation" not in st.session_state:
    st.session_state.active_citation = None
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None
 

# Chat Section
st.header("Chat with Data")

# Configure the sidebar for search functionalities
st.sidebar.header("Search Options")

# Sidebar filter for LoadDate  
load_date_after = st.sidebar.date_input("Load Date After (Filter)", value=None)  

# Add vector store cleanup button to sidebar
if st.sidebar.button("Clean Up Vector Store", key="cleanup_button"):
    try:
        response = requests.post(f"{API_URL}/vector/cleanup")
        response.raise_for_status()
        st.sidebar.success("Vector store cleaned up successfully!")
        st.session_state.messages = []  # Clear chat history
        
        # Clear any active citations
        for key in list(st.session_state.keys()):
            if key.startswith("active_citation_"):
                del st.session_state[key]
                
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error cleaning up: {str(e)}")

# Person Search in Sidebar
st.sidebar.subheader("Search by Person ID")

person_id = st.sidebar.text_input(
     "Enter Person ID",
     value = None, step=1, placeholder=15
 )

# Add new person ID to recent queries if it's not empty and not already in the list
if person_id and person_id not in st.session_state.recent_person_queries:
    try:
        person_id_int = int(person_id)
        st.session_state.recent_person_queries.insert(0, person_id_int)
        # Keep only the last 10 queries
        st.session_state.recent_person_queries = st.session_state.recent_person_queries[:10]
    except ValueError:
        st.sidebar.error("Please enter a valid number")

# Process person data when Enter is pressed or button is clicked
if st.sidebar.button("Process Person Data") or (person_id and st.session_state.get("person_id_input") != ""):
    try:
        person_id_int = int(person_id)
        with st.spinner("Processing person data..."):
            print(f"Making request to: {API_URL}/person/process")
            response = requests.post(
                f"{API_URL}/person/process",
                 json={"person_id": person_id_int},
                 headers= {"Content-Type": "application/json"}
            )
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text}")
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "error":
                st.error(result["message"])
            else:
                st.success(result["message"])
    except requests.exceptions.RequestException as e:
        print(f"Request error: {str(e)}")
        st.error(f"Error processing person data: {str(e)}")
    except ValueError:
        st.error("Please enter a valid number")

# Group Search in Sidebar
st.sidebar.subheader("Search by Group ID")

group_id = st.sidebar.text_input(
     "Enter Group ID",
     key="group_id_input",
     placeholder="Type ID and press Enter"
 )

# Add new group ID to recent queries if it's not empty and not already in the list
if group_id and group_id not in st.session_state.recent_group_queries:
    try:
        group_id_int = int(group_id)
        st.session_state.recent_group_queries.insert(0, group_id_int)
        # Keep only the last 10 queries
        st.session_state.recent_group_queries = st.session_state.recent_group_queries[:10]
    except ValueError:
        st.sidebar.error("Please enter a valid number")

# Process group data when Enter is pressed or button is clicked
if st.sidebar.button("Process Group Data") or (group_id and st.session_state.get("group_id_input") != ""):
    try:
        group_id_int = int(group_id)
        with st.spinner("Processing group data..."):
            response = requests.post(
                f"{API_URL}/group/process",
                 json={"group_id": group_id_int}
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "error":
                st.error(result["message"])
            else:
                st.success(result["message"])
    except ValueError:
        st.error("Please enter a valid number")
    except requests.exceptions.RequestException as e:
        st.error(f"Error processing group data: {str(e)}")

# Citations Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.header("Reference Materials")

# Track the currently displayed citations
if "current_citations" not in st.session_state:
    st.session_state.current_citations = None

# Initialize session state for active citation
if "active_citation" not in st.session_state:
    st.session_state.active_citation = None

# Initialize session state for tracking selected citation
if "selected_citation" not in st.session_state:
    st.session_state.selected_citation = None

def select_citation(citation_id):
    st.session_state.selected_citation = citation_id

# Main Page Content
st.title("Mosaic AI Assistant PoC")
st.markdown("""
This application allows you to interact with Mosaic data using natural language queries.
You can either search by person ID or group ID to access relevant information.
""")

# Create a dictionary to track newest message positions
newest_msg = {"user": None, "assistant": None}

# Add a function to create a clickable citation link
def create_citation_link(citation):
    return f"[{citation['id']} ({citation['type']})](javascript:void(0) '{citation['id']}')"

# Chat input
if prompt := st.chat_input("Ask a question about the data"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Query the vector index for relevant context using our enhanced retrieval
    with st.spinner("Retrieving relevant information..."):
        try:
            vector_response = requests.post(
                f"{API_URL}/vector/query",
                 json={"query": prompt, "top_k": 30}
            )
            vector_response.raise_for_status()
            documents = vector_response.json().get("documents", [])
            
            # Store source documents for citation
            sources = {}
            formatted_docs = []
            
            # Format documents as context with citation IDs
            if documents:
                for i, doc in enumerate(documents):
                    citation_id = i + 1
                    
                    # Get doc_type and identifier from metadata
                    metadata = doc.get("metadata", {})
                    doc_type = metadata.get("doc_type", "Unknown")
                    identifier = metadata.get("identifier", "")
                    id_tag = metadata.get("id", "")  # Get the ID tag from metadata
                    
                    # Extract any dates if present
                    date_info = ""
                    date_patterns = [r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})']
                    for pattern in date_patterns:
                        dates = re.findall(pattern, doc["text"])
                        if dates:
                            date_info = f" ({dates[0]})"
                            break
                    
                    # Look for specific identifiers in the text if not in metadata
                    identifier_info = ""
                    
                    if not identifier:
                         doc_lower = doc["text"].lower()
                         
                         # Case ID pattern
                         case_id_match = re.search(r'case\s*(?:id|note\s*id|number)[:\s]*(\d+)', doc_lower)
                         if case_id_match:
                             identifier_info = f" (Case ID: {case_id_match.group(1)})"
                         
                         # Form ID pattern
                         elif "form" in doc_lower:
                             form_id_match = re.search(r'form\s*(?:id|number)[:\s]*(\d+)', doc_lower)
                             if form_id_match:
                                 identifier_info = f" (Form ID: {form_id_match.group(1)})"
                         
                         # Subgroup ID pattern
                         elif "subgroup" in doc_lower:
                             subgroup_id_match = re.search(r'subgroup\s*(?:id|number)[:\s]*(\d+)', doc_lower)
                             if subgroup_id_match:
                                 identifier_info = f" (Subgroup ID: {subgroup_id_match.group(1)})"
                    
                    # Choose the more informative identifier
                    metadata_suffix = identifier_info if identifier_info else date_info
                    
                    # Store both document and its metadata with enhanced information
                    sources[citation_id] = {
                        "text": doc["text"],
                        "type": doc_type + metadata_suffix,
                        "id": citation_id,
                        "id_tag": id_tag,  # Add the ID tag to the source
                        "excerpt": doc["text"][:150] + "..." if len(doc["text"]) > 150 else doc["text"]
                    }
                    
                    # Format with citation ID and metadata
                    formatted_docs.append(f"Document [{citation_id}] ({doc_type}):\n{doc['text']}")
                
                context = "\n\n".join(formatted_docs)
                st.session_state["last_context_length"] = len(documents)
            else:
                context = "No relevant documents found."
                st.session_state["last_context_length"] = 0
        except requests.exceptions.RequestException as e:
            error_message = f"Error retrieving context: {str(e)}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            # Instead of 'return', use a context variable and break the flow
            context = "Unable to retrieve context."
            continue_execution = False
        else:
            continue_execution = True
            
        # Only proceed if we have valid context
        if continue_execution:
            # Format the prompt with context for the LLM
            full_prompt = (
                f"You are an AI assistant helping with queries about vulnerable children data from a social services database. "
                f"Below is the relevant information retrieved from the database about the person or group in question.\n\n"
                f"Use the following context to answer the user's question. The context contains factual information from "
                f"real child services records. When you use information from a specific document, cite the source using the document number in square brackets, "
                f"like this: [1], [2], etc. MAKE SURE to include these citations for every piece of information you use from the documents.\n\n"
                f"If the answer is in the context, provide it with as much relevant detail as possible. "
                f"If you cannot find the answer or reasonably infer it from the context, "
                f"say 'I don't have that specific information in the records available to me.'\n\n"
                f"**DO NOT** provide social care advice OR advice on next steps OR subsequent actions."
                f"CONTEXT INFORMATION:\n{context}\n\n"
                f"USER QUESTION: {prompt}\n\n"
                f"ANSWER (with citations in square brackets):"
            )
            
            # Get AI response with context
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                         f"{API_URL}/chat/with_context",
                        json={"prompt": full_prompt}
                    )
                    response.raise_for_status()
                    result = response.json()
                    response_text = result["response"]
                    
                    # Process citations and add them to the message
                    citations = []
                    used_citation_ids = set()
                    
                    try:
                        # Check for citation patterns like [1], [2], etc. in the response
                        citation_pattern = r'\[(\d+)\]'
                        all_citations = re.findall(citation_pattern, response_text)
                        
                        # Convert to integers and remove duplicates by using a set
                        unique_citations = set(map(int, all_citations))
                        
                        print(f"Found citations: {unique_citations}")
                        
                        # Add all cited sources to the citations list
                        for citation_id in unique_citations:
                            if citation_id <= len(documents) and citation_id in sources:  # Ensure the citation ID is valid
                                used_citation_ids.add(citation_id)
                                # Copy the source metadata to the citation
                                citations.append(sources[citation_id])
                            else:
                                print(f"Warning: Citation [{citation_id}] referenced but not found in sources")
                        
                        # Sort citations by ID for consistent presentation
                        citations.sort(key=lambda x: x["id"])
                    except Exception as e:
                        print(f"Error processing citations: {str(e)}")
                        # Fallback method implemented if needed
                    
                    # Create new message with all data
                    new_message = {
                        "role": "assistant", 
                        "content": response_text,
                        "citations": citations
                    }
                    
                    # Add the message to session state
                    st.session_state.messages.append(new_message)
                    
                    # Store sources for this response
                    st.session_state.sources[len(st.session_state.messages) - 1] = sources
                    
                except requests.exceptions.RequestException as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# Display chat history in chronological order (from oldest to newest)
for i, message in enumerate(st.session_state.messages):
    # Track the newest message of each type
    newest_msg[message["role"]] = i
    
    # Create a message container
    message_container = st.container()
    
    # Now display the chat message and its content
    with message_container.chat_message(message["role"]):
        # Display the message content
        st.write("**Warning!**  *This is AI generated content - always check veracity of source documentation before making any decisions on the data.*")
        st.markdown(message["content"])
        
        # For assistant messages with citations, add radio buttons for selection
        if message["role"] == "assistant" and "citations" in message and message["citations"]:
            st.session_state.current_citations = message["citations"]
            
            # Add radio buttons for citation selection
            st.write("View source document:")
            selected_idx = st.radio(
                "Select citation to view:",
                options=range(len(message["citations"])),
                format_func=lambda x: f"[{message['citations'][x]['id']}] {message['citations'][x]['type']} (ID: {message['citations'][x].get('id_tag', 'N/A')})",
                key=f"citation_radio_{i}",
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Update selected citation
            st.session_state.selected_citation = message["citations"][selected_idx]["id"]

# Check for citation selection in URL parameters
#query_params = st.experimental_get_query_params()
if "cite" in st.query_params:
    try:
        citation_id = int(st.query_params["cite"])
        st.session_state.selected_citation = citation_id
    except (ValueError, IndexError):
        pass

# Display citations in sidebar if they exist
if st.session_state.current_citations:
    citations = st.session_state.current_citations
    
    # Show selected citation first
    if st.session_state.selected_citation is not None:
        selected_citation = next(
            (c for c in citations if c['id'] == st.session_state.selected_citation),
            None
        )
        if selected_citation:
            st.sidebar.markdown(f"### Selected Source")
            st.sidebar.markdown(f"**[{selected_citation['id']}] {selected_citation['type']} (ID: {selected_citation.get('id_tag', 'N/A')})**")
            st.sidebar.text_area(
                "", 
                selected_citation['text'], 
                height=400, 
                disabled=True,
                key=f"sidebar_text_{selected_citation['id']}"
            )
else:
    st.sidebar.info("No references available for the current response.")

# Chat input with clear chat button
if "messages" in st.session_state and len(st.session_state.messages) > 0:
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Chat", key="clear_chat", type="primary"):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.rerun()