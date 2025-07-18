from fastapi import FastAPI, HTTPException, Depends, Request, Body, APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import streamlit as st
import streamlit.web.bootstrap as bootstrap
import threading
import os
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from dotenv import load_dotenv
import time
from data_processing import get_person_documents, get_group_documents, run_sql_queries
from vector_store import create_qdrant_store, create_vector_index, create_chat_engine, cleanup_vector_store, check_collection_status, get_qdrant_client
from llm_setup import setup_llama_index
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from collections import Counter
from evaluation_metrics import EvaluationMetrics

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mosaic-assist.azurewebsites.net/", "https://localhost:8501"], 
    allow_credentials=True,
    allow_methods=["GET","POST","DELETE","OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize evaluation metrics
evaluator = EvaluationMetrics()

# Create routers
main_router = APIRouter()
person_router = APIRouter(prefix="/person", tags=["person"])
group_router = APIRouter(prefix="/group", tags=["group"])
chat_router = APIRouter(prefix="/chat", tags=["chat"])
vector_router = APIRouter(prefix="/vector", tags=["vector"])
eval_router = APIRouter(prefix = "/eval_summary", tags=["eval"])

# Models for request validation
class PersonRequest(BaseModel):
    person_id: int

class GroupRequest(BaseModel):
    group_id: int

class ChatRequest(BaseModel):
    message: str

class VectorQueryRequest(BaseModel):
    query: str
    top_k: int = 15

class ContextQueryRequest(BaseModel):
    prompt: str
    query: str
    context: str
    person_id: Optional[int] = None
    group_id: Optional[int] = None
    sources: dict
    len_docs: int

# Helper functions for advanced retrieval
def preprocess_query(query: str) -> str:
    """
    Preprocess the query by removing special characters, lowercasing, etc.
    """
    # Remove special characters and extra whitespace
    query = re.sub(r'[^\w\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query.lower()

def expand_query(query: str) -> str:
    """
    Expand the query with related terms for better recall
    This is a simple implementation - could be enhanced with domain-specific synonyms
    """
    # Split the query into terms
    terms = query.split()
    
    # Add some common domain-specific expansions
    expansions = {
        "child": ["minor", "youth", "juvenile"],
        "address": ["location", "residence", "home"],
        "parent": ["guardian", "caregiver", "mother", "father"],
        "school": ["education", "academic", "learning"],
        "health": ["medical", "wellness", "condition"],
        "legal": ["law", "court", "judicial"]
    }
    
    # Expand the query
    expanded_terms = terms.copy()
    for term in terms:
        if term in expansions:
            expanded_terms.extend(expansions[term])
    
    return " ".join(expanded_terms)

def calculate_keyword_similarity(query: str, documents: List[str]) -> List[float]:
    """
    Calculate keyword-based similarity scores using TF-IDF
    """
    if not documents:
        return []
        
    vectorizer = TfidfVectorizer()
    try:
        # Fit the vectorizer on the documents and query
        tfidf_matrix = vectorizer.fit_transform(documents + [query])
        
        # Get the query vector (last item)
        query_vector = tfidf_matrix[-1]
        
        # Calculate similarity with each document
        doc_vectors = tfidf_matrix[:-1]
        similarities = (doc_vectors @ query_vector.T).toarray().flatten()
        
        return similarities
    except Exception as e:
        print(f"Error calculating keyword similarity: {e}")
        return [0.0] * len(documents)

def rerank_results(semantic_scores: List[float], keyword_scores: List[float], 
                   documents: List[str], alpha: float = 0.7) -> List[tuple]:
    """
    Rerank results using a weighted combination of semantic and keyword scores.

    semantic_score: Measures how conceptually similar a document is to the query (using embeddings & cosine similarity).
    keyword_score: Measures how many exact or partial keyword matches exist between the document and the query.
    alpha: A weight between 0 and 1 that controls the influence of each score. If alpha = 1.0: Only semantic scores are used, If alpha = 0.0: Only keyword scores are used.
        - If alpha = 0.7 (default): 70% of the final score comes from semantic similarity, and 30% from keyword matching.
    """
    if not documents:
        return []
        
    # Handle numpy arrays by converting to list if needed
    if hasattr(keyword_scores, 'tolist'):
        keyword_scores = keyword_scores.tolist()
    if hasattr(semantic_scores, 'tolist'):
        semantic_scores = semantic_scores.tolist()
    
    # Normalize scores
    max_semantic = max(semantic_scores) if semantic_scores else 1.0
    max_keyword = max(keyword_scores) if keyword_scores else 1.0
    
    norm_semantic = [s/max_semantic for s in semantic_scores] if max_semantic > 0 else [0.0] * len(documents)
    norm_keyword = [k/max_keyword for k in keyword_scores] if max_keyword > 0 else [0.0] * len(documents)
    
    # Combine scores with weighting
    combined_scores = [alpha * s + (1-alpha) * k for s, k in zip(norm_semantic, norm_keyword)]
    #print("Combine Score",combined_scores)
    
    # Create tuples of (document, score)
    scored_docs = [(doc, score) for doc, score in zip(documents, combined_scores)]
    
    # Sort by score in descending order
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return scored_docs

# Dependency to setup the index
async def get_llm_and_embed_model():
    return setup_llama_index()

# Main router endpoints
@main_router.get("/")
async def root():
    return RedirectResponse(url="/streamlit")

@main_router.get("/health")
async def health_check():
    return {"status": "healthy"}

@main_router.get("/")
async def root():
    return RedirectResponse(url="/streamlit")

@main_router.get("/streamlit")
async def streamlit_app(request: Request):
    # Start Streamlit in a separate thread if not already running
    if not hasattr(app, 'streamlit_thread') or not app.streamlit_thread.is_alive():
        def run_streamlit():
            logging.info("Starting Streamlit subprocess...")
            # Set Streamlit configuration
            os.environ['STREAMLIT_SERVER_PORT'] = '8000'
            os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'  # Changed from localhost to 0.0.0.0
            os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
            os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
           
            # Run Streamlit
            bootstrap.run(
                Path(__file__).parent / "app.py",
                '',
                [],
                flag_options={}
            )
            logging.info("Streamlit subprocess launched")
       
        app.streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
        app.streamlit_thread.start()

    # get the host URL
    base_url = str(request.base_url).rstrip('/')
    
    streamlit_url = f"https://mosaic-assist.azurewebsites.net"

    # Return a simple HTML page that embeds the Streamlit iframe
    return HTMLResponse(f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Mosaic Assistant</title>
            <style>
                body, html {{
                    margin: 0;
                    padding: 0;
                    height: 100%;
                    overflow: hidden;
                }}
                iframe {{
                    width: 100%;
                    height: 100vh;
                    border: none;
                }}
            </style>
        </head>
        <body>
            </iframe> <iframe src="{streamlit_url}" frameborder="0"></iframe> 
        </body>
    </html>
    """)
                # <iframe src="{request.base_url}streamlit" frameborder="0"></iframe> <iframe src="{streamlit_url}" frameborder="0"></iframe> 
# Person router endpoints
@person_router.post("/process")
async def process_person(request: PersonRequest, llm_and_embed=Depends(get_llm_and_embed_model)):
    print(f"Processing person with ID: {request.person_id}")
    try:
        llm, embed_model = llm_and_embed
        person_key = request.person_id

        print(f"Processing documents for person {person_key}")
        #get documents and check for restricted records
        documents, is_restricted, message = get_person_documents(person_key)
        print(f"Retrieved {len(documents) if documents else 0} documents, restricted {is_restricted}")
        if is_restricted:
            return {"message": message, "status": "error"}
        
        if not documents:
            return {"message": "No documents found for this person.", "status": "error"}
        
        # create vector store
        vector_store = create_qdrant_store("person_store")

        # add docs to vectore store
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        index.insert_nodes(documents)
        print(f"Successfully processed {len(documents)} documents for person {person_key}")

        return {"message": f"Successfully processed {len(documents)} documents for person {person_key}", "status":"success"}
    
    except Exception as e:
        print(f"Error in processing person: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Group router endpoints
@group_router.post("/process")
async def process_group(request: GroupRequest, llm_and_embed=Depends(get_llm_and_embed_model)):
    print(f"Processing group with ID: {request.group_id}")
    try:
        llm, embed_model = llm_and_embed
        group_key = request.group_id

        documents, is_restricted, message = get_group_documents(group_key)

        if is_restricted:
            return {"message": message, "status": "error"}
        
        if not documents:
            return {"message": "No documents found for this group.", "status":"error"}
        
        #create vector store
        vector_store = create_qdrant_store("group_store")

        #add documents to vectore store
        index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=embed_model
        )
        index.insert_nodes(documents)

        return {"message": f"Successfully processed {len(documents)} documents for group {group_key}", "status":"success"}

    except Exception as e:
        print(f"Error in processing group: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Chat router endpoints
@chat_router.post("/")
async def chat(request: ChatRequest, llm_and_embed=Depends(get_llm_and_embed_model)):
    try:
        llm, embed_model = llm_and_embed
        chat_engine = create_chat_engine(llm, "vector_store")
        
        # Get response from the chat engine
        response = chat_engine.chat(request.message)
        return {"response": str(response)}
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@vector_router.post("/cleanup")
async def cleanup():
    try:
        cleanup_vector_store()
        return {"message": "Vector store cleaned up successfully"}
    except Exception as e:
        print(f"Error in cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up vector store: {str(e)}")

# Vector router endpoints
@vector_router.post("/query")
async def query_vector_index(request: VectorQueryRequest, llm_and_embed=Depends(get_llm_and_embed_model)):
    try:
        llm, embed_model = llm_and_embed
        
        # Step 1: Preprocess the query
        preprocessed_query = preprocess_query(request.query)
        print(f"Preprocessed query: {preprocessed_query}")
        
        # Step 2: Expand the query with related terms
        expanded_query = expand_query(preprocessed_query)
        print(f"Expanded query: {expanded_query}")
        
        # Step 3: Check if collections have documents
        person_has_docs = check_collection_status("person_store")
        group_has_docs = check_collection_status("group_store")
        
        if not person_has_docs and not group_has_docs:
            print("WARNING: No documents found in any store! Please process a person or group first.")
            return {"documents": ["No documents have been processed yet. Please process a person or group first."]}
            
        # Try person store first if it has documents
        documents = []
        semantic_scores = []
         
        if person_has_docs:
            print("Querying person_store...")
            try:
                # Create vector store and query engine
                vector_store = create_qdrant_store("person_store")
                
                # Create retriever directly for more control
                retriever = VectorStoreIndex.from_vector_store(
                    vector_store,
                    embed_model=embed_model
                ).as_retriever(similarity_top_k=30)  # Set higher k for more results
                
                # Query directly with retriever for more control
                retrieval_nodes = retriever.retrieve(expanded_query)
                
                if retrieval_nodes:
                    print(f"Retrieved {len(retrieval_nodes)} nodes from person_store")
                    for i, node in enumerate(retrieval_nodes):
                        doc_text = node.get_content()
                        score = getattr(node, 'score', 1.0)
                        metadata = node.metadata if hasattr(node, 'metadata') else {}
                        
                        # Debug output
                        truncated = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        truncated = truncated.replace('\n', ' ')
                        print(f"Doc {i+1}: Score={score:.4f}, Content={truncated}")
                        
                        documents.append({
                             "text": doc_text,
                             "score": score,
                             "metadata": metadata
                         })
                        semantic_scores.append(score)
                else:
                    print("No results from person_store retriever")
            except Exception as e:
                print(f"Error querying person_store: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Try group store if person store had no results or if we should try both
        if (not documents) and group_has_docs:
            print("Querying group_store...")
            try:
                # Create vector store and query engine
                vector_store = create_qdrant_store("group_store")
                
                # Create retriever directly for more control
                retriever = VectorStoreIndex.from_vector_store(
                    vector_store,
                    embed_model=embed_model
                ).as_retriever(similarity_top_k=15)  # Set higher k for more results
                
                # Query directly with retriever for more control
                retrieval_nodes = retriever.retrieve(expanded_query)
                
                if retrieval_nodes:
                    print(f"Retrieved {len(retrieval_nodes)} nodes from group_store")
                    for i, node in enumerate(retrieval_nodes):
                        doc_text = node.get_content()
                        score = getattr(node, 'score', 1.0) # this is the cosine similarity score from within the vector store
                        metadata = node.metadata if hasattr(node, 'metadata') else {}
                        
                        # Debug output
                        truncated = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        truncated = truncated.replace('\n', ' ')
                        print(f"Doc {i+1}: Score={score:.4f}, Content={truncated}")
                        
                        documents.append({
                             "text": doc_text,
                             "score": score,
                             "metadata": metadata
                         })
                        semantic_scores.append(score)
                else:
                    print("No results from group_store retriever")
            except Exception as e:
                print(f"Error querying group_store: {e}")
                import traceback
                print(traceback.format_exc())
        
        # Log the retrieved documents
        if documents:
                        
            # Calculate keyword similarity and rerank if we have documents
            keyword_scores = calculate_keyword_similarity(request.query, [doc["text"] for doc in documents])
            
            semantic_scores = semantic_scores if len(semantic_scores) == len(documents) else [1.0] * len(documents)
            
            reranked_docs = rerank_results(semantic_scores, keyword_scores, [doc["text"] for doc in documents])
                        
            #mapping of document text to all doc data
            doc_map = {doc["text"]: doc for doc in documents}
        
            # Return documents with their metadata
            return {
                 "documents": [
                     {
                         "text": doc_text,
                         "metadata": doc_map[doc_text]["metadata"],
                         "score":score
                     }
                     for doc_text,score in reranked_docs[:request.top_k]
                 ]
             }
        else:
            # No documents found
            print("No documents found in any store that match the query")
            return {"documents": ["No relevant documents found that match your query."]}
            
    except Exception as e:
        print(f"Error in query_vector_index: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error querying vector index: {str(e)}")

async def get_documents(person_id: Optional[int] = None, group_id: Optional[int] = None, format: str = "text") -> Union[str, List[Dict]]:
    """
    Get documents for a person or group.
    
    Args:
        person_id: The person ID if available
        group_id: The group ID if available
        format: The return format - either "text" (concatenated string) or "dict" (list of document dictionaries)
        
    Returns:
        Either a string of concatenated document contents or a list of document dictionaries
    """
    try:
        if person_id is not None:
            print(f"Retrieving documents for person_id: {person_id}")
            documents, is_restricted, message = get_person_documents(person_id)
        elif group_id is not None:
            print(f"Retrieving documents for group_id: {group_id}")
            documents, is_restricted, message = get_group_documents(group_id)
        else:
            return "No ID provided" if format == "text" else []
            
        if is_restricted:
            return "Record is restricted" if format == "text" else []
        if not documents:
            return "No documents found" if format == "text" else []
            
        if format == "text":
            return "\n".join([doc.get_content() for doc in documents])
        else:
            return [
                {
                    "text": doc.get_content(),
                    "id_tag": doc.metadata.get('id', ''),
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return "Error retrieving documents" if format == "text" else []


@chat_router.post("/with_context")
async def chat_with_context(request: ContextQueryRequest, llm_and_embed=Depends(get_llm_and_embed_model)):
    try:
        llm, embed_model = llm_and_embed
        
        # Use the prompt directly with the context already included
        response = llm.complete(request.prompt)
        response_text = str(response)

        # Get ground truth and relevant documents using the same function 
        ground_truth = await get_documents(
            person_id=request.person_id,
            group_id=request.group_id,
            format="text"
        )
        relevant_docs = await get_documents(
            person_id=request.person_id,
            group_id=request.group_id,
            format="dict"
        )
        
        # Process citations and add them to the message
        citations = []
        used_citation_ids = set()
        sources = request.sources # sources is retrieved context in dictionary with these keys: text, type, id, id_tag, excerpt
        #print("\nSources: ",sources)
        len_docs = request.len_docs

        try:
            # Check for citation patterns like [1], [2], etc. in the response
            citation_pattern = r'\[(\d+)\]'
            all_citations = re.findall(citation_pattern, response_text)
            
            # Convert to integers and remove duplicates by using a set
            unique_citations = set(map(int, all_citations))
            
            print(f"Found citations: {unique_citations}")
            
            # Add all cited sources to the citations list
            for citation_id in unique_citations:
                if citation_id <= len_docs and citation_id in sources:  # Ensure the citation ID is valid
                    used_citation_ids.add(citation_id)
                    # Copy the source metadata to the citation
                    citations.append(sources[citation_id])
            
            # Sort citations by ID for consistent presentation
            citations.sort(key=lambda x: x["id"])
        except Exception as e:
            print(f"Error processing citations: {str(e)}")
            # Fallback method implemented if needed

        # Evaluate the response using the new metrics
        evaluation_results = evaluator.evaluate_response(
            sources = request.sources,
            citations= list(unique_citations),
            query_context = request.context,
            user_query = request.query,
            prompt=request.prompt, #full prompt with system message and attached context
            response=response_text,
            retrieved_docs=relevant_docs,
            ground_truth=ground_truth
        )
        
        return {
            "response": response_text,
            "evaluation": evaluation_results
        }
    except Exception as e:
        print(f"Error in chat_with_context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@eval_router.post("/eval_summary")
async def get_evaluation_summary():
    """Get summary of all evaluation metrics"""
    try:
        summary = evaluator.get_evaluation_summary()
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting evaluation summary: {str(e)}")

# Include all routers in the main app
app.include_router(main_router)
app.include_router(person_router)
app.include_router(group_router)
app.include_router(chat_router)
app.include_router(vector_router)
app.include_router(eval_router)

# Shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up vector store when the application shuts down"""
    print("Cleaning up vector store on application shutdown...")
    cleanup_vector_store()

if __name__ == "__main__":
    # Get port from environment variable or default to 8000
     port = int(os.getenv("PORT", 8000))
     # Use 0.0.0.0 to allow external connections
     host = os.getenv("HOST", "0.0.0.0")
     uvicorn.run(app, host=host, port=port) 