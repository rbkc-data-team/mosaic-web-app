from typing import List, Tuple
import pandas as pd
from llama_index.core.schema import Document
from database import create_sql_connection, execute_query, close_connection
from llama_index.core.node_parser import SentenceSplitter
from fastapi import HTTPException
import tiktoken

def run_sql_queries(person_key: int) -> List[Tuple[str, pd.DataFrame]]:
    """Run all SQL queries for a person and return results as DataFrames"""
    conn = create_sql_connection()
    queries = {
        "demo_query": """SELECT 
            MosaicICSPersonDetails.Fullname, MosaicICSPersonAddresses.AddressType, MosaicICSPersonAddresses.FlatNumber, MosaicICSPersonAddresses.Building, 
            MosaicICSPersonAddresses.StreetNumber, MosaicICSPersonAddresses.Street, MosaicICSPersonAddresses.PostcodeKey, MosaicICSRelatedPersonDetails.Fullname as RelatedPerson_Name, 
            MosaicICSRelatedPersonDetails.PersonID, MosaicICSPersonDetails.Restricted
            FROM [DWH_Outputs].[dbo].[MosaicICSPersonDetails]
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSPersonAddresses ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSPersonAddresses.PersonKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSRelatedPersonDetails ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSRelatedPersonDetails.PersonDetailKey
            WHERE MosaicICSPersonDetails.PersonDetailKey={}
        """,
        "relationships_query": """SELECT 
            MosaicICSRelationships.RelationshipKey, MosaicICSRelationships.RelationshipTypeCode, MosaicICSRelationships.RelationshipClassification, MosaicICSRelationships.RelationshipClassCode, 
            MosaicICSRelationships.StartDateKey, MosaicICSRelationships.EndDateKey, MosaicICSRelationships.RelationshipType, MosaicICSRelationships.Relationship, 
            MosaicICSRelationships.RelatedToAs, MosaicICSRelationships.RelationshipFlag, MosaicICSRelationships.Relationship
            FROM [DWH_Outputs].[dbo].[MosaicICSRelationships]
            WHERE MosaicICSRelationships.PersonDetailKey={}
        """,
        "legal_query": """SELECT 
            MosaicICSLegalStatuses.LegalStatusKey, MosaicICSLegalStatuses.LegalStatusCode, MosaicICSLegalStatuses.LegalStatus, MosaicICSLegalStatuses.LegalStatusType, 
            MosaicICSLegalStatuses.StartDateKey, MosaicICSLegalStatuses.EndDateKey, MosaicICSLegalStatuses.Status, MosaicICSLegalStatuses.PersonKey
            FROM [DWH_Outputs].[dbo].[MosaicICSLegalStatuses]
            WHERE MosaicICSLegalStatuses.PersonKey={}
        """,
        "placements_query": """SELECT 
            MosaicICSPlacements.PlacementType as Placements, MosaicICSPlacements.PlacementTypeCode, MosaicICSPlacements.StartDateKey, MosaicICSPlacements.EndDateKey, 
            MosaicICSPlacements.EndReason, MosaicICSPlacements.ChangeReason, MosaicICSPlacements.CategoryOfNeed, MosaicICSPlacements.PersonKey, MosaicICSPlacements.ID
            FROM [DWH_Outputs].[dbo].[MosaicICSPlacements]
            WHERE MosaicICSPlacements.PersonKey={}
        """,
        "references_query": """SELECT 
            MosaicICSReferences.ReferenceType, MosaicICSReferences.Reference, MosaicICSReferences.[References], MosaicICSReferences.PersonKey, MosaicICSReferences.ReferenceKey
            FROM [DWH_Outputs].[dbo].[MosaicICSReferences]
            WHERE MosaicICSReferences.PersonKey={}
        """,
        "health_query": """SELECT 
            MosaicICSPersonDetails.Fullname, MosaicICSPersonDetails.DOBKey, MosaicICSPersonDetails.DODKey, MosaicICSPersonDetails.AgeEstimated, 
            MosaicICSPersonDetails.CountryOfBirth, MosaicICSPersonDetails.Gender, MosaicICSPersonDetails.Language, MosaicICSPersonDetails.Nationality, 
            MosaicICSPersonDetails.PersonType, MosaicICSPersonDetails.Restricted, MosaicICSDisabilities.DisabilityType, MosaicICSDisabilities.Disabilities, 
            MosaicICSHealthInterventions.HealthInterventionType, MosaicICSHealthInterventions.HealthInterventions, MosaicICSHealthInterventions.HealthInterventionKey, 
            MosaicICSHealthInterventions.InterventionDateKey, MosaicICSHealthInterventions.SDQ, MosaicICSHealthInterventions.Score, MosaicICSHealthInterventions.RequestedDateKey, 
            MosaicICSHealthInterventions.ID, MosaicICSImmunisations.ImmunisationType, MosaicICSImmunisations.AgeMonths, MosaicICSImmunisations.AgeYears, MosaicICSImmunisations.Immunisations
            FROM [DWH_Outputs].[dbo].[MosaicICSPersonDetails]
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSDisabilities ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSDisabilities.PersonKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSHealthInterventions ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSHealthInterventions.PersonKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSImmunisations ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSImmunisations.PersonKey
            WHERE MosaicICSPersonDetails.PersonDetailKey={}
        """,
        "forms_query": """SELECT 
            MosaicICSForms.FormID, MosaicICSForms.Question, MosaicICSForms.AnswerText, MosaicICSForms.AnswerLookup, MosaicICSForms.TemplateName, MosaicICSForms.CreatedDateKey, 
            MosaicICSForms.FinishedDateKey, MosaicICSWorkflowSteps.Status, MosaicICSSubgroupDetails.SubgroupID, MosaicICSPersonDetails.Fullname
            FROM [DWH_Outputs].[dbo].[MosaicICSForms]
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSWorkflowSteps ON MosaicICSWorkflowSteps.WorkflowStepKey=MosaicICSForms.WorkflowStepKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSSubgroupDetails ON MosaicICSSubgroupDetails.SubgroupKey=MosaicICSWorkflowSteps.SubgroupKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSPersonDetails ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSSubgroupDetails.PersonKey
            WHERE MosaicICSPersonDetails.PersonDetailKey={} 
            --AND MosaicICSForms.AnswerText != 'Not Applicable'
        """,
        "case_notes_query": """SELECT 
            MosaicICSCaseNotes.Title, MosaicICSCaseNotes.Notes, MosaicICSCaseNotes.CaseNoteID, MosaicICSCaseNotes.SubgroupKey, MosaicICSPersonDetails.Fullname, 
            MosaicICSPersonDetails.PersonDetailKey, MosaicWorkers.FullNameDisplay as WorkerName, MosaicOrganisations.Organisation, MosaicOrganisations.OrganisationType
            FROM [DWH_Outputs].[dbo].[MosaicICSCaseNotes]
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSSubgroupDetails ON MosaicICSSubgroupDetails.SubgroupKey=MosaicICSCaseNotes.SubgroupKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicICSPersonDetails ON MosaicICSPersonDetails.PersonDetailKey=MosaicICSSubgroupDetails.PersonKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicWorkers ON MosaicWorkers.WorkerKey=MosaicICSCaseNotes.WorkerKey
            LEFT JOIN [DWH_Outputs].[dbo].MosaicOrganisations ON MosaicOrganisations.OrganisationKey=MosaicWorkers.OrganisationKey
            WHERE MosaicICSPersonDetails.PersonDetailKey={}
        """
    }
    
    results = []
    try:
        for name, query in queries.items():
            formatted_query = query.format(person_key)
            df = execute_query(conn, formatted_query)
            if not df.empty:
                results.append((name, df))
    finally:
        close_connection(conn)
    
    return results

def split_text_into_chunks(text: str, chunk_size: int = 1500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of approximately equal size, trying to break at sentence boundaries.
    """
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Create sentence splitter with a custom tokenizer function
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        tokenizer=lambda x: tokenizer.encode(x)  # Return the tokens themselves, not their length
    )
    
    # Split the text into chunks
    chunks = splitter.split_text(text)
    return chunks

def df_to_documents(df: pd.DataFrame, name: str) -> List[Document]:
    """Convert DataFrame to Document list"""
    df_string = df.select_dtypes(include=['object'])
    cols = df_string.columns
    df.fillna("Not Recorded", inplace=True)
    df['doc_type'] = name
    df['combined_text'] = df[cols].agg('-'.join, axis=1)
    df['combined_text'] = df['combined_text'].astype(str)
    
    # Process each row and split if needed
    documents = []
    for _, row in df.iterrows():
        text = row['combined_text']
        # Only keep essential metadata
        metadata = {
             'doc_type': row['doc_type'],
             'person_ID': row.get('PersonID', row.get('PersonID', None)),
             'case_id': row.get('CaseNoteID', None),
             'form_id': row.get('FormID', None),
             'subgroup_id': row.get('SubgroupID', None),
             'restriction': row.get('Restricted', None)
         }
        # Add the appropriate ID based on document type
        if name == 'case_notes_query':
            metadata['id'] = row.get('CaseNoteID', None)
        elif name == 'forms_query':
            metadata['id'] = row.get('FormID', None)
        elif name == 'demo_query':
            metadata['id'] = row.get('PersonID', None)
        elif name == 'relationships_query':
            metadata['id'] = row.get('RelationshipKey', None)
        elif name == 'legal_query':
            metadata['id'] = row.get('LegalStatusKey', None)
        elif name == 'placements_query':
            metadata['id'] = row.get('ID', None)
        elif name == 'references_query':
            metadata['id'] = row.get('ReferenceKey', None)
        elif name == 'health_query':
            metadata['id'] = row.get('ID', None)
        
        # Split text if it's too large
        chunks = split_text_into_chunks(text)
        
        # Create documents for each chunk
        for i, chunk in enumerate(chunks):
            # If this is a split document, add chunk information to metadata
            if len(chunks) > 1:
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = i + 1
                chunk_metadata['total_chunks'] = len(chunks)
                #chunk_metadata['original_text'] = text  # Store original text for reference
                documents.append(Document(text=chunk, metadata=chunk_metadata))
            else:
                documents.append(Document(text=chunk, metadata=metadata))
    
    return documents

def check_restricted_records(documents: list[Document]) -> Tuple[bool, str]:
    """Checks if any docs have a restricted flag (Y) and returns (is_restricted, message)"""
    for doc in documents:
        if doc.metadata.get('restriction') == 'Y':
            doc_type = doc.metadata.get('doc_type', 'Unknown')
            person_id = doc.metadata.get('person_ID', 'Unknown')
            return True, f"This record is restricted (Person ID: {person_id}, Document Type: {doc_type})"
        return False, ""

def get_person_documents(person_key: int) -> Tuple[List[Document], bool, str]:
    """Get all data for a specific person, returns tuple of documents, boolean for restricted flag and user message"""
    conn = create_sql_connection()
    try:
        # Run all queries for the person
        query_results = run_sql_queries(person_key)
        
        # Process documents
        all_documents = []
        for name, df in query_results:
            documents = df_to_documents(df, name)
            all_documents.extend(documents)

        # check if any docs are restricted
        is_restricted, message = check_restricted_records(all_documents)
        
        return all_documents, is_restricted, message
    finally:
        close_connection(conn)

def get_group_documents(group_key: int) -> Tuple[List[Document], bool, str]:
    """Get all documents for a group and its members"""
    group_docs = []
    conn = create_sql_connection()
    try:
        print(f"Processing group key: {group_key}")
        
        # Get subgroup details
        print("Fetching subgroup details...")
        subgroup_query = """SELECT PersonKey FROM [DWH_Outputs].[dbo].[MosaicICSSubgroupDetails] WHERE GroupKey = {}""".format(group_key)
        group_query = """SELECT * FROM [DWH_Outputs].[dbo].[MosaicICSGroupDetails] WHERE GroupKey = {}""".format(group_key)
        
        group_sql = pd.read_sql(group_query, conn)
        subgroup_sql = pd.read_sql(subgroup_query, conn)
        
        if group_sql.empty or subgroup_sql.empty:
            print("No group or subgroup data found")
            return [], False, "No group or subgroup data found"
            
        pkey_list = subgroup_sql['PersonKey'].to_list()
        print(f"Found {len(pkey_list)} people in group")
        
        # Get data for each person in the group
        for i, person_key in enumerate(pkey_list, 1):
            print(f"Processing person {i}/{len(pkey_list)} (ID: {person_key})")
            try:
                dataset, is_restricted, message = get_person_documents(person_key)
                if is_restricted:
                    print(f"Found restricted record for person {person_key}")
                    return [], True, message
                if dataset:
                    print(f"Found {len(dataset)} documents for person {person_key}")
                    group_docs.extend(dataset)
                else:
                    print(f"No documents found for person {person_key}")
            except Exception as e:
                print(f"Error processing person {person_key}: {str(e)}")
                continue
            
        print(f"Total documents collected: {len(group_docs)}")
        return group_docs, False, ""
        
    except Exception as e:
        print(f"Error in get_group_documents: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        close_connection(conn)

def get_group_info(group_key: int) -> Tuple[str, str]:
    """Get group type and name for a given group key"""
    conn = create_sql_connection()
    try:
        group_query = """SELECT GroupType, GroupName FROM [DWH_Outputs].[dbo].[MosaicICSGroupDetails] WHERE GroupKey = {}""".format(group_key)
        group_sql = pd.read_sql(group_query, conn)
        
        if group_sql.empty:
            return "", ""
            
        return group_sql['GroupType'].iloc[0], group_sql['GroupName'].iloc[0]
        
    finally:
        close_connection(conn) 