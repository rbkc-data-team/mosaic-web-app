from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from azure.ai.evaluation import (
    evaluate,
    #CoherenceEvaluator,
    F1ScoreEvaluator,
    #FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
    BleuScoreEvaluator,
    RetrievalEvaluator,
    RougeScoreEvaluator, RougeType,
    MeteorScoreEvaluator,
    ResponseCompletenessEvaluator
    #DocumentRetrievalEvaluator
)
import re
from collections import defaultdict
from datetime import datetime
import json
import os
from promptflow.core import AzureOpenAIModelConfiguration
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_DEPLOYMENT,)

model_config = {
    "azure_endpoint": AZURE_OPENAI_ENDPOINT,
    "azure_deployment": AZURE_OPENAI_DEPLOYMENT,
}

class EvaluationMetrics:
    def __init__(self, log_dir: str = "evaluation_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_history = []
        
    def evaluate_response(self, sources: dict, citations: list, query_context: str,user_query: str, prompt: str, response: str, retrieved_docs: Dict, ground_truth: str) -> Dict:
        """
        Evaluate the LLM's response for accuracy and consistency with the source documents.
        
        Args:
            sources: retrieved context in dictionary with these keys: text, type, id, id_tag, excerpt. Excerpt is first 150 characters of context string.
            citations: list of citations used in LLM response. Unique and ordered (created as a set).
            query_context: documents attached to prompt as context for LLM response (all context retrieved from vector store)
            user_query: The user's question
            prompt: entire message sent to llm, with associated context and system prompt
            response: The LLM's response
            retrieved_docs: The documents used as context
            ground_truth: The source documents for verification - these are all the documents pulled from the database, not just those retrieved from vector store
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Extract key facts from the source documents. breaks context docs into sentence length 'facts'
            source_facts = self._extract_facts(query_context)
                     
            ######################## not sure this is useful anymore - better to use claims dictionary
            # Extract claims from the LLM's response
            response_claims = self._extract_claims(response)
            print(f"Extracted {len(response_claims)} claims from response")#,"\nClaims: ",response_claims)
            
            # extracts each claim from response and assigns to dictionary as key with values being the citation(s) referenced in the response
            claims_dict = self._extract_claims_with_citations(response) #_extract_citations(response) 
            print("Claims extracted and attached to reference ID")

            # extracts all the retrieved context from prompt and arranges into a dictionary with citation numbers as keys
            context_dict = self._extract_context(query_context)
            print("Context extracted and attached to reference ID")

            ################# need to evaluate the usefullness of F1 score without a human made ground truth doc
            # Evaluate F1 score
            f1_score = self._get_documentf1_score(response_claims, source_facts)
            print(f"F1 score: {f1_score:.2f}")

            # Evaluate Rouge score
            precision, recall, f1 = self._getRougeScore(response, query_context)
            print(f"Rouge Precision Score: {precision:.2f}")
            print(f"Rouge Recall Score: {recall:.2f}")
            print(f"Rouge F1 Score: {f1:.2f}")

            # Evaluate groundedness
            groundedness, ground_explanation = self._getGroundedness_evaluator(response, ground_truth)
            print(f"Groundedness: {groundedness}")

            # Evaluate similarity
            similarity, sim_explanation = self._similarity_evaluation(prompt, response, ground_truth)
            print(f"Similarity Score: {similarity}")

            # Evaluate retrieval
            retrieval_score, retr_reason = self._getRetrieval_evaluator(context_dict, user_query, response)
            print(f"Retrieval Score: {retrieval_score}")
            
            # # Evaluate document retrieval
            # doc_retrieval_score, retr_reason = self._getdoc_retrieval_eval(context_dict, user_query, response)
            # print(f"Document Retrieval Score: {doc_retrieval_score}")

            # METEOR Evaluation
            meteor_score = self._getMeteorEval(claims_dict,context_dict)
            print(f"METEOR Score: {meteor_score}")
            
            # Evaluate citation accuracy
            citation_accuracy = self._evaluate_citations(response, retrieved_docs)
            print(f"Citation accuracy score: {citation_accuracy:.2f}")
            
            # Evaluate response relevance
            relevance_score = self._evaluate_relevance_tfidf(prompt, response)
            print(f"Relevance score: {relevance_score:.2f}")

            # Evaluate response relevance
            azure_relevance_score, relevance_reason, relevance_result = self._getRelevance_evaluator(user_query, response)
            print(f"Azure Relevance score: {azure_relevance_score:.2f}")#, f"Relevance Reason: {relevance_reason}", f"Relevance Result: {relevance_result}")
                       
            # Evaluate response completeness - changed from TF-IDF metric to LLM as a judge
            completeness_score, completeness_reason = self._evaluate_completeness(response, query_context)
            print(f"Completeness score: {completeness_score:.2f}")
            
            metrics = {
                "Groundedness": groundedness,
                "completeness_score": completeness_score,
                "Retrieval": retrieval_score,
                "llm_relevance": azure_relevance_score,
                "Similarity": similarity,
                "F1_score": f1_score,
                "citation_accuracy": citation_accuracy,
                "relevance_score (TF-IDF)": relevance_score,
                "METEOR_score": meteor_score
            }
            eval_docs = {}
            for item in range(len(retrieved_docs)):
                data = retrieved_docs[item]["metadata"]
                eval_docs.update({item:data})
            
            # Log the evaluation results
            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "query": user_query,
                "full prompt": prompt,
                "response": response,
                "metrics": metrics,
                "retrieved docs count": len(retrieved_docs),
                "groundedness explanation": ground_explanation["groundedness_reason"],
                "Completeness reason": completeness_reason,
                "retrieval score reason": retr_reason,
                "LLM relevance reason": relevance_reason,
                "claims dictionary": claims_dict,
                "context dictionary": context_dict
            }
            #############################Need to define where this gets logged to in web app########################################
            #self._log_evaluation(evaluation_results)
            
            self.metrics_history.append(metrics)
            return {"metrics": metrics}
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            import traceback
            print(traceback.format_exc())
            return {"metrics": {
                "F1_score": 0.0,
                "Groundedness": 0.0,
                "Similarity": 0.0,
                "Retrieval": 0.0,
                "citation_accuracy": 0.0,
                "relevance_score": 0.0,
                "llm_relevance": 0.0,
                "completeness_score": 0.0,
                "METEOR_score": 0.0,
                "error": str(e)
            }}
    
    #################not sure how useful extract facts function is, think about removal
    def _extract_facts(self, text: str) -> List[str]:
        """Extract key facts from source documents."""
        # can adapt by only pulling those facts that have been cited in the answer instead of all supporting documents
        facts = []
        # Split into sentences
        sentences = text.split('.')
        for sentence in sentences:
            # Clean and normalize
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only consider substantial sentences
                facts.append(sentence)
        #print("Evaluation Facts: ",facts)
        return facts   
    
    def _extract_context(self,query_context: str):
        # Regular expression to find document references and their content
        pattern = re.compile(r'Document \[(\d+)\] \(.*?\):\s*(.*?)\s*(?=Document \[\d+\] \(.*?\):|$)', re.DOTALL)
    
        # Find all matches in the context documents
        matches = pattern.findall(query_context)
    
        # Create a dictionary to store document content with citation numbers as keys
        documents_dict = defaultdict(str)
    
        for citation, content in matches:
            documents_dict[int(citation)] = content.strip()

        #print(documents_dict)
        return dict(documents_dict)
 
    
    def _extract_claims_with_citations(self, text):
        # Match all claim + citation groups
        #pattern = re.compile(r'(.*?)(\s*(\[\d+\])+)', re.DOTALL)
        pattern = re.compile(r'(.*?(?:\[\d+\])+[.?!])', re.DOTALL) # changed regex pattern to handle instances where llm included commas between citations
        matches = pattern.findall(text)

        claim_dict = {}
        
        for match in matches:
            # Extract citations from the sentence
            citations = re.findall(r'\[(\d+)\]', match)
            # Clean up the claim text (remove citations)
            claim_text = re.sub(r'(\s*\[\d+\])+', '', match).strip()
            claim_dict[claim_text] = citations

        new_dict = {}

        # re-order dict so that the citation number is the key and each claim associated with that form a value or list of values
        for claim, citations in claim_dict.items():
            for citation in citations:
                if citation not in new_dict:
                    new_dict[citation] = []
                new_dict[citation].append(claim)

        return new_dict


    def _extract_claims(self, text: str) -> List[str]:
        """Extract claims from the LLM's response."""
        claims = []
        # Split into sentences
        sentences = text.split('.')
        for sentence in sentences:
            # Clean and normalize
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only consider substantial sentences
                claims.append(sentence)
        #print("Evaluation Claims: ",claims)
        return claims  

    ####### can probably get rid of this one - it's always 1, not providing much value       
    def _evaluate_citations(self, response: str, documents: List[Dict]) -> float:
        """Evaluate the accuracy of citations in the response."""
        if not documents:
            return 0.0
            
        # Extract citation numbers from response
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, response)
        
        if not citations:
            return 0.0
            
        # Check if citations are valid
        valid_citations = 0
        for citation in citations:
            try:
                idx = int(citation) - 1  # Convert to 0-based index
                if 0 <= idx < len(documents):
                    valid_citations += 1
            except ValueError:
                continue
                
        return round(valid_citations / len(citations),1) if citations else 0.0
    
    def _evaluate_relevance_tfidf(self, prompt: str, response: str) -> float:
        """Evaluate how relevant the response is to the query."""
        if not prompt or not response:
            return 0.0
            
        # Use TF-IDF to measure similarity between query and response
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform([prompt, response])
            similarity = (tfidf_matrix[0] @ tfidf_matrix[1].T).toarray()[0][0]
            return round(float(similarity),3)
        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return 0.0
 
    # changed below function to use azure.ai.evaluation metric for completeness     
    def _evaluate_completeness(self, response: str, context: List[str]) -> float:
        """Given ground truth response, ResponseCompletenessEvaluator that captures the recall aspect of response alignment with the expected response. """
        #print(context)
        response_completeness = ResponseCompletenessEvaluator(model_config=model_config, threshold=3)
        results = response_completeness(
            response=response,
            ground_truth=context
        )
        return results['response_completeness'],results['response_completeness_reason']
                    
    
    def _log_evaluation(self, evaluation_results: Dict[str, Any]):
        """Log evaluation results to a JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"evaluation_{timestamp}.json")
        
        try:
            with open(log_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
        except Exception as e:
            print(f"Error logging evaluation results: {e}")

# this probably will only work properly if we have human created ground truth answers to compare against
    def _get_documentf1_score(self, claims: List[str], facts: List[str]) -> float:
        """ Calcuate F1 score:
            precision = 1.0 * num_common_tokens / len(prediction_tokens)
            recall = 1.0 * num_common_tokens / len(reference_tokens)

            f1 = (2.0 * precision * recall) / (precision + recall)""" 
        if not claims or not facts:
            return 0.0
        
        eval_fn = F1ScoreEvaluator()
        try:
            # Calculate similarity between claims and facts
            scores = []
            
            for i in range(len(facts)):
                claim_scores = []
                for c in range(len(claims)):
                    f1_result = eval_fn(response= claims[c],ground_truth= facts[i])
                    claim_scores.append(f1_result["f1_score"])
                
                scores.append(max(claim_scores)) # return the most highest value = claim most similar to fact           
                
            # Return average score
            return round(sum(scores) / len(scores),3) if scores else 0.0
            
        except Exception as e:
            print(f"Error in F1 evaluation: {e}")
            return 0.0
               
    def _similarity_evaluation (self, prompt: str, response: str, ground_truth: str) -> int:
        """Evaluates similarity score for a given prompt, response, and ground truth.

            The similarity measure evaluates the likeness between a ground truth sentence (or document) and the
            AI model's generated prediction. This calculation involves creating sentence-level embeddings for both
            the ground truth and the model's prediction, which are high-dimensional vector representations capturing
            the semantic meaning and context of the sentences.

            Use it when you want an objective evaluation of an AI model's performance, particularly in text generation
            tasks where you have access to ground truth responses. Similarity enables you to assess the generated
            text's semantic alignment with the desired content, helping to gauge the model's quality and accuracy.

            Similarity scores range from 1 to 5, with 1 being the least similar and 5 being the most similar.  3 is the default pass threshold """
        # this is an LLM as the judge evaluation

        eval_fn = SimilarityEvaluator(model_config)
        result = eval_fn(
            query=prompt,
            response=response,
            ground_truth= ground_truth)
        #print("similarity result: ",result)
        score = result["similarity"]

        return score, result if score else 0
    
    def _getMeteorEval (self, claim: dict, reference: dict ) -> float: 
        # ground truth can be used if we have an empirical grount truth dataset collated by social workers       
        """Calculates the METEOR score for a given response and ground truth.

        The METEOR (Metric for Evaluation of Translation with Explicit Ordering) score grader evaluates generated text by comparing it to reference texts, 
        focusing on precision, recall, and content alignment. It addresses limitations of other metrics like BLEU by considering synonyms, 
        stemming, and paraphrasing. METEOR score considers synonyms and word stems to more accurately capture meaning and language variations.
        In addition to machine translation and text summarization, paraphrase detection is an optimal use case for the METEOR score.

        Use the METEOR score when you want a more linguistically informed evaluation metric that captures not only n-gram overlap but also accounts 
        for synonyms, stemming, and word order. This is particularly useful for evaluating tasks like machine translation, text summarization, 
        and text generation.

        The METEOR score ranges from 0 to 1, with 1 indicating a perfect match. https://www.cs.cmu.edu/~alavie/METEOR/pdf/Lavie-Agarwal-2007-METEOR.pdf"""

        scores = [] # placeholder list for individual scores for each claim
        
        combined_dict = {}
        for citation, claims in claim.items():
            if int(citation) in reference:
                combined_dict[citation] = {
                    "context": reference[int(citation)],
                    "claims": claims
                }
    
        meteor_evaluator = MeteorScoreEvaluator(alpha=0.8, threshold=0.3)
        for key, value in combined_dict.items():
            ref = value['context']
            for c in value['claims']:
                result = meteor_evaluator(response=c, ground_truth=ref)
                #meteor_score = result['meteor_score'], result['meteor_result']
                scores.append(result['meteor_score'])
        
        # Return average score
        return round(sum(scores) / len(scores),3) if scores else 0.0
        
    def _getBleuScore (self):
        """Evaluator that computes the BLEU Score between two strings.
            BLEU (Bilingual Evaluation Understudy) score is commonly used in natural language processing (NLP) and machine translation. 
            It is widely used in text summarization and text generation use cases. It evaluates how closely the generated text matches the reference text. 
            The BLEU score ranges from 0 to 1, with higher scores indicating better quality."""
        
        eval_fn = BleuScoreEvaluator()
        result = eval_fn(
            response="",
            ground_truth="")

    def _getRougeScore (self, response: str, ground_truth: str):
        """Evaluator for computes the ROUGE scores between two strings.

        ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used to evaluate automatic summarization and machine translation. 
        It measures the overlap between generated text and reference summaries. ROUGE focuses on recall-oriented measures to assess how well the generated text covers the reference text. 
        Text summarization and document comparison are among optimal use cases for ROUGE, particularly in scenarios where text coherence and relevance are critical."""
        
        eval_fn = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_1)
        result = eval_fn(
            response=response,
            ground_truth=ground_truth)
        precision = result['rouge_precision']
        recall = result['rouge_recall']
        f1 = result['rouge_f1_score']
        return precision, recall, f1        
        

    def _getRelevance_evaluator (self, user_query: str, response: str):
        """measures how effectively a response addresses a query. 
        It assesses the accuracy, completeness, and direct relevance of the response based solely on the given query. Higher scores mean better relevance. """
        
        eval_fn = RelevanceEvaluator(model_config)
        result = eval_fn(
            query=user_query,
            response=response)
        
        return result['relevance'], result['relevance_reason'], result['relevance_result']
        
    def _getGroundedness_evaluator (self, response: str, ground_truth: str) -> int:
        """Evaluates groundedness score for a given query (optional), response, and context or a multi-turn conversation,
        including reasoning.

        The groundedness measure assesses the correspondence between claims in an AI-generated answer and the source
        context, making sure that these claims are substantiated by the context. Even if the responses from LLM are
        factually correct, they'll be considered ungrounded if they can't be verified against the provided sources
        (such as your input source or your database). Use the groundedness metric when you need to verify that
        AI-generated responses align with and are validated by the provided context.

        Groundedness scores range from 1 to 5, with 1 being the least grounded and 5 being the most grounded.
        Default threshold for pass is 3 """
        # this is an LLM as the judge evaluation
     
        eval_fn = GroundednessEvaluator(model_config) #, http_client= httpx.Client(verify=False))
        result = eval_fn(
            response= response,
            context= ground_truth)
        
        score = result["groundedness"]
        return score, result if score else 0
        
    def _getRetrieval_evaluator (self, context_dict: dict ,user_query: str, response: str):
        """Evaluates retrieval score for a given query and context or a multi-turn conversation, including reasoning.

        The retrieval measure assesses the AI system's performance in retrieving information
        for additional context (e.g. a RAG scenario).

        Retrieval scores range from 1 to 5, with 1 being the worst and 5 being the best."""

        # turn context dict into a single string to pass into the retrieval evaluator
        context = []
        for key, value in context_dict.items():
            context.append(value)
        context_str = " ".join(context)
        
        chat_eval = RetrievalEvaluator(model_config)
        result = chat_eval(query = user_query, context = context_str)
       
        return result["retrieval"], result['retrieval_reason']
    
    ############## need annotated ground truth dataset to run this function
    def _getdoc_retrieval_eval (self, context_dict: dict ,user_query: str, response: str):
        """Document Retrieval evaluator measures how well the RAG retrieves the correct documents from the document store. 
        As a composite evaluator useful for RAG scenario with ground truth, it computes a list of useful search quality metrics for debugging your RAG pipelines.
        
        Fidelity, NDCG, XDCG, Max Relevance N, Holes"""
        # these query_relevance_label are given by your human- or LLM-judges.
        retrieval_ground_truth = [
            {
                "document_id": "1",
                "query_relevance_label": 4
            },]
        ground_truth_label_min = 0
        ground_truth_label_max = 4
        # these relevance scores come from your search retrieval system
        retrieved_documents = [
            {
                "document_id": "2",
                "relevance_score": 45.1
            },]
        
        document_retrieval_evaluator = DocumentRetrievalEvaluator(
            ground_truth_label_min=ground_truth_label_min, 
            ground_truth_label_max=ground_truth_label_max,
            ndcg_threshold = 0.5,
            xdcg_threshold = 50.0,
            fidelity_threshold = 0.5,
            top1_relevance_threshold = 50.0,
            top3_max_relevance_threshold = 50.0,
            total_retrieved_documents_threshold = 50,
            total_ground_truth_documents_threshold = 50
        )
        results = document_retrieval_evaluator(retrieval_ground_truth=retrieval_ground_truth, retrieved_documents=retrieved_documents)   
       
        return results

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluation metrics."""
        if not self.metrics_history:
            return {
                "total_evaluations": 0,
                "average_metrics": {
                    "F1_score": 0.0,
                    "Groundedness": 0.0,
                    "Similarity": 0.0,
                    "Retrieval": 0.0,
                    "citation_accuracy": 0.0,
                    "relevance_score (TF-IDF)": 0.0,
                    "llm_relevance":0.0,
                    "completeness_score": 0.0,
                    "METEOR_score": 0.0
                }
            }   

        # Calculate averages
        total = len(self.metrics_history)
        avg_metrics = {
            "F1_score": sum(m["F1_score"] for m in self.metrics_history) / total,
            "Groundedness": sum(m["Groundedness"] for m in self.metrics_history) / total,
            "Similarity": sum(m["Similarity"] for m in self.metrics_history) / total,
            "Retrieval": sum(m["Retrieval"] for m in self.metrics_history) / total,
            "citation_accuracy": sum(m["citation_accuracy"] for m in self.metrics_history) / total,
            "relevance_score (TF-IDF)": sum(m["relevance_score (TF-IDF)"] for m in self.metrics_history) / total,
            "llm_relevance": sum(m["llm_relevance"] for m in self.metrics_history) / total,
            "completeness_score": sum(m["completeness_score"] for m in self.metrics_history) / total,
            "METEOR_score": sum(m["METEOR_score"] for m in self.metrics_history) / total
        }
        
        return {
            "total_evaluations": total,
            "average_metrics": avg_metrics
        }