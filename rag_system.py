import google.generativeai as genai
from google.generativeai.types import Tool, GenerateContentConfig, GoogleSearch
from typing import List, Dict, Any
import json

class RAGSystem:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.model = "gemini-2.0-flash"
        self.google_search_tool = Tool(
            google_search=GoogleSearch()
        )

    def get_response(self, query: str, cases: List[Dict[str, Any]]) -> str:
        """
        Get a context-aware response using Gemini with Google Search grounding
        """
        # Prepare context from cases
        context = self._prepare_context(cases)
        
        # Create prompt with context and query
        prompt = f"""
        You are an expert in international arbitration, specializing in arbitrator challenges and conflicts of interest.
        
        Context from relevant cases:
        {context}
        
        Question: {query}
        
        Please provide a detailed analysis based on the context provided, focusing on:
        1. Patterns in arbitrator challenges
        2. Common grounds for challenges
        3. Outcomes and their implications
        4. Any notable trends or insights
        """
        
        try:
            # Get response with Google Search grounding
            response = genai.generate_content(
                model=self.model,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[self.google_search_tool],
                    response_modalities=["TEXT"],
                )
            )
            
            # Extract entities and relations from the response
            self._extract_and_update_graph(response)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _prepare_context(self, cases: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from cases
        """
        context_parts = []
        
        for case in cases:
            case_context = f"""
            Case: {case.get('title', 'Untitled')}
            Date: {case.get('date', 'Unknown')}
            Challenge Grounds: {case.get('challenge_grounds', 'Not specified')}
            Outcome: {case.get('outcome', 'Unknown')}
            
            Arbitrators:
            """
            
            for arbitrator in case.get('arbitrators', []):
                arbitrator_context = f"""
                - {arbitrator.get('name', 'Unknown')}
                  Role: {arbitrator.get('role', 'Unknown')}
                  Challenges: {len(arbitrator.get('challenges', []))}
                """
                case_context += arbitrator_context
            
            context_parts.append(case_context)
        
        return "\n".join(context_parts)

    def _extract_and_update_graph(self, response):
        """
        Extract entities and relations from the response and update the knowledge graph
        """
        try:
            # Extract entities and relations using Gemini
            extraction_prompt = """
            Extract the following from the text:
            1. Arbitrators and their roles
            2. Cases and their outcomes
            3. Challenge grounds and their frequency
            4. Relationships between entities
            
            Return the result in JSON format with the following structure:
            {
                "entities": {
                    "arbitrators": [{"name": "...", "role": "..."}],
                    "cases": [{"title": "...", "outcome": "..."}],
                    "challenges": [{"grounds": "...", "frequency": "..."}]
                },
                "relations": [
                    {"from": "...", "to": "...", "type": "..."}
                ]
            }
            """
            
            extraction_response = genai.generate_content(
                model=self.model,
                contents=extraction_prompt + response.text
            )
            
            # Parse the extraction result
            extraction_result = json.loads(extraction_response.text)
            
            # Update the knowledge graph
            self._update_graph_from_extraction(extraction_result)
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")

    def _update_graph_from_extraction(self, extraction_result):
        """
        Update the knowledge graph with extracted entities and relations
        """
        # Add entities
        for arbitrator in extraction_result["entities"]["arbitrators"]:
            self.knowledge_graph.graph.add_node(
                f"extracted_{arbitrator['name']}",
                type="arbitrator",
                name=arbitrator["name"],
                role=arbitrator["role"],
                source="web"
            )
        
        for case in extraction_result["entities"]["cases"]:
            self.knowledge_graph.graph.add_node(
                f"extracted_{case['title']}",
                type="case",
                title=case["title"],
                outcome=case["outcome"],
                source="web"
            )
        
        # Add relations
        for relation in extraction_result["relations"]:
            self.knowledge_graph.graph.add_edge(
                f"extracted_{relation['from']}",
                f"extracted_{relation['to']}",
                type=relation["type"],
                source="web"
            )

    def get_similar_cases(self, case_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar cases based on embeddings
        """
        case = self.knowledge_graph.get_case_details(case_id)
        if not case:
            return []

        case_text = f"{case.get('title', '')} {case.get('challenge_grounds', '')} {case.get('outcome', '')}"
        similar_docs = self.vector_store.similarity_search(case_text, k=limit)
        
        return [doc.metadata for doc in similar_docs] 