from google import genai
from typing import List, Dict, Any, Optional
import json
import os

class RAGSystem:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.model = "gemini-2.0-flash"
        self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def get_response(self, query: str, cases: List[Dict[str, Any]]) -> str:
        """
        Get a context-aware response using Gemini
        """
        # Prepare context from cases
        context = self._prepare_context(cases)
        
        # Check if we have enough context
        if not context or context.strip() == "":
            return """
Conclusion:

With the context provided, I cannot provide any meaningful analysis specific to Mexico or any other aspect of arbitrator challenges. The information is simply too limited. If you can provide more data, I would be happy to perform a more detailed and relevant analysis.
"""
        
        # Create prompt with context and query
        prompt = f"""
        You are an expert in international arbitration, specializing in arbitrator challenges and conflicts of interest.
        
        Context from relevant cases:
        {context}
        
        Question: {query}
        
        Based ONLY on the context provided (do not use prior knowledge), provide a detailed analysis focusing on:
        1. Patterns in arbitrator challenges (if any can be identified from the context)
        2. Common grounds for challenges (if specified in the context)
        3. Outcomes of the challenges and their implications
        4. Any notable trends or insights, especially focusing on cases involving Mexico
        
        If the information provided is insufficient to draw conclusions on any of these points, explicitly state this limitation.
        Format your response with clear headings and bullet points where appropriate.
        Cite specific cases from the context when making observations.
        """
        
        try:
            # Get response from Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            
            # Extract entities and relations from the response
            self._extract_and_update_graph(response.text)
            
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _prepare_context(self, cases: List[Dict[str, Any]]) -> str:
        """
        Prepare detailed context string from cases
        """
        if not cases:
            return ""
            
        context_parts = []
        
        for case in cases:
            case_id = case.get('id', 'Unknown')
            attributes = case.get('attributes', {})
            title = attributes.get('title', 'Untitled Case')
            organization = attributes.get('organization', 'Unknown Organization')
            outcome = attributes.get('outcome', 'Unknown Outcome')
            commencement_date = attributes.get('commencement_date', 'Unknown Date')
            
            # Extract enriched parties information
            enriched_parties = case.get('enriched_parties', [])
            parties_info = []
            has_mexico = False
            
            for party in enriched_parties:
                party_attrs = party.get('attributes', {})
                party_name = party_attrs.get('name', 'Unknown Party')
                party_role = party_attrs.get('role', 'Unknown Role')
                party_type = party_attrs.get('type', 'Unknown Type')
                party_nationality = party_attrs.get('nationality', 'Unknown Nationality')
                
                # Check if Mexico is involved
                if 'Mexico' in party_name or 'Mexican' in party_nationality:
                    has_mexico = True
                
                parties_info.append(f"- {party_name} ({party_role}, {party_type}, Nationality: {party_nationality})")
            
            # Extract enriched decisions information
            enriched_decisions = case.get('enriched_decisions', [])
            decisions_info = []
            challenge_info = []
            arbitrators_info = []
            
            for decision in enriched_decisions:
                decision_attrs = decision.get('attributes', {})
                decision_id = decision.get('id', 'Unknown')
                decision_title = decision_attrs.get('title', f'Decision {decision_id}')
                decision_type = decision_attrs.get('type', 'Unknown Type')
                decision_date = decision_attrs.get('date', 'Unknown Date')
                decision_status = decision_attrs.get('status', 'Unknown Status')
                decision_description = decision_attrs.get('description', '')
                
                # Check if this decision is a challenge decision
                is_challenge = any(term in decision_title.lower() or term in decision_description.lower() 
                                for term in ['challenge', 'disqualification', 'recusal', 'conflict of interest'])
                
                decision_summary = f"- Decision {decision_id}: {decision_title} ({decision_type}, {decision_date}, {decision_status})"
                decisions_info.append(decision_summary)
                
                # Extract challenge information if present
                if is_challenge:
                    challenge_grounds = ""
                    challenge_outcome = ""
                    
                    # Try to extract challenge grounds and outcome from description
                    if decision_description:
                        # Extract potential grounds
                        if "grounds" in decision_description.lower():
                            parts = decision_description.split("grounds")
                            if len(parts) > 1:
                                challenge_grounds = parts[1].split(".")[0]
                        
                        # Extract potential outcome
                        outcome_indicators = ["dismissed", "rejected", "accepted", "upheld", "granted"]
                        for indicator in outcome_indicators:
                            if indicator in decision_description.lower():
                                idx = decision_description.lower().find(indicator)
                                challenge_outcome = decision_description[idx:idx+50].split(".")[0]
                                break
                    
                    challenge_info.append(f"- Challenge in Decision {decision_id}: {decision_title}")
                    if challenge_grounds:
                        challenge_info.append(f"  Grounds: {challenge_grounds}")
                    if challenge_outcome:
                        challenge_info.append(f"  Outcome: {challenge_outcome}")
                
                # Extract arbitrator information if available
                individuals = decision.get('individuals', [])
                for individual in individuals:
                    individual_attrs = individual.get('attributes', {})
                    individual_name = individual_attrs.get('name', 'Unknown Individual')
                    individual_role = individual_attrs.get('role', 'Unknown Role')
                    
                    # Check if this individual is an arbitrator
                    if 'arbitrator' in individual_role.lower() or 'president' in individual_role.lower() or 'chairman' in individual_role.lower():
                        arbitrators_info.append(f"- {individual_name} ({individual_role})")
            
            # Construct case context
            case_context = f"""
Case: {title} (ID: {case_id})
Organization: {organization}
Outcome: {outcome}
Commencement Date: {commencement_date}
Mexico Involvement: {"Yes" if has_mexico else "No"}

Parties:
{chr(10).join(parties_info) if parties_info else "No party information available"}

Decisions:
{chr(10).join(decisions_info) if decisions_info else "No decision information available"}

Arbitrators:
{chr(10).join(arbitrators_info) if arbitrators_info else "No arbitrator information available"}

Challenges:
{chr(10).join(challenge_info) if challenge_info else "No challenges identified"}
"""
            context_parts.append(case_context)
        
        return "\n\n---\n\n".join(context_parts)

    def _extract_and_update_graph(self, response_text):
        """
        Extract entities and relations from the response and update the knowledge graph
        """
        try:
            # Extract entities and relations using Gemini
            extraction_prompt = f"""
            Extract the following from the text:
            1. Arbitrators and their roles
            2. Cases and their outcomes
            3. Challenge grounds and their frequency
            4. Relationships between entities
            
            Text to analyze:
            {response_text}
            
            Return the result in JSON format with the following structure:
            {{
                "entities": {{
                    "arbitrators": [{{"name": "...", "role": "..."}}],
                    "cases": [{{"title": "...", "outcome": "..."}}],
                    "challenges": [{{"grounds": "...", "frequency": "..."}}]
                }},
                "relations": [
                    {{"from": "...", "to": "...", "type": "..."}}
                ]
            }}
            """
            
            extraction_response = self.client.models.generate_content(
                model=self.model,
                contents=extraction_prompt
            )
            
            # Parse the extraction result
            extraction_text = extraction_response.text
            
            # Remove backticks if present
            if extraction_text.startswith("```json") and extraction_text.endswith("```"):
                extraction_text = extraction_text[7:-3]
            elif extraction_text.startswith("```") and extraction_text.endswith("```"):
                extraction_text = extraction_text[3:-3]
            
            extraction_result = json.loads(extraction_text)
            
            # Update the knowledge graph
            self._update_graph_from_extraction(extraction_result)
            
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")

    def _update_graph_from_extraction(self, extraction_result):
        """
        Update the knowledge graph with extracted entities and relations
        """
        try:
            # Add arbitrator entities
            for arbitrator in extraction_result.get("entities", {}).get("arbitrators", []):
                arbitrator_name = arbitrator.get("name")
                if arbitrator_name:
                    arbitrator_id = f"extracted_arbitrator_{arbitrator_name.replace(' ', '_')}"
                    self.knowledge_graph.graph.add_node(
                        arbitrator_id,
                        type="arbitrator",
                        name=arbitrator_name,
                        role=arbitrator.get("role", "Unknown"),
                        source="extraction"
                    )
            
            # Add case entities
            for case in extraction_result.get("entities", {}).get("cases", []):
                case_title = case.get("title")
                if case_title:
                    case_id = f"extracted_case_{case_title.replace(' ', '_')[:30]}"
                    self.knowledge_graph.graph.add_node(
                        case_id,
                        type="case",
                        title=case_title,
                        outcome=case.get("outcome", "Unknown"),
                        source="extraction"
                    )
            
            # Add challenge entities
            for challenge in extraction_result.get("entities", {}).get("challenges", []):
                challenge_grounds = challenge.get("grounds")
                if challenge_grounds:
                    challenge_id = f"extracted_challenge_{challenge_grounds.replace(' ', '_')[:30]}"
                    self.knowledge_graph.graph.add_node(
                        challenge_id,
                        type="challenge",
                        grounds=challenge_grounds,
                        frequency=challenge.get("frequency", "Unknown"),
                        source="extraction"
                    )
            
            # Add relationships
            for relation in extraction_result.get("relations", []):
                source = relation.get("from")
                target = relation.get("to")
                rel_type = relation.get("type")
                
                if source and target and rel_type:
                    # Create IDs for source and target
                    source_id = f"extracted_{'arbitrator' if 'arbitrator' in source.lower() else 'case'}_{source.replace(' ', '_')[:30]}"
                    target_id = f"extracted_{'arbitrator' if 'arbitrator' in target.lower() else 'case'}_{target.replace(' ', '_')[:30]}"
                    
                    # Check if nodes exist
                    if source_id in self.knowledge_graph.graph and target_id in self.knowledge_graph.graph:
                        self.knowledge_graph.graph.add_edge(
                            source_id,
                            target_id,
                            relationship_type=rel_type,
                            source="extraction"
                        )
        except Exception as e:
            print(f"Error updating graph: {str(e)}")

    def analyze_mexico_cases(self, cases: List[Dict[str, Any]]) -> str:
        """
        Perform specialized analysis of cases involving Mexico
        """
        # Filter cases related to Mexico
        mexico_cases = []
        for case in cases:
            attributes = case.get('attributes', {})
            title = attributes.get('title', '')
            
            # Check if Mexico is mentioned in the title
            if 'Mexico' in title:
                mexico_cases.append(case)
                continue
                
            # Check if Mexico is mentioned in parties
            for party in case.get('enriched_parties', []):
                party_attrs = party.get('attributes', {})
                party_name = party_attrs.get('name', '')
                party_nationality = party_attrs.get('nationality', '')
                
                if 'Mexico' in party_name or 'Mexican' in party_nationality:
                    mexico_cases.append(case)
                    break
        
        if not mexico_cases:
            return "No cases involving Mexico were found in the provided data."
            
        # Create prompt for Mexico-specific analysis
        context = self._prepare_context(mexico_cases)
        prompt = f"""
        You are an expert in international investment arbitration, specializing in cases involving Mexico.
        
        Context from relevant cases involving Mexico:
        {context}
        
        Based ONLY on the context provided (do not use prior knowledge), provide a detailed analysis focusing on:
        1. Common themes in arbitrator challenges in cases involving Mexico
        2. Grounds for challenges and their success rates
        3. Patterns in outcomes of cases involving Mexico
        4. Insights regarding Mexico's approach to arbitrator appointments and challenges
        
        Format your response with clear headings and bullet points where appropriate.
        Cite specific cases from the context when making observations.
        If the information provided is insufficient to draw conclusions on any specific point, explicitly state this limitation.
        """
        
        try:
            # Get response from Gemini
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generating Mexico analysis: {str(e)}"

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