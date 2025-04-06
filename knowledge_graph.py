import networkx as nx
from typing import List, Dict, Any
import os
from google import genai
import json

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.case_nodes = set()
        self.party_nodes = set()
        self.decision_nodes = set()
        
        # Initialize Gemini
        self.genai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

    def extract_case_parties(self, title: str) -> tuple:
        """
        Extract claimant and respondent from case title
        """
        if ' v. ' in title:
            claimant, respondent = title.split(' v. ')
            return claimant.strip(), respondent.strip()
        return None, None

    def extract_entities_and_relations(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use Gemini to extract entities and relationships from enriched case data
        """
        # Prepare case text for analysis
        attributes = case.get('attributes', {})
        enriched_parties = case.get('enriched_parties', [])
        enriched_decisions = case.get('enriched_decisions', [])
        
        case_text = f"""
        Title: {attributes.get('title', '')}
        Organization: {attributes.get('organization', '')}
        Outcome: {attributes.get('outcome', '')}
        Date: {attributes.get('commencement_date', '')}
        
        Parties:
        {json.dumps(enriched_parties, indent=2)}
        
        Decisions:
        {json.dumps(enriched_decisions, indent=2)}
        """
        
        prompt = f"""
        Analyze this arbitration case and extract the following:
        1. All parties involved and their roles (claimants, respondents, etc.)
        2. Key dates and events
        3. Type of dispute and subject matter
        4. Main issues or grounds for the case
        5. Decisions and their implications
        6. Relationships between entities
        7. Any monetary amounts or awards mentioned
        8. Jurisdiction and applicable laws
        
        Case details:
        {case_text}
        
        Format the response as JSON with these keys:
        {{
            "parties": [
                {{"name": "", "role": "", "type": "", "description": ""}}
            ],
            "dates": [
                {{"date": "", "event": ""}}
            ],
            "dispute_type": "",
            "subject_matter": "",
            "issues": [],
            "decisions": [
                {{"id": "", "type": "", "outcome": "", "date": "", "description": ""}}
            ],
            "relationships": [
                {{"source": "", "target": "", "type": "", "description": ""}}
            ],
            "monetary_amounts": [
                {{"amount": "", "currency": "", "description": ""}}
            ],
            "jurisdiction": {{"forum": "", "applicable_law": "", "treaty": ""}}
        }}
        """
        
        try:
            response = self.genai_client.generate_content(prompt)
            extracted_info = json.loads(response.text)
            return extracted_info
        except Exception as e:
            print(f"Error extracting entities: {str(e)}")
            return {}

    def update_graph(self, cases: List[Dict[str, Any]]):
        """
        Update the knowledge graph with enriched case data
        """
        for case in cases:
            case_id = case.get('id')
            if not case_id:
                continue

            # Extract entities and relationships using Gemini
            extracted_info = self.extract_entities_and_relations(case)
            
            # Add case node
            attributes = case.get('attributes', {})
            self.graph.add_node(
                case_id,
                type='case',
                title=attributes.get('title'),
                organization=attributes.get('organization'),
                outcome=attributes.get('outcome'),
                date=attributes.get('commencement_date'),
                jurisdiction=extracted_info.get('jurisdiction', {}),
                dispute_type=extracted_info.get('dispute_type'),
                subject_matter=extracted_info.get('subject_matter')
            )
            self.case_nodes.add(case_id)

            # Add parties from extracted info
            for party in extracted_info.get('parties', []):
                party_name = party.get('name')
                if party_name:
                    party_id = f"party_{party_name}"
                    self.graph.add_node(
                        party_id,
                        type='party',
                        name=party_name,
                        role=party.get('role'),
                        party_type=party.get('type'),
                        description=party.get('description')
                    )
                    self.party_nodes.add(party_id)
                    
                    # Add relationship between party and case
                    self.graph.add_edge(
                        party_id,
                        case_id,
                        relationship_type=party.get('role', 'involved_in'),
                        description=party.get('description')
                    )

            # Add decisions from extracted info
            for decision in extracted_info.get('decisions', []):
                decision_id = decision.get('id')
                if decision_id:
                    self.graph.add_node(
                        decision_id,
                        type='decision',
                        decision_type=decision.get('type'),
                        outcome=decision.get('outcome'),
                        date=decision.get('date'),
                        description=decision.get('description')
                    )
                    self.decision_nodes.add(decision_id)
                    
                    # Add relationship between decision and case
                    self.graph.add_edge(
                        case_id,
                        decision_id,
                        relationship_type='has_decision',
                        description=decision.get('description')
                    )

            # Add additional relationships
            for rel in extracted_info.get('relationships', []):
                source = rel.get('source')
                target = rel.get('target')
                if source and target:
                    # Add edge if both nodes exist
                    if source in self.graph and target in self.graph:
                        self.graph.add_edge(
                            source,
                            target,
                            relationship_type=rel.get('type'),
                            description=rel.get('description')
                        )

            # Add issue nodes
            for i, issue in enumerate(extracted_info.get('issues', [])):
                issue_id = f"issue_{case_id}_{i}"
                self.graph.add_node(
                    issue_id,
                    type='issue',
                    description=issue
                )
                self.graph.add_edge(
                    case_id,
                    issue_id,
                    relationship_type='has_issue'
                )

            # Add monetary amount nodes
            for i, amount in enumerate(extracted_info.get('monetary_amounts', [])):
                amount_id = f"amount_{case_id}_{i}"
                self.graph.add_node(
                    amount_id,
                    type='monetary_amount',
                    amount=amount.get('amount'),
                    currency=amount.get('currency'),
                    description=amount.get('description')
                )
                self.graph.add_edge(
                    case_id,
                    amount_id,
                    relationship_type='involves_amount'
                )

    def get_party_cases(self, party_name: str) -> List[Dict[str, Any]]:
        """
        Get all cases involving a specific party
        """
        party_id = f"party_{party_name}"
        if party_id not in self.graph:
            return []

        cases = []
        for neighbor in self.graph.neighbors(party_id):
            if self.graph.nodes[neighbor]['type'] == 'case':
                case_data = self.graph.nodes[neighbor]
                edge_data = self.graph.edges[(party_id, neighbor)]
                cases.append({
                    'id': neighbor,
                    'title': case_data.get('title'),
                    'organization': case_data.get('organization'),
                    'outcome': case_data.get('outcome'),
                    'date': case_data.get('date'),
                    'relationship': edge_data.get('relationship_type')
                })
        return cases

    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a case including all related entities
        """
        if case_id not in self.graph:
            return {}

        case_data = self.graph.nodes[case_id]
        
        # Get all related entities
        related_entities = {
            'parties': [],
            'decisions': [],
            'issues': [],
            'monetary_amounts': [],
            'relationships': []
        }
        
        for neighbor in self.graph.neighbors(case_id):
            node_data = self.graph.nodes[neighbor]
            edge_data = self.graph.edges[(case_id, neighbor)]
            
            node_type = node_data.get('type')
            if node_type == 'party':
                related_entities['parties'].append({
                    'id': neighbor,
                    'name': node_data.get('name'),
                    'role': node_data.get('role'),
                    'description': node_data.get('description'),
                    'relationship': edge_data.get('relationship_type')
                })
            elif node_type == 'decision':
                related_entities['decisions'].append({
                    'id': neighbor,
                    'type': node_data.get('decision_type'),
                    'outcome': node_data.get('outcome'),
                    'date': node_data.get('date'),
                    'description': node_data.get('description')
                })
            elif node_type == 'issue':
                related_entities['issues'].append({
                    'description': node_data.get('description')
                })
            elif node_type == 'monetary_amount':
                related_entities['monetary_amounts'].append({
                    'amount': node_data.get('amount'),
                    'currency': node_data.get('currency'),
                    'description': node_data.get('description')
                })

        return {
            'case_data': case_data,
            'related_entities': related_entities
        } 