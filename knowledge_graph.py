import networkx as nx
from typing import List, Dict, Any

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.case_nodes = set()
        self.arbitrator_nodes = set()

    def update_graph(self, cases: List[Dict[str, Any]]):
        """
        Update the knowledge graph with new cases
        """
        for case in cases:
            case_id = case.get('id')
            if not case_id:
                continue

            # Add case node if not exists
            if case_id not in self.case_nodes:
                self.graph.add_node(case_id, 
                                  type='case',
                                  title=case.get('title'),
                                  date=case.get('date'),
                                  outcome=case.get('outcome'),
                                  challenge_grounds=case.get('challenge_grounds'))
                self.case_nodes.add(case_id)

            # Add arbitrator nodes and relationships
            for arbitrator in case.get('arbitrators', []):
                arbitrator_id = arbitrator.get('id')
                if not arbitrator_id:
                    continue

                # Add arbitrator node if not exists
                if arbitrator_id not in self.arbitrator_nodes:
                    self.graph.add_node(arbitrator_id,
                                      type='arbitrator',
                                      name=arbitrator.get('name'))
                    self.arbitrator_nodes.add(arbitrator_id)

                # Add relationship between case and arbitrator
                self.graph.add_edge(case_id, arbitrator_id,
                                  role=arbitrator.get('role'))

                # Add challenge relationships
                for challenge in arbitrator.get('challenges', []):
                    challenge_id = f"{arbitrator_id}_challenge_{len(arbitrator.get('challenges', []))}"
                    self.graph.add_node(challenge_id,
                                      type='challenge',
                                      grounds=challenge.get('grounds'),
                                      outcome=challenge.get('outcome'))
                    self.graph.add_edge(arbitrator_id, challenge_id)

    def get_related_cases(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Get cases related to a specific case through arbitrators
        """
        if case_id not in self.graph:
            return []

        related_cases = []
        # Get all arbitrators in the case
        arbitrators = [n for n in self.graph.neighbors(case_id) 
                      if self.graph.nodes[n]['type'] == 'arbitrator']
        
        # For each arbitrator, find other cases they're involved in
        for arbitrator in arbitrators:
            cases = [n for n in self.graph.neighbors(arbitrator)
                    if self.graph.nodes[n]['type'] == 'case' and n != case_id]
            for case in cases:
                case_data = self.graph.nodes[case]
                related_cases.append({
                    'id': case,
                    'title': case_data.get('title'),
                    'date': case_data.get('date'),
                    'outcome': case_data.get('outcome')
                })

        return related_cases

    def get_arbitrator_challenges(self, arbitrator_id: str) -> List[Dict[str, Any]]:
        """
        Get all challenges related to a specific arbitrator
        """
        if arbitrator_id not in self.graph:
            return []

        challenges = []
        # Get all challenge nodes connected to the arbitrator
        challenge_nodes = [n for n in self.graph.neighbors(arbitrator_id)
                         if self.graph.nodes[n]['type'] == 'challenge']
        
        for challenge in challenge_nodes:
            challenge_data = self.graph.nodes[challenge]
            challenges.append({
                'grounds': challenge_data.get('grounds'),
                'outcome': challenge_data.get('outcome')
            })

        return challenges 