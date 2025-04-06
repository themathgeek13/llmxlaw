from neo4j import GraphDatabase
from typing import List, Dict, Any

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password")  # Replace with your Neo4j credentials
        )

    def update_graph(self, cases: List[Dict[str, Any]]):
        """
        Update the knowledge graph with new cases
        """
        with self.driver.session() as session:
            for case in cases:
                # Create case node
                session.write_transaction(
                    self._create_case_node,
                    case
                )
                
                # Create relationships with arbitrators
                for arbitrator in case.get('arbitrators', []):
                    session.write_transaction(
                        self._create_arbitrator_relationship,
                        case['id'],
                        arbitrator
                    )

    @staticmethod
    def _create_case_node(tx, case: Dict[str, Any]):
        query = """
        MERGE (c:Case {id: $id})
        SET c.title = $title,
            c.date = $date,
            c.outcome = $outcome,
            c.challenge_grounds = $challenge_grounds
        """
        tx.run(query, {
            'id': case['id'],
            'title': case.get('title'),
            'date': case.get('date'),
            'outcome': case.get('challenge_outcome'),
            'challenge_grounds': case.get('challenge_grounds')
        })

    @staticmethod
    def _create_arbitrator_relationship(tx, case_id: str, arbitrator: Dict[str, Any]):
        query = """
        MATCH (c:Case {id: $case_id})
        MERGE (a:Arbitrator {id: $arbitrator_id})
        SET a.name = $name
        MERGE (a)-[r:ARBITRATED_IN]->(c)
        SET r.role = $role
        """
        tx.run(query, {
            'case_id': case_id,
            'arbitrator_id': arbitrator['id'],
            'name': arbitrator.get('name'),
            'role': arbitrator.get('role')
        })

    def get_related_cases(self, arbitrator_id: str) -> List[Dict[str, Any]]:
        """
        Get all cases related to a specific arbitrator
        """
        with self.driver.session() as session:
            result = session.read_transaction(
                self._get_related_cases,
                arbitrator_id
            )
            return result

    @staticmethod
    def _get_related_cases(tx, arbitrator_id: str):
        query = """
        MATCH (a:Arbitrator {id: $arbitrator_id})-[:ARBITRATED_IN]->(c:Case)
        RETURN c
        """
        result = tx.run(query, arbitrator_id=arbitrator_id)
        return [dict(record['c']) for record in result] 