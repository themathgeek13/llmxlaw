import requests
from typing import List, Dict, Any

class JusMundiAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.jusmundi.com/stanford"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def search_cases(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for cases related to arbitrator challenges using the Stanford API
        """
        search_params = {
            "query": query,
            "filters": {
                "has_arbitrator_challenge": True,
                "document_type": "decision"
            },
            "limit": 10,
            "offset": 0
        }
        
        response = requests.post(
            f"{self.base_url}/search",
            headers=self.headers,
            json=search_params
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json().get('results', [])

    def get_case_details(self, case_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific case
        """
        response = requests.get(
            f"{self.base_url}/cases/{case_id}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json()

    def get_arbitrator_challenges(self, arbitrator_id: str) -> List[Dict[str, Any]]:
        """
        Get all challenges related to a specific arbitrator
        """
        response = requests.get(
            f"{self.base_url}/arbitrators/{arbitrator_id}/challenges",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json().get('results', [])

    def get_related_cases(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Get cases related to a specific case
        """
        response = requests.get(
            f"{self.base_url}/cases/{case_id}/related",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.text}")
        
        return response.json().get('results', []) 