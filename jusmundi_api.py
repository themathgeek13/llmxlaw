import requests
from typing import List, Dict, Any, Optional
import logging
import json
import time

class JusMundiAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.jusmundi.com/stanford"
        self.headers = {
            "X-API-Key": api_key,
            "accept": "application/json"
        }
        self.logger = logging.getLogger(__name__)
        
        # Log initialization
        self.logger.info(f"Initializing JusMundiAPI with base_url: {self.base_url}")
        self.logger.info(f"Headers configured: {self.headers}")

    def _handle_response(self, response: requests.Response, endpoint: str) -> Dict[str, Any]:
        """
        Handle API response and errors
        """
        try:
            self.logger.info(f"API Response Status: {response.status_code} for {endpoint}")
            self.logger.debug(f"Response content: {response.text[:500]}")
            
            if response.status_code == 404:
                self.logger.warning(f"Resource not found at {endpoint}")
                return {"data": []}
            
            if response.status_code == 401:
                self.logger.error("Invalid API key or unauthorized access")
                raise Exception("Invalid API key or unauthorized access")
            
            if response.status_code == 429:
                self.logger.error("Rate limit exceeded")
                raise Exception("Rate limit exceeded")
            
            if response.status_code != 200:
                self.logger.error(f"API request failed: {response.text}")
                raise Exception(f"API request failed with status {response.status_code}")
            
            return response.json()
        except requests.exceptions.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from {endpoint}")
            return {"data": []}

    def get_party_details(self, party_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a party
        """
        endpoint = f"{self.base_url}/parties/{party_id}"
        self.logger.info(f"Getting party details for ID: {party_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', {})
        except Exception as e:
            self.logger.error(f"Error getting party details: {str(e)}")
            return {}

    def get_decision_details(self, decision_id: str, include: Optional[str] = "cases,individuals") -> Dict[str, Any]:
        """
        Get detailed information about a decision with related information
        """
        endpoint = f"{self.base_url}/decisions/{decision_id}"
        params = {}
        if include:
            params["include"] = include
            
        self.logger.info(f"Getting decision details for ID: {decision_id} with include: {include}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            result = data.get('data', {})
            
            # Include related entities if present
            if 'included' in data:
                result['included'] = data['included']
                
            return result
        except Exception as e:
            self.logger.error(f"Error getting decision details: {str(e)}")
            return {}
            
    def get_decision_individuals(self, decision_id: str) -> List[Dict[str, Any]]:
        """
        Get individuals associated with a specific decision
        """
        endpoint = f"{self.base_url}/decisions/{decision_id}/individuals"
        self.logger.info(f"Getting individuals for decision ID: {decision_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"Error getting decision individuals: {str(e)}")
            return []

    def enrich_case_data(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich case data with detailed information about parties and decisions
        """
        case_id = case.get('id')
        if not case_id:
            return case
            
        self.logger.info(f"Enriching data for case ID: {case_id}")
        
        # Get full case details with included relationships
        detailed_case = self.get_case_details(case_id, include="decisions,parties")
        if not detailed_case:
            return case
            
        enriched_case = detailed_case.copy()
        
        # Process included items to organize them
        included_items = detailed_case.get('included', [])
        parties_data = []
        decisions_data = []
        
        for item in included_items:
            item_type = item.get('type')
            if item_type == 'parties':
                parties_data.append(item)
            elif item_type == 'decisions':
                decisions_data.append(item)
        
        # Add detailed party information
        enriched_parties = []
        for party in parties_data:
            party_id = party.get('id')
            if party_id:
                # Get more details if needed
                party_details = self.get_party_details(party_id)
                if party_details:
                    enriched_parties.append(party_details)
                else:
                    enriched_parties.append(party)
                time.sleep(0.2)  # Rate limiting
                
        enriched_case['enriched_parties'] = enriched_parties
        
        # Add detailed decision information
        enriched_decisions = []
        for decision in decisions_data:
            decision_id = decision.get('id')
            if decision_id:
                # Get decision details with individuals
                decision_details = self.get_decision_details(decision_id, include="individuals")
                
                # Get individuals specific to this decision
                individuals = self.get_decision_individuals(decision_id)
                if decision_details:
                    decision_details['individuals'] = individuals
                    enriched_decisions.append(decision_details)
                else:
                    decision['individuals'] = individuals
                    enriched_decisions.append(decision)
                time.sleep(0.2)  # Rate limiting
                
        enriched_case['enriched_decisions'] = enriched_decisions
        
        return enriched_case

    def search_cases(self, query: str, fields: Optional[str] = None, include: Optional[str] = None, page: int = 1, count: int = 10) -> List[Dict[str, Any]]:
        """
        Search for cases with optional field filtering and inclusion of related resources
        """
        endpoint = f"{self.base_url}/cases"
        self.logger.info(f"Searching cases with query: {query}, fields: {fields}, include: {include}")
        
        params = {
            "page": page,
            "count": count,
            "search": query
        }
        
        if fields:
            params["fields"] = fields
            
        if include:
            params["include"] = include
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=15
            )
            
            data = self._handle_response(response, endpoint)
            cases = data.get('data', [])
            
            # Process included items if present
            included_items = data.get('included', [])
            included_by_type_id = {}
            
            for item in included_items:
                item_type = item.get('type')
                item_id = item.get('id')
                if item_type and item_id:
                    key = f"{item_type}_{item_id}"
                    included_by_type_id[key] = item
            
            # Enrich each case with detailed information
            enriched_cases = []
            for case in cases:
                # For simplicity, we'll get full details directly from API
                case_id = case.get('id')
                if case_id:
                    enriched_case = self.enrich_case_data(case)
                    enriched_cases.append(enriched_case)
                    time.sleep(0.5)  # Rate limiting
                else:
                    enriched_cases.append(case)
            
            return enriched_cases
                
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []

    def get_case_details(self, case_id: str, include: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific case with optional related resources
        """
        endpoint = f"{self.base_url}/cases/{case_id}"
        params = {}
        
        if include:
            params["include"] = include
            
        self.logger.info(f"Getting case details for ID: {case_id} with include: {include}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', {})
        except Exception as e:
            self.logger.error(f"Error getting case details: {str(e)}")
            return {}

    def get_case_decisions(self, case_id: str, page: int = 1, count: int = 10, include: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get decisions associated with a specific case
        """
        endpoint = f"{self.base_url}/cases/{case_id}/decisions"
        params = {
            "page": page,
            "count": count
        }
        
        if include:
            params["include"] = include
            
        self.logger.info(f"Getting decisions for case ID: {case_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"Error getting case decisions: {str(e)}")
            return []

    def get_case_parties(self, case_id: str, page: int = 1, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get parties associated with a specific case
        """
        endpoint = f"{self.base_url}/cases/{case_id}/parties"
        params = {
            "page": page,
            "count": count
        }
        
        self.logger.info(f"Getting parties for case ID: {case_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"Error getting case parties: {str(e)}")
            return []

    def search_decisions(self, query: str, fields: Optional[str] = None, include: Optional[str] = None, page: int = 1, count: int = 10) -> List[Dict[str, Any]]:
        """
        Search for decisions with optional field filtering and inclusion of related resources
        """
        endpoint = f"{self.base_url}/decisions"
        params = {
            "page": page,
            "count": count,
            "search": query
        }
        
        if fields:
            params["fields"] = fields
            
        if include:
            params["include"] = include
        
        self.logger.info(f"Searching decisions with query: {query}, fields: {fields}, include: {include}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=15
            )
            data = self._handle_response(response, endpoint)
            return data.get('data', [])
        except Exception as e:
            self.logger.error(f"Error searching decisions: {str(e)}")
            return []

    def get_arbitrator_challenges(self, arbitrator_id: str) -> List[Dict[str, Any]]:
        """
        Get all challenges related to a specific arbitrator
        """
        endpoint = f"{self.base_url}/arbitrators/{arbitrator_id}/challenges"
        self.logger.info(f"Getting challenges for arbitrator ID: {arbitrator_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            result = self._handle_response(response, endpoint)
            return result.get('data', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting arbitrator challenges: {str(e)}")
            return []

    def get_related_cases(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Get cases related to a specific case
        """
        endpoint = f"{self.base_url}/cases/{case_id}/related"
        self.logger.info(f"Getting related cases for ID: {case_id}")
        
        try:
            response = requests.get(
                endpoint,
                headers=self.headers,
                timeout=10
            )
            result = self._handle_response(response, endpoint)
            return result.get('data', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error getting related cases: {str(e)}")
            return [] 