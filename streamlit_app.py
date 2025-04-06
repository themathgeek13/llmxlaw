import streamlit as st

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Arbitrator Challenge Analysis",
    page_icon="⚖️",
    layout="wide"
)

import os
from dotenv import load_dotenv
from google import genai
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import json
import base64
from io import BytesIO
import logging
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock
import threading
import re

from jusmundi_api import JusMundiAPI
from knowledge_graph import KnowledgeGraph
from rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize thread-safe data structures
class ThreadSafeCounter:
    def __init__(self):
        self._value = 0
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    @property
    def value(self):
        with self._lock:
            return self._value

class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = Lock()
    
    def update(self, key, value):
        with self._lock:
            self._dict[key] = value
    
    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)
    
    def items(self):
        with self._lock:
            return list(self._dict.items())

# Initialize session state for thread-safe data
if 'processed_cases' not in st.session_state:
    st.session_state.processed_cases = ThreadSafeDict()
if 'party_info' not in st.session_state:
    st.session_state.party_info = ThreadSafeDict()
if 'decision_info' not in st.session_state:
    st.session_state.decision_info = ThreadSafeDict()
if 'entity_analysis' not in st.session_state:
    st.session_state.entity_analysis = ThreadSafeDict()
if 'cases_counter' not in st.session_state:
    st.session_state.cases_counter = ThreadSafeCounter()

# Initialize components
@st.cache_resource(ttl=0)  # Set TTL to 0 to prevent caching
def init_components():
    api_key = os.getenv('JUSMUNDI_API_KEY')
    if not api_key:
        st.error("JUSMUNDI_API_KEY not found in environment variables")
        st.stop()
    
    # Initialize API with correct configuration
    api = JusMundiAPI(api_key=api_key)
    
    # Verify API configuration
    if api.base_url != "https://api.jusmundi.com/stanford" or \
       'X-API-Key' not in api.headers or \
       api.headers.get('X-API-Key') != api_key:
        # Force reinitialization if configuration is wrong
        st.cache_resource.clear()
        api = JusMundiAPI(api_key=api_key)
    
    return (
        api,
        KnowledgeGraph(),
        RAGSystem(KnowledgeGraph())
    )

# Add a button to clear cache
if st.sidebar.button("Clear API Cache"):
    st.cache_resource.clear()
    st.rerun()

jusmundi_api, knowledge_graph, rag_system = init_components()

# Function to generate graph visualization
def generate_network_graph(graph_data):
    """Generate an interactive network visualization"""
    net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Configure physics - adjusted for more entities
    net.force_atlas_2based(gravity=-80, central_gravity=0.01, spring_length=150, spring_strength=0.05)
    
    # Add nodes with different colors based on more detailed types
    colors = {
        # Main entity types
        'case': '#4CAF50',        # Green
        'party': '#2196F3',       # Blue
        'decision': '#FFC107',    # Yellow
        'arbitrator': '#FF9800',  # Orange
        'challenge': '#E91E63',   # Pink
        'event': '#673AB7',       # Deep Purple
        
        # More specific entity types
        'company': '#03A9F4',     # Light Blue
        'individual': '#9C27B0',  # Purple
        'country': '#8BC34A',     # Light Green
        'government': '#009688',  # Teal
        'law_firm': '#3F51B5',    # Indigo
        'treaty': '#CDDC39',      # Lime
        'issue': '#9C27B0',       # Purple
        'monetary_amount': '#F44336',  # Red
        'legal_provision': '#FF5722',  # Deep Orange
        'date': '#607D8B',        # Blue Grey
        'document': '#795548',    # Brown
        'organization': '#00BCD4', # Cyan
        'place': '#FFEB3B'        # Yellow
    }
    
    # Get counts of each node type for sizing
    node_type_counts = {}
    for node in graph_data.nodes(data=True):
        node_type = node[1].get('type', 'unknown')
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    
    # Scale node size based on node type frequency (rarer types are larger)
    def get_node_size(node_type):
        count = node_type_counts.get(node_type, 1)
        total_nodes = len(graph_data.nodes)
        # Base size on inverse frequency (rarer = bigger)
        size_multiplier = 25 * (1 - (count / total_nodes))
        return max(15, min(50, 20 + size_multiplier))  # Between 15 and 50
    
    # Add nodes
    for node in graph_data.nodes(data=True):
        node_id = node[0]
        node_data = node[1]
        node_type = node_data.get('type', 'unknown')
        
        # Create label and title based on node type
        if node_type == 'case':
            label = node_data.get('name', 'Unknown Case')[:30] + '...' if len(node_data.get('name', 'Unknown Case')) > 30 else node_data.get('name', 'Unknown Case')
            title = f"""Case: {node_data.get('name', '')}
Organization: {node_data.get('organization', '')}
Outcome: {node_data.get('outcome', '')}
Date: {node_data.get('date', '')}
Type: {node_data.get('dispute_type', '')}"""
        elif node_type in ['party', 'company', 'organization', 'government']:
            label = node_data.get('name', f'Unknown {node_type}')[:20] + '...' if len(node_data.get('name', f'Unknown {node_type}')) > 20 else node_data.get('name', f'Unknown {node_type}')
            title = f"""{node_type.capitalize()}: {node_data.get('name', '')}
Role: {node_data.get('role', '')}
Type: {node_data.get('party_type', '')}
Nationality: {node_data.get('nationality', '')}"""
        elif node_type == 'decision':
            label = node_data.get('name', 'Unknown Decision')[:20] + '...' if len(node_data.get('name', 'Unknown Decision')) > 20 else node_data.get('name', 'Unknown Decision')
            title = f"""Decision: {node_data.get('name', '')}
Type: {node_data.get('decision_type', '')}
Outcome: {node_data.get('outcome', '')}
Date: {node_data.get('date', '')}"""
        elif node_type in ['arbitrator', 'individual']:
            label = node_data.get('name', f'Unknown {node_type}')[:20] + '...' if len(node_data.get('name', f'Unknown {node_type}')) > 20 else node_data.get('name', f'Unknown {node_type}')
            title = f"""{node_type.capitalize()}: {node_data.get('name', '')}
Role: {node_data.get('role', '')}
Nationality: {node_data.get('nationality', '')}
Firm: {node_data.get('firm', '')}"""
        elif node_type == 'challenge':
            label = "Challenge"
            title = f"""Challenge: {node_data.get('name', '')}
Grounds: {node_data.get('grounds', '')}
Outcome: {node_data.get('outcome', '')}
Date: {node_data.get('date', '')}"""
        elif node_type == 'event':
            label = node_data.get('name', 'Event')[:15] + '...' if len(node_data.get('name', 'Event')) > 15 else node_data.get('name', 'Event')
            title = f"""Event: {node_data.get('name', '')}
Date: {node_data.get('date', '')}
Description: {node_data.get('description', '')}"""
        elif node_type == 'country':
            label = node_data.get('name', 'Unknown Country')
            title = f"""Country: {node_data.get('name', '')}"""
        elif node_type == 'treaty':
            label = node_data.get('name', 'Treaty')[:15] + '...' if len(node_data.get('name', 'Treaty')) > 15 else node_data.get('name', 'Treaty')
            title = f"""Treaty: {node_data.get('name', '')}
Type: {node_data.get('treaty_type', '')}
Date: {node_data.get('date', '')}"""
        elif node_type == 'legal_provision':
            label = node_data.get('name', 'Provision')[:10] + '...' if len(node_data.get('name', 'Provision')) > 10 else node_data.get('name', 'Provision')
            title = f"""Provision: {node_data.get('name', '')}
Source: {node_data.get('source', '')}"""
        elif node_type == 'monetary_amount':
            label = f"{node_data.get('amount', '')} {node_data.get('currency', '')}"
            title = f"""Amount: {node_data.get('amount', '')} {node_data.get('currency', '')}
Type: {node_data.get('type', '')}
Description: {node_data.get('description', '')}"""
        else:
            label = node_data.get('name', str(node_id))[:15] + '...' if len(node_data.get('name', str(node_id))) > 15 else node_data.get('name', str(node_id))
            title = f"{node_type.capitalize()}: {node_data.get('name', '')}"
            # Add all attributes to the title
            for k, v in node_data.items():
                if k not in ['name', 'type', 'id'] and v:
                    title += f"\n{k}: {v}"
        
        # Get appropriate color and size
        color = colors.get(node_type, '#666666')
        size = get_node_size(node_type)
        
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=color,
            size=size
        )
    
    # Add edges with tooltips and improved styling
    for edge in graph_data.edges(data=True):
        source, target, edge_data = edge
        
        # Get relationship type for styling
        rel_type = edge_data.get('type', 'related_to')
        
        # Build detailed tooltip
        title = f"Type: {rel_type}"
        for k, v in edge_data.items():
            if k != 'type' and v:
                title += f"\n{k}: {v}"
        
        # Set edge color and arrow properties based on relationship type
        if 'appointed' in rel_type:
            edge_color = 'blue'
        elif 'challenge' in rel_type:
            edge_color = 'red'
        elif 'filed' in rel_type:
            edge_color = 'orange'
        elif 'represents' in rel_type:
            edge_color = 'green'
        elif 'owns' in rel_type or 'subsidiary' in rel_type:
            edge_color = 'purple'
        elif 'nationality' in rel_type:
            edge_color = 'teal'
        else:
            edge_color = '#888888'  # Default gray
            
        net.add_edge(
            source, 
            target, 
            title=title,
            color=edge_color,
            width=1.5
        )
    
    # Generate temporary file with community detection enabled for better visualization
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp:
        # Add configuration for better visualization of large graphs
        net.set_options("""
        var options = {
            "nodes": {
                "borderWidth": 2,
                "borderWidthSelected": 4,
                "font": {
                    "size": 12,
                    "face": "Tahoma"
                }
            },
            "edges": {
                "smooth": {
                    "type": "continuous",
                    "forceDirection": "none"
                },
                "arrows": {
                    "to": {
                        "enabled": true,
                        "scaleFactor": 0.5
                    }
                }
            },
            "physics": {
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                },
                "maxVelocity": 50,
                "minVelocity": 0.75,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000,
                    "updateInterval": 100
                }
            }
        }
        """)
        
        net.save_graph(tmp.name)
        return tmp.name

# Function to plot graph using Matplotlib
def plot_matplotlib_graph():
    G = knowledge_graph.graph
    if len(G.nodes) == 0:
        return None
    
    # Create node color list
    node_colors = []
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'unknown')
        if node_type == 'case':
            node_colors.append('#4CAF50')  # Green
        elif node_type == 'arbitrator':
            node_colors.append('#FF9800')  # Orange
        elif node_type == 'challenge':
            node_colors.append('#E91E63')  # Pink
        elif node_type == 'party':
            node_colors.append('#2196F3')  # Blue
        elif node_type == 'decision':
            node_colors.append('#FFC107')  # Yellow
        else:
            node_colors.append('#CCCCCC')  # Gray
    
    # Create plot
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw_networkx(
        G, 
        pos=pos, 
        with_labels=True, 
        node_color=node_colors,
        node_size=500, 
        font_size=8,
        edge_color='#CCCCCC'
    )
    
    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # Encode plot to base64 for display
    return base64.b64encode(buf.read()).decode()

def display_case_info(case):
    """Display basic case information"""
    attributes = case.get('attributes', {})
    case_id = case.get('id', 'Unknown')
    
    st.markdown(f"### {attributes.get('title', 'Untitled Case')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Case ID:** {case_id}")
        st.write(f"**Organization:** {attributes.get('organization', 'N/A')}")
        st.write(f"**Outcome:** {attributes.get('outcome', 'N/A')}")
        st.write(f"**Commencement Date:** {attributes.get('commencement_date', 'N/A')}")
    
    with col2:
        st.write(f"**Treaty:** {attributes.get('treaty', 'N/A')}")
        st.write(f"**Rules:** {attributes.get('rules', 'N/A')}")
        st.write(f"**Industry:** {attributes.get('industry', 'N/A')}")
        st.write(f"**Subject Matter:** {attributes.get('subject_matter', 'N/A')}")
    
    # Display monetary information if available
    monetary_info = attributes.get('monetary_info', {})
    if monetary_info:
        st.write("**Monetary Information:**")
        st.write(f"- Amount: {monetary_info.get('amount', 'N/A')} {monetary_info.get('currency', '')}")
        st.write(f"- Type: {monetary_info.get('type', 'N/A')}")

def fetch_party_details(case_id, party):
    """Fetch party details in a separate thread"""
    try:
        party_id = party.get('id')
        if party_id:
            party_details = jusmundi_api.get_party_details(party_id)
            if party_details:
                st.session_state.party_info.update(f"{case_id}_{party_id}", party_details)
                return party_details
    except Exception as e:
        logger.error(f"Error fetching party details: {str(e)}")
    return None

def fetch_decision_details(case_id, decision):
    """Fetch decision details in a separate thread"""
    try:
        decision_id = decision.get('id')
        if decision_id:
            decision_details = jusmundi_api.get_decision_details(decision_id)
            individuals = jusmundi_api.get_decision_individuals(decision_id)
            if decision_details:
                decision_details['individuals'] = individuals
                st.session_state.decision_info.update(f"{case_id}_{decision_id}", decision_details)
                return decision_details
    except Exception as e:
        logger.error(f"Error fetching decision details: {str(e)}")
    return None

def extract_entities_relations(text, client):
    """Extract detailed entities and relations using Gemini API with a two-step approach"""
    
    # Gather all available case information into a single text
    case_attrs = text.get('attributes', {})
    case_text = f"""
    Case Title: {case_attrs.get('title', '')}
    Organization: {case_attrs.get('organization', '')}
    Outcome: {case_attrs.get('outcome', '')}
    Date: {case_attrs.get('commencement_date', '')}
    Treaty: {case_attrs.get('treaty', '')}
    Rules: {case_attrs.get('rules', '')}
    Industry: {case_attrs.get('industry', '')}
    Subject Matter: {case_attrs.get('subject_matter', '')}
    
    Parties:
    {text.get('parties_text', '')}
    
    Decisions:
    {text.get('decisions_text', '')}
    
    Individuals:
    {text.get('individuals_text', '')}
    """

    # STEP 1: First search with Google grounding to get detailed information
    try:
        logger.info(f"Starting entity extraction for case: {case_attrs.get('title', 'Unknown')}")
        
        # Configure the search tool for grounding
        google_search_tool = genai.types.Tool(
            google_search=genai.types.GoogleSearch()
        )

        # First prompt - to gather deep context about the case
        search_prompt = f"""
        I need extremely detailed information about this arbitration case:
        {case_attrs.get('title', '')} 
        
        Find ALL specific details about:
        1. Every party involved (companies, governments, individuals, subsidiaries)
        2. Every arbitrator and counsel member, including their nationalities and firms
        3. All challenges, including precise grounds, outcomes, and dates
        4. All key dates and events in chronological order (filings, hearings, decisions)
        5. All monetary claims and awards with exact amounts
        6. All cited legal provisions, treaties, and rules
        7. All relationships between entities (ownership, representation, appointment)
        8. All jurisdictions, locations, and countries mentioned
        
        Provide the most granular and specific details possible about this case.
        
        Context: {case_text}
        """
        
        # Make first API call with Google Search grounding
        logger.info("Making first API call to search for case details...")
        try:
            search_response = client.models.generate_content(
                model="gemini-2.0-flash",  # Switch back to flash model
                contents=search_prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    temperature=0.1,
                )
            )
            
            if not search_response.candidates:
                logger.error("Empty response from Gemini search API")
                return {"entities": [], "relations": [], "events": [], "challenges": []}
                
            # Get the enhanced context from search
            enhanced_context = search_response.candidates[0].content.parts[0].text
            
            logger.info(f"Enhanced context obtained: {enhanced_context[:200]}...")
        except Exception as e:
            logger.error(f"Error in first API call: {str(e)}")
            # Continue with just the original case text
            enhanced_context = "No additional details found."
        
        # Combine original case details with enhanced context
        combined_context = f"""
        ORIGINAL CASE DETAILS:
        {case_text}
        
        ENHANCED DETAILS:
        {enhanced_context}
        """
        
        # STEP 2: Use the combined context to extract structured entities and relations
        extraction_prompt = """
        Extract EVERY POSSIBLE entity and relationship from this arbitration case information. 
        
        Be EXTREMELY detailed and granular. The goal is to create a comprehensive network graph, so extract as many entities, relationships, and attributes as possible.
        
        Extract ALL of the following:
        
        1. ENTITIES - Extract every entity mentioned, including but not limited to:
           - Parties (claimants, respondents, third parties, intervenors)  
           - Individuals (arbitrators, counsel, witnesses, experts, judges)
           - Organizations (firms, companies, tribunals, courts, institutions)
           - Countries & governments (states, agencies, ministries)
           - Places (cities, jurisdictions, venues, tribunals)
           - Legal instruments (treaties, contracts, laws, rules, provisions)
           - Dates (filings, hearings, appointments, decisions, awards)
           - Monetary amounts (claims, awards, costs, damages)
           - Documents (awards, decisions, submissions, evidence)
           - Issues (jurisdictional, procedural, substantive, factual, legal)
        
        2. RELATIONSHIPS - Extract every possible relationship between entities:
           - Party-Party relationships (ownership, subsidiaries, joint ventures)
           - Party-Individual (representation, employment, appointment, challenge)
           - Individual-Individual (panel relationships, conflicts)
           - Entity-Country (nationality, jurisdiction, incorporation)
           - Entity-Date (event timing, appointment dates, decision dates)
           - Entity-Document (authorship, issuance, reference)
           - Document-Legal Instrument (citation, application, interpretation)
           - Any other relationships you can identify
        
        3. EVENTS - Extract all chronological events:
           - Process events (filings, hearings, deliberations)
           - Decision events (awards, rulings, dissents)
           - Challenge events (filings, responses, decisions)
           - Procedural events (bifurcations, stays, consolidations)
           
        4. CHALLENGES - Extract all arbitrator challenges:
           - Who filed the challenge and who was challenged
           - Specific grounds with detailed rationale
           - Outcomes with reasons given
           - Dates of challenge and decision
           - Authorities making the decision
        
        IMPORTANT: You MUST respond in valid JSON format with this structure:
        {
            "entities": [
                {"id": "unique_id", "type": "precise_type", "name": "exact_name", "attributes": {"role": "", "nationality": "", "position": "", "date": "", "amount": "", "currency": "", "any_other_relevant_attributes": ""}}
            ],
            "relations": [
                {"source": "entity1_id", "target": "entity2_id", "type": "precise_relationship_type", "attributes": {"date": "", "details": "", "any_other_relevant_attributes": ""}}
            ],
            "events": [
                {"id": "event_id", "date": "exact_date", "description": "detailed_description", "entities_involved": [{"id": "entity_id", "role": "role_in_event"}]}
            ],
            "challenges": [
                {"id": "challenge_id", "challenger": "entity_id", "challenged": "entity_id", "grounds": "detailed_grounds", "outcome": "detailed_outcome", "date": "exact_date", "deciding_authority": "authority_id"}
            ]
        }
        
        Your response MUST be valid JSON. Ensure ALL keys and values are properly quoted. Do not include any text outside the JSON.
        
        If you're not sure about an entity or relationship, create a reasonable estimate based on the available information.
        At minimum, include the case itself, all named parties, arbitrators, and decisions as entities.
        """
        
        # Make second API call with the enhanced context
        logger.info("Making second API call to extract structured entities...")
        try:
            extraction_response = client.models.generate_content(
                model="gemini-2.0-flash", # Use flash model for consistent results
                contents=[
                    extraction_prompt,
                    combined_context
                ],
                config=genai.types.GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more consistent output
                    max_output_tokens=8192, # Increase token limit for more detailed output
                    top_p=0.95,
                    top_k=40
                )
            )
            
            if not extraction_response.text:
                logger.error("Empty response from Gemini extraction API")
                # Return minimal structure to avoid breaking code
                return create_fallback_analysis(text)
                
            response_text = extraction_response.text.strip()
            logger.info(f"Received extraction response of length: {len(response_text)}")
            
            # Clean up the response text to ensure it's valid JSON
            response_text = re.sub(r'^```.*?\n', '', response_text)  # Remove opening ```
            response_text = re.sub(r'\n```$', '', response_text)     # Remove closing ```
            response_text = response_text.strip()
            
            # Find JSON blocks in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                logger.info("Found JSON block in response")
            
            try:
                parsed_json = json.loads(response_text)
                if not isinstance(parsed_json, dict):
                    logger.error(f"Parsed JSON is not a dictionary: {type(parsed_json)}")
                    return create_fallback_analysis(text)
                
                # Ensure all required keys exist
                required_keys = ["entities", "relations", "events", "challenges"]
                for key in required_keys:
                    if key not in parsed_json:
                        parsed_json[key] = []
                    elif not isinstance(parsed_json[key], list):
                        parsed_json[key] = []
                
                # Convert challenges to entities and relations for better graph representation
                challenges = parsed_json.get('challenges', [])
                for idx, challenge in enumerate(challenges):
                    challenge_id = challenge.get('id', f"challenge_{idx}")
                    # Add challenge as an entity
                    parsed_json['entities'].append({
                        "id": challenge_id,
                        "type": "challenge",
                        "name": f"Challenge: {challenge.get('grounds', 'Unknown grounds')}",
                        "attributes": {
                            "grounds": challenge.get('grounds', ''),
                            "outcome": challenge.get('outcome', ''),
                            "date": challenge.get('date', ''),
                            "deciding_authority": challenge.get('deciding_authority', '')
                        }
                    })
                    
                    # Add challenger relationship
                    if challenge.get('challenger'):
                        parsed_json['relations'].append({
                            "source": challenge.get('challenger'),
                            "target": challenge_id,
                            "type": "filed_challenge",
                            "attributes": {"date": challenge.get('date', '')}
                        })
                    
                    # Add challenged relationship
                    if challenge.get('challenged'):
                        parsed_json['relations'].append({
                            "source": challenge_id,
                            "target": challenge.get('challenged'),
                            "type": "challenged",
                            "attributes": {"date": challenge.get('date', '')}
                        })
                
                # Convert events to entities and relations for better graph representation
                events = parsed_json.get('events', [])
                for idx, event in enumerate(events):
                    event_id = event.get('id', f"event_{idx}")
                    # Add event as an entity
                    parsed_json['entities'].append({
                        "id": event_id,
                        "type": "event",
                        "name": event.get('description', 'Unknown event'),
                        "attributes": {
                            "date": event.get('date', ''),
                            "description": event.get('description', '')
                        }
                    })
                    
                    # Add relationships for entities involved in the event
                    for entity in event.get('entities_involved', []):
                        if entity.get('id'):
                            parsed_json['relations'].append({
                                "source": entity.get('id'),
                                "target": event_id,
                                "type": entity.get('role', 'involved_in'),
                                "attributes": {"date": event.get('date', '')}
                            })
                
                # Log success with entity counts
                entity_count = len(parsed_json.get('entities', []))
                relation_count = len(parsed_json.get('relations', []))
                event_count = len(parsed_json.get('events', []))
                challenge_count = len(parsed_json.get('challenges', []))
                
                logger.info(f"Successfully extracted: {entity_count} entities, {relation_count} relations, {event_count} events, {challenge_count} challenges")
                
                # Add the case itself as an entity if not already included
                case_exists = False
                case_id = f"case_{text.get('id', 'unknown')}"
                
                for entity in parsed_json.get('entities', []):
                    if entity.get('id') == case_id:
                        case_exists = True
                        break
                        
                if not case_exists:
                    parsed_json['entities'].append({
                        "id": case_id,
                        "type": "case",
                        "name": case_attrs.get('title', 'Unknown Case'),
                        "attributes": {
                            "organization": case_attrs.get('organization', ''),
                            "outcome": case_attrs.get('outcome', ''),
                            "date": case_attrs.get('commencement_date', ''),
                            "treaty": case_attrs.get('treaty', ''),
                            "rules": case_attrs.get('rules', ''),
                            "industry": case_attrs.get('industry', ''),
                            "subject_matter": case_attrs.get('subject_matter', '')
                        }
                    })
                    
                    # Connect all entities to the case if they don't have a relation
                    connected_entities = set()
                    for relation in parsed_json.get('relations', []):
                        connected_entities.add(relation.get('source', ''))
                        connected_entities.add(relation.get('target', ''))
                    
                    for entity in parsed_json.get('entities', []):
                        entity_id = entity.get('id', '')
                        if entity_id and entity_id != case_id and entity_id not in connected_entities:
                            parsed_json['relations'].append({
                                "source": case_id,
                                "target": entity_id,
                                "type": "contains",
                                "attributes": {}
                            })
                
                return parsed_json
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON from response: {str(e)}\nResponse text: {response_text[:200]}...")
                return create_fallback_analysis(text)
                
        except Exception as e:
            logger.error(f"Error in second API call: {str(e)}")
            return create_fallback_analysis(text)

    except Exception as e:
        logger.error(f"Error in two-step entity extraction: {str(e)}")
        return create_fallback_analysis(text)

def create_fallback_analysis(text):
    """Create a basic analysis when API extraction fails"""
    logger.info("Creating fallback analysis with minimal entities")
    
    case_attrs = text.get('attributes', {})
    case_id = f"case_{text.get('id', 'unknown')}"
    
    # Create a minimal analysis with the case and directly available information
    analysis = {
        "entities": [
            {
                "id": case_id,
                "type": "case",
                "name": case_attrs.get('title', 'Unknown Case'),
                "attributes": {
                    "organization": case_attrs.get('organization', ''),
                    "outcome": case_attrs.get('outcome', ''),
                    "date": case_attrs.get('commencement_date', ''),
                    "treaty": case_attrs.get('treaty', ''),
                    "rules": case_attrs.get('rules', ''),
                    "industry": case_attrs.get('industry', ''),
                    "subject_matter": case_attrs.get('subject_matter', '')
                }
            }
        ],
        "relations": [],
        "events": [],
        "challenges": []
    }
    
    # Add organizations as entities
    if case_attrs.get('organization'):
        org_id = f"organization_{case_attrs.get('organization', '').replace(' ', '_')}"
        analysis["entities"].append({
            "id": org_id,
            "type": "organization",
            "name": case_attrs.get('organization', ''),
            "attributes": {}
        })
        # Relate to case
        analysis["relations"].append({
            "source": org_id,
            "target": case_id,
            "type": "administers",
            "attributes": {}
        })
    
    # Try to add basic party info if available
    parties_text = text.get('parties_text', '')
    if parties_text:
        party_lines = parties_text.strip().split("\n")
        for i, line in enumerate(party_lines):
            if line.startswith("-"):
                party_info = line[1:].strip()
                # Extract name
                name_match = re.search(r'(.*?)\s*\(', party_info)
                if name_match:
                    party_name = name_match.group(1).strip()
                    party_id = f"party_{i}_{party_name.replace(' ', '_')}"
                    
                    # Extract role, type, nationality if possible
                    role = "Unknown"
                    party_type = "Unknown"
                    nationality = "Unknown"
                    
                    role_match = re.search(r'role: (.*?)(?:,|\))', party_info, re.IGNORECASE)
                    if role_match:
                        role = role_match.group(1).strip()
                    
                    type_match = re.search(r'type: (.*?)(?:,|\))', party_info, re.IGNORECASE)
                    if type_match:
                        party_type = type_match.group(1).strip()
                    
                    nationality_match = re.search(r'nationality: (.*?)(?:,|\))', party_info, re.IGNORECASE)
                    if nationality_match:
                        nationality = nationality_match.group(1).strip()
                    
                    analysis["entities"].append({
                        "id": party_id,
                        "type": "party",
                        "name": party_name,
                        "attributes": {
                            "role": role,
                            "party_type": party_type,
                            "nationality": nationality
                        }
                    })
                    
                    # Add relation to case
                    analysis["relations"].append({
                        "source": party_id,
                        "target": case_id,
                        "type": "involved_in",
                        "attributes": {}
                    })
    
    # Try to add basic decision info if available
    individuals_text = text.get('individuals_text', '')
    if individuals_text:
        individual_lines = individuals_text.strip().split("\n")
        for i, line in enumerate(individual_lines):
            if line.startswith("-"):
                individual_info = line[1:].strip()
                # Extract name
                name_match = re.search(r'(.*?)\s*\(', individual_info)
                if name_match:
                    individual_name = name_match.group(1).strip()
                    
                    # Determine if arbitrator
                    is_arbitrator = False
                    if "arbitrator" in individual_info.lower() or "president" in individual_info.lower():
                        is_arbitrator = True
                        
                    ind_id = f"{'arbitrator' if is_arbitrator else 'individual'}_{i}_{individual_name.replace(' ', '_')}"
                    
                    # Extract role
                    role = "Unknown"
                    role_match = re.search(r'\((.*?)\)', individual_info)
                    if role_match:
                        role = role_match.group(1).strip()
                    
                    # Extract firm if present
                    firm = None
                    firm_match = re.search(r'firm: (.*)', individual_info, re.IGNORECASE)
                    if firm_match:
                        firm = firm_match.group(1).strip()
                    
                    analysis["entities"].append({
                        "id": ind_id,
                        "type": "arbitrator" if is_arbitrator else "individual",
                        "name": individual_name,
                        "attributes": {
                            "role": role,
                            "firm": firm if firm else ""
                        }
                    })
                    
                    # Add relation to case
                    analysis["relations"].append({
                        "source": ind_id,
                        "target": case_id,
                        "type": "involved_in",
                        "attributes": {}
                    })
                    
                    # If firm exists, add it and create relation
                    if firm:
                        firm_id = f"firm_{firm.replace(' ', '_')}"
                        
                        # Check if firm already exists
                        firm_exists = False
                        for entity in analysis["entities"]:
                            if entity.get("id") == firm_id:
                                firm_exists = True
                                break
                                
                        if not firm_exists:
                            analysis["entities"].append({
                                "id": firm_id,
                                "type": "law_firm",
                                "name": firm,
                                "attributes": {}
                            })
                        
                        # Add relation between individual and firm
                        analysis["relations"].append({
                            "source": ind_id,
                            "target": firm_id,
                            "type": "member_of",
                            "attributes": {}
                        })
    
    return analysis

def update_knowledge_graph_with_analysis(graph, analysis):
    """Update knowledge graph with extracted entities and relations"""
    if not analysis:
        return
    
    # Use a lock to ensure thread safety when updating the graph
    graph_lock = threading.Lock()
    
    # Log before adding to help debug
    logger.info(f"Adding {len(analysis.get('entities', []))} entities and {len(analysis.get('relations', []))} relations to knowledge graph")
    
    with graph_lock:
        # Add entities with their unique IDs
        for entity in analysis.get('entities', []):
            entity_id = entity.get('id', f"{entity.get('type', 'unknown')}_{entity.get('name', 'unnamed')}")
            
            # Create a clean copy of attributes for the node
            node_attrs = {
                'type': entity.get('type', 'unknown'),
                'name': entity.get('name', 'Unnamed Entity')
            }
            
            # Add all additional attributes
            if entity.get('attributes'):
                for key, value in entity.get('attributes', {}).items():
                    if value:  # Only add non-empty attributes
                        node_attrs[key] = value
            
            # Add node to graph
            graph.add_node(entity_id, **node_attrs)
            logger.info(f"Added node: {entity_id} of type {node_attrs['type']}")
        
        # Add relations using the IDs
        for relation in analysis.get('relations', []):
            source_id = relation.get('source')
            target_id = relation.get('target')
            
            if not source_id or not target_id:
                continue
                
            # Create edge attributes
            edge_attrs = {
                'type': relation.get('type', 'related_to')
            }
            
            # Add all additional attributes
            if relation.get('attributes'):
                for key, value in relation.get('attributes', {}).items():
                    if value:  # Only add non-empty attributes
                        edge_attrs[key] = value
            
            # Add edge to graph only if both nodes exist
            if source_id in graph.nodes and target_id in graph.nodes:
                graph.add_edge(source_id, target_id, **edge_attrs)
                logger.info(f"Added edge: {source_id} -> {target_id} of type {edge_attrs['type']}")
            else:
                logger.warning(f"Cannot add edge: {source_id} -> {target_id}, one or both nodes missing")

def display_decision_info(decision, is_challenge):
    """Display information about a decision"""
    decision_attrs = decision.get('attributes', {})
    decision_id = decision.get('id', 'Unknown')
    
    st.markdown(f"##### {decision_attrs.get('title', f'Decision {decision_id}')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**ID:** {decision_id}")
        st.write(f"**Type:** {decision_attrs.get('type', 'N/A')}")
        st.write(f"**Date:** {decision_attrs.get('date', 'N/A')}")
        st.write(f"**Status:** {decision_attrs.get('status', 'N/A')}")
    
    with col2:
        if is_challenge:
            description = decision_attrs.get('description', '')
            
            # Try to identify challenge grounds
            grounds = "Not specified"
            if "grounds" in description.lower():
                parts = description.split("grounds")
                if len(parts) > 1:
                    grounds = parts[1].split(".")[0].strip()
            
            # Try to identify challenge outcome
            outcome = "Not specified"
            outcome_indicators = ["dismissed", "rejected", "accepted", "upheld", "granted"]
            for indicator in outcome_indicators:
                if indicator in description.lower():
                    idx = description.lower().find(indicator)
                    outcome = description[idx:idx+50].split(".")[0].strip()
                    break
            
            st.write("**Challenge Information:**")
            st.write(f"- Grounds: {grounds}")
            st.write(f"- Outcome: {outcome}")
        
        # Show decision description
        description = decision_attrs.get('description', 'No description available')
        st.write("**Description:**")
        st.write(description)
    
    # Display individuals - create a new section outside the columns
    individuals = decision.get('individuals', [])
    if individuals:
        st.markdown("**Individuals**")
        
        # Separate arbitrators from other individuals
        arbitrators = []
        others = []
        
        for individual in individuals:
            individual_attrs = individual.get('attributes', {})
            individual_role = individual_attrs.get('role', '').lower()
            
            if 'arbitrator' in individual_role or 'president' in individual_role or 'chairman' in individual_role:
                arbitrators.append(individual)
            else:
                others.append(individual)
        
        # Display arbitrators in a flat structure
        if arbitrators:
            st.write("**Arbitrators:**")
            arb_text = ""
            for arb in arbitrators:
                arb_attrs = arb.get('attributes', {})
                arb_text += f"- {arb_attrs.get('name', 'Unknown')}: {arb_attrs.get('role', 'Unknown Role')}\n"
                if 'firm' in arb_attrs:
                    arb_text += f"  Firm: {arb_attrs.get('firm', 'Unknown')}\n"
            st.markdown(arb_text)
        
        # Display other individuals in a flat structure
        if others:
            st.write("**Other Participants:**")
            other_text = ""
            for other in others:
                other_attrs = other.get('attributes', {})
                other_text += f"- {other_attrs.get('name', 'Unknown')}: {other_attrs.get('role', 'Unknown Role')}\n"
            st.markdown(other_text)

def process_case(case, status_placeholder=None):
    """Process a single case and update its status placeholder"""
    case_id = case.get('id')
    if not case_id:
        return
    
    try:
        # Initialize storage for this case in session state if not present
        if 'all_case_details' not in st.session_state:
            st.session_state.all_case_details = {}
        if case_id not in st.session_state.all_case_details:
             st.session_state.all_case_details[case_id] = {
                'case': case,
                'parties': [],
                'decisions': [],
                'analysis': None,
                'party_status': 'pending',
                'decision_status': 'pending',
                'analysis_status': 'pending'
            }
        
        # Use existing entry
        case_details = st.session_state.all_case_details[case_id]

        # Gather all case information for analysis
        case_info = case.copy()

        # --- Fetch Party Information (Store data, update status) ---
        parties = jusmundi_api.get_case_parties(case_id)
        parties_text = ""
        party_fetch_success = False
        if parties:
            fetched_parties = []
            # Basic rate limiting
            delay = 0.1
            for party in parties:
                time.sleep(delay)
                party_details = jusmundi_api.get_party_details(party.get('id'))
                if party_details:
                    fetched_parties.append(party_details)
                    attrs = party_details.get('attributes', {})
                    parties_text += f"- {attrs.get('name', 'N/A')} (...)\n"
                    party_fetch_success = True
            case_details['parties'] = fetched_parties
        case_details['party_status'] = 'success' if party_fetch_success else 'nodata'
        case_info['parties_text'] = parties_text

        # --- Fetch Decision Information (Store data, update status) ---
        decisions = jusmundi_api.get_case_decisions(case_id)
        decisions_text = ""
        individuals_text = ""
        decision_fetch_success = False
        if decisions:
            fetched_decisions = []
            try:
                delay = 0.1
                for decision in decisions:
                    time.sleep(delay)
                    decision_id = decision.get('id')
                    if not decision_id: continue
                    decision_details = jusmundi_api.get_decision_details(decision_id)
                    if not decision_details: continue
                    time.sleep(delay) # Add delay for individual call too
                    individuals = jusmundi_api.get_decision_individuals(decision_id)
                    decision_details['individuals'] = individuals
                    fetched_decisions.append(decision_details)
                    decision_fetch_success = True
                    attrs = decision_details.get('attributes', {})
                    decisions_text += f"- {attrs.get('title', 'N/A')}\n"
                    if individuals:
                         for individual in individuals:
                             ind_attrs = individual.get('attributes', {})
                             individuals_text += f"- {ind_attrs.get('name', 'N/A')} (...)\n"
                case_details['decisions'] = fetched_decisions
                case_details['decision_status'] = 'success' if decision_fetch_success else 'nodata'
            except Exception as e:
                 logger.error(f"Error fetching decisions detail for {case_id}: {str(e)}")
                 case_details['decision_status'] = 'error'
        else:
             case_details['decision_status'] = 'nodata'
        
        case_info['decisions_text'] = decisions_text
        case_info['individuals_text'] = individuals_text

        # --- Analyze Case Text (Store analysis dict, update status) ---
        analysis = None
        analysis_status = 'pending'
        analysis_error_msg = None
        try:
            analysis = extract_entities_relations(case_info, client)
            if isinstance(analysis, dict):
                case_details['analysis'] = analysis
                # Wrap knowledge graph update in a try-except to catch expander errors
                try:
                    update_knowledge_graph_with_analysis(knowledge_graph.graph, analysis)
                    analysis_status = 'success'
                except Exception as kg_error:
                    logger.error(f"Error updating knowledge graph for case {case_id}: {str(kg_error)}")
                    # Still consider analysis successful even if graph update fails
                    analysis_status = 'success'
                    analysis_error_msg = f"Knowledge graph update failed: {str(kg_error)}"
            else:
                analysis_status = 'fail'
                analysis_error_msg = "Analysis did not return expected format."
                logger.error(f"Analysis failed for {case_id}: {analysis_error_msg}")
        except Exception as e:
            logger.error(f"Error in entity analysis for case {case_id}: {str(e)}")
            analysis_status = 'error'
            analysis_error_msg = f"Error during analysis: {str(e)}"

        case_details['analysis_status'] = analysis_status
        case_details['analysis_error_msg'] = analysis_error_msg # Store error message
        
        # Increment processed cases counter with safety check
        if 'cases_counter' in st.session_state:
            st.session_state.cases_counter.increment()
        else:
            logger.error("cases_counter not found in session_state")
        
        # Return processing result information
        return {
            'case_id': case_id,
            'status': 'success'
        }

    except Exception as e:
        logger.error(f"Unhandled error processing case {case_id}: {str(e)}")
        # Ensure details are marked as error in session state
        if case_id in st.session_state.all_case_details:
            st.session_state.all_case_details[case_id]['party_status'] = 'error'
            st.session_state.all_case_details[case_id]['decision_status'] = 'error'
            st.session_state.all_case_details[case_id]['analysis_status'] = 'error'
            st.session_state.all_case_details[case_id]['analysis_error_msg'] = f"Unhandled: {str(e)}"
        
        # Return error information
        return {
            'case_id': case_id,
            'status': 'error',
            'error': str(e)
        }

def analyze_conflicts_iba(cases, party_info, decision_info):
    """Analyze conflicts based on IBA Guidelines traffic light system"""
    conflicts = {
        'red': [],  # Non-waivable conflicts
        'orange': [], # Waivable conflicts if parties agree
        'green': []  # Minor issues to disclose
    }
    
    # Build relationship maps
    arbitrators = {}  # arbitrator -> cases/roles
    law_firms = {}   # firm -> cases/roles
    parties = {}     # party -> cases/roles
    
    for case in cases:
        case_id = case.get('id')
        if not case_id:
            continue
            
        # Get all decisions for this case
        case_decisions = [
            decision for key, decision in decision_info.items()
            if key.startswith(f"{case_id}_")
        ]
        
        # Get all parties for this case
        case_parties = [
            party for key, party in party_info.items()
            if key.startswith(f"{case_id}_")
        ]
        
        # Process arbitrators from decisions
        for decision in case_decisions:
            individuals = decision.get('individuals', [])
            for individual in individuals:
                attrs = individual.get('attributes', {})
                name = attrs.get('name')
                role = attrs.get('role', '').lower()
                firm = attrs.get('firm')
                
                if 'arbitrator' in role or 'president' in role or 'chairman' in role:
                    if name not in arbitrators:
                        arbitrators[name] = {'cases': set(), 'roles': set(), 'firms': set()}
                    arbitrators[name]['cases'].add(case_id)
                    arbitrators[name]['roles'].add(role)
                    if firm:
                        arbitrators[name]['firms'].add(firm)
                        if firm not in law_firms:
                            law_firms[firm] = {'cases': set(), 'roles': set()}
                        law_firms[firm]['cases'].add(case_id)
                        law_firms[firm]['roles'].add('arbitrator firm')
        
        # Process parties
        for party in case_parties:
            attrs = party.get('attributes', {})
            name = attrs.get('name')
            role = attrs.get('role')
            if name:
                if name not in parties:
                    parties[name] = {'cases': set(), 'roles': set()}
                parties[name]['cases'].add(case_id)
                parties[name]['roles'].add(role)
    
    # Analyze conflicts based on IBA Guidelines
    
    # Red List (Non-waivable)
    for arb_name, arb_data in arbitrators.items():
        # Check identity with party
        if arb_name in parties:
            conflicts['red'].append({
                'type': 'identity_with_party',
                'arbitrator': arb_name,
                'details': f"Arbitrator is also a party in related cases"
            })
        
        # Check repeat appointments
        if len(arb_data['cases']) > 3:
            conflicts['orange'].append({
                'type': 'repeat_appointments',
                'arbitrator': arb_name,
                'details': f"Arbitrator appointed in {len(arb_data['cases'])} related cases"
            })
        
        # Check law firm conflicts
        for firm in arb_data['firms']:
            if firm in law_firms and len(law_firms[firm]['cases']) > 1:
                conflicts['orange'].append({
                    'type': 'law_firm_conflict',
                    'arbitrator': arb_name,
                    'firm': firm,
                    'details': f"Arbitrator's firm involved in multiple related cases"
                })
    
    # Orange List (Waivable if parties agree)
    for party_name, party_data in parties.items():
        # Check party involvement in multiple cases
        if len(party_data['cases']) > 1:
            conflicts['orange'].append({
                'type': 'multiple_cases',
                'party': party_name,
                'details': f"Party involved in {len(party_data['cases'])} related cases"
            })
    
    # Green List (Minor issues to disclose)
    for firm, firm_data in law_firms.items():
        if len(firm_data['cases']) > 1:
            conflicts['green'].append({
                'type': 'law_firm_multiple_cases',
                'firm': firm,
                'details': f"Law firm involved in {len(firm_data['cases'])} related cases"
            })
    
    return conflicts

def process_search_results(query, max_concurrent_cases=5):
    """Process search results in parallel and update UI"""
    cases = []
    
    # Create containers for dynamic updates
    progress_container = st.empty()
    stats_container = st.container()
    results_container = st.container()
    analysis_container = st.container()
    
    # Reset counter and data for new search
    st.session_state.cases_counter = ThreadSafeCounter()
    st.session_state.placeholders = {}
    st.session_state.processed_cases = ThreadSafeDict()
    st.session_state.party_info = ThreadSafeDict()
    st.session_state.decision_info = ThreadSafeDict()
    st.session_state.entity_analysis = ThreadSafeDict()
    
    # Reset case details for new search
    st.session_state.all_case_details = {}
    
    try:
        # Fetch first page of cases
        with progress_container:
            st.info(f"Searching for cases matching '{query}'...")
        page_cases = jusmundi_api.search_cases(
            query, 
            page=1,
            count=10, # Limit to 10 cases for now
            include="decisions,parties" 
        )
        
        if not page_cases:
            with progress_container:
                st.warning("No results found. Please try a different search term.")
            return
        
        cases.extend(page_cases)
        
        with progress_container:
            st.info(f"Found {len(cases)} cases. Processing in parallel...")
        
        # Display cases immediately with placeholders
        with results_container:
            st.write("## Results")
            st.write("Case details will load below. Expand sections for more information.")
            st.markdown("---")
            
            # Create simple placeholders for status updates only
            st.session_state.status_placeholders = {}
            for case in cases:
                case_id = case.get('id')
                if case_id:
                    title = case.get('attributes', {}).get('title', 'Untitled Case')
                    # Create a single placeholder for the status of this case
                    st.markdown(f"**Case: {title} ({case_id})**") # Display title
                    status_placeholder = st.empty()
                    status_placeholder.info("⏳ Pending...")
                    st.session_state.status_placeholders[case_id] = status_placeholder
                    st.markdown("***") # Separator
        
        # Initialize storage for detailed data
        if 'all_case_details' not in st.session_state:
            st.session_state.all_case_details = {}
        
        # Create a mapping of case IDs to placeholders for easier updating after thread completion
        case_id_to_placeholder = {case.get('id'): st.session_state.status_placeholders.get(case.get('id')) 
                                 for case in cases if case.get('id')}
        
        # Store all futures to wait for their completion
        all_futures = []
        
        # Process cases in parallel
        with ThreadPoolExecutor(max_workers=max_concurrent_cases) as executor:
            # Submit all tasks to the executor
            future_to_case = {}
            for case in cases:
                if case.get('id'):
                    # We don't pass the placeholder to the process_case function anymore
                    future = executor.submit(process_case, case)
                    future_to_case[future] = case
                    all_futures.append(future)
            
            # Process results as they complete for real-time updates
            for i, future in enumerate(as_completed(future_to_case)):
                case = future_to_case[future]
                case_id = case.get('id')
                
                try:
                    # Wait for the result from the thread
                    result = future.result()
                    
                    # Now update the UI from the main thread
                    if result and result.get('status') == 'success':
                        # Update placeholder 
                        if case_id in case_id_to_placeholder and case_id_to_placeholder[case_id]:
                            case_id_to_placeholder[case_id].success("✅ Done")
                    elif result and result.get('status') == 'error':
                        # Update placeholder with error
                        if case_id in case_id_to_placeholder and case_id_to_placeholder[case_id]:
                            case_id_to_placeholder[case_id].error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
                    processed_count = i + 1
                    
                    # Update progress from main thread
                    with progress_container:
                        st.info(f"Processed {processed_count}/{len(cases)} cases")
                        
                    # Update stats in real-time from main thread
                    with stats_container:
                        st.write("## Case Statistics")
                        status_counts = {}
                        current_cases_data = list(st.session_state.all_case_details.values())
                        for c_data in current_cases_data:
                            if c_data and c_data.get('case'):
                                status = c_data['case'].get('attributes', {}).get('outcome', 'Unknown')
                                status_counts[status] = status_counts.get(status, 0) + 1
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Cases Found", len(cases))
                        col2.metric("Processed Cases", st.session_state.cases_counter.value)
                        col3.metric("Pending Cases", len(cases) - st.session_state.cases_counter.value)
                        
                        st.write("\n**Case Status Breakdown (Processed Cases):**")
                        if status_counts: [st.write(f"- {status}: {count}") for status, count in status_counts.items()]
                        else: st.write("-")
                
                except Exception as e:
                    logger.error(f"Error processing case: {str(e)}")
                    # Update placeholder with error from main thread
                    if case_id in case_id_to_placeholder and case_id_to_placeholder[case_id]:
                        case_id_to_placeholder[case_id].error(f"❌ Error: {str(e)}")

        # Wait for all futures to complete before proceeding with analysis
        with progress_container:
            st.info("Waiting for all cases to complete processing before analysis...")
            
        # Make sure all tasks are fully completed before analysis
        for future in all_futures:
            try:
                future.result()  # Wait for each future to complete
            except Exception as e:
                logger.error(f"Error waiting for future completion: {str(e)}")
        
        # Double-check that all cases are processed
        processed_count = st.session_state.cases_counter.value
        if processed_count < len(cases):
            with progress_container:
                st.warning(f"Only {processed_count}/{len(cases)} cases were fully processed. Analysis may be incomplete.")
        else:
            with progress_container:
                st.success(f"All {len(cases)} cases processed successfully. Proceeding with analysis.")

        # --- ANALYSIS & RENDERING SECTION (After processing loop) ---
        with progress_container:
            st.info("Processing complete. Rendering detailed results...")
            time.sleep(1) # Brief pause

        # Clear the initial results container (which only had expanders and placeholders)
        results_container.empty()

        # Create a new container for the detailed results
        detailed_results_container = st.container()

        with detailed_results_container:
            st.write("## Detailed Results")
            st.markdown("---")
            
            if 'all_case_details' not in st.session_state or not st.session_state.all_case_details:
                st.warning("No case details were processed.")
            else:
                # Iterate through the stored details and render everything
                for case_id, details in st.session_state.all_case_details.items():
                    case = details['case']
                    parties = details['parties']
                    decisions = details['decisions']
                    analysis = details['analysis']
                    
                    attributes = case.get('attributes', {})
                    title = attributes.get('title', 'Untitled Case')
                    
                    # --- Render Case Title and Basic Info (No Expander) ---
                    st.markdown(f"### 📋 Case: {title} ({case_id})")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Organization:** {attributes.get('organization', 'N/A')}")
                        st.write(f"**Outcome:** {attributes.get('outcome', 'N/A')}")
                        st.write(f"**Commencement Date:** {attributes.get('commencement_date', 'N/A')}")
                    with col2:
                        st.write(f"**Treaty:** {attributes.get('treaty', 'N/A')}")
                        st.write(f"**Rules:** {attributes.get('rules', 'N/A')}")
                        st.write(f"**Industry:** {attributes.get('industry', 'N/A')}")
                        monetary_info = attributes.get('monetary_info', {})
                        if monetary_info:
                                st.write("**Monetary Information:**")
                                st.write(f"- Amount: {monetary_info.get('amount', 'N/A')} {monetary_info.get('currency', '')}")
                                st.write(f"- Type: {monetary_info.get('type', 'N/A')}")
                    
                    st.markdown("---")

                    # --- Display Parties ---
                    st.markdown("#### Parties")
                    if details['party_status'] == 'success':
                        party_data = []
                        for party in parties:
                            attrs = party.get('attributes', {})
                            party_data.append({
                                "Name": attrs.get('name', 'N/A'),
                                "Role": attrs.get('role', 'N/A'),
                                "Type": attrs.get('type', 'N/A'),
                                "Nationality": attrs.get('nationality', 'N/A')
                            })
                        if party_data: st.table(party_data)
                        else: st.info("No party data available.")
                    elif details['party_status'] == 'nodata':
                        st.info("No party data found.")
                    else: # error or loading
                        st.warning(f"Party status: {details['party_status']}")
                    
                    # --- Display Decisions ---
                    st.markdown("#### Decisions")
                    if details['decision_status'] == 'success':
                        if not decisions:
                            st.info("No decisions found for this case.")
                        else:
                            challenge_decisions = []
                            other_decisions = []
                            for decision in decisions:
                                decision_attrs = decision.get('attributes', {})
                                title = decision_attrs.get('title', '').lower()
                                description = decision_attrs.get('description', '').lower()
                                is_challenge = any(term in title or term in description 
                                               for term in ['challenge', 'disqualification', 'recusal', 'conflict of interest'])
                                if is_challenge: challenge_decisions.append(decision)
                                else: other_decisions.append(decision)
                            
                            # Display challenge decisions with individual containers
                            if challenge_decisions:
                                st.markdown("##### 🚨 Challenge Decisions")
                                for i, d in enumerate(challenge_decisions): 
                                    # Create a separate container for each decision to avoid nesting
                                    decision_container = st.container()
                                    with decision_container:
                                        display_decision_info(d, True)
                                        st.markdown("---") # Separator between decisions
                            
                            # Display other decisions with individual containers
                            if other_decisions:
                                st.markdown("##### 📄 Other Decisions")
                                for i, d in enumerate(other_decisions): 
                                    # Create a separate container for each decision to avoid nesting
                                    decision_container = st.container()
                                    with decision_container:
                                        display_decision_info(d, False)
                                        st.markdown("---") # Separator between decisions
                                
                            if not challenge_decisions and not other_decisions: 
                                st.info("No decision details could be processed.")
                    elif details['decision_status'] == 'nodata':
                         st.info("No decision data found.")
                    else: # error or loading
                         st.error(f"Decision status: {details['decision_status']}")
                    
                    st.markdown("***") # Separator between cases

                    # Add detailed entity visualization section for each case
                    if details['analysis_status'] == 'success' and details['analysis']:
                        st.markdown("#### 🔍 Extracted Entities")
                        analysis = details['analysis']
                        
                        # Group entities by type
                        entities_by_type = {}
                        for entity in analysis.get('entities', []):
                            entity_type = entity.get('type', 'unknown')
                            if entity_type not in entities_by_type:
                                entities_by_type[entity_type] = []
                            entities_by_type[entity_type].append(entity)
                        
                        # Create tabs for different entity types
                        if entities_by_type:
                            entity_tabs = st.tabs(list(entities_by_type.keys()))
                            for i, (entity_type, entities) in enumerate(entities_by_type.items()):
                                with entity_tabs[i]:
                                    # Create a table of entities
                                    entity_data = []
                                    for entity in entities:
                                        entity_info = {
                                            "ID": entity.get('id', 'Unknown'),
                                            "Name": entity.get('name', 'Unnamed')
                                        }
                                        # Add attributes
                                        for key, value in entity.get('attributes', {}).items():
                                            if value:  # Only add non-empty attributes
                                                entity_info[key] = value
                                        entity_data.append(entity_info)
                                    
                                    if entity_data:
                                        import pandas as pd
                                        df = pd.DataFrame(entity_data)
                                        st.dataframe(df)
                        
                        # Show relations separately - not nested in expanders
                        if analysis.get('relations'):
                            st.markdown("#### Relations")
                            relation_data = []
                            for relation in analysis.get('relations', []):
                                relation_info = {
                                    "Source": relation.get('source', 'Unknown'),
                                    "Target": relation.get('target', 'Unknown'),
                                    "Type": relation.get('type', 'Unknown')
                                }
                                # Add attributes
                                for key, value in relation.get('attributes', {}).items():
                                    if value:  # Only add non-empty attributes
                                        relation_info[key] = value
                                relation_data.append(relation_info)
                            
                            if relation_data:
                                import pandas as pd
                                df = pd.DataFrame(relation_data)
                                st.dataframe(df)

        # --- Final RAG/Conflict Analysis (After rendering loop) ---
        with analysis_container:
            st.write("## Overall Analysis")
            
            # --- Conflict Analysis ---
            st.write("### Conflict Analysis (Based on Loaded Data)")
            with st.spinner("Analyzing potential conflicts based on IBA Guidelines..."):
                # Use all_case_details from session state
                processed_cases = [d['case'] for d in st.session_state.all_case_details.values() if d['case']]
                all_parties_flat = [p for d in st.session_state.all_case_details.values() for p in d['parties']]
                all_decisions_flat = [dec for d in st.session_state.all_case_details.values() for dec in d['decisions']]
                
                # We need a way to reconstruct party_info and decision_info from the stored lists
                # For simplicity, we'll pass the lists and adapt the analyze_conflicts_iba function if needed
                # OR, we recreate the dictionaries here if the function strictly requires them
                party_info_dict = { f"{p.get('id')}": p for p in all_parties_flat }
                decision_info_dict = { f"{dec.get('id')}": dec for dec in all_decisions_flat }

                conflicts = analyze_conflicts_iba(
                    processed_cases,
                    party_info_dict,
                    decision_info_dict
                )
                # ... (rest of conflict display code) ...
                has_conflicts = False
                if conflicts['red']: st.error("### ⛔ Non-Waivable Conflicts (Red List)"); [st.write(f"- **{c.get('type','N/A')}**: {c.get('details','N/A')}") for c in conflicts['red']]; has_conflicts = True
                if conflicts['orange']: st.warning("### ⚠️ Waivable Conflicts (Orange List)"); [st.write(f"- **{c.get('type','N/A')}**: {c.get('details','N/A')}") for c in conflicts['orange']]; has_conflicts = True
                if conflicts['green']: st.success("### ✅ Issues to Disclose (Green List)"); [st.write(f"- **{c.get('type','N/A')}**: {c.get('details','N/A')}") for c in conflicts['green']]; has_conflicts = True
                if not has_conflicts: st.success("No significant conflicts identified based on IBA Guidelines from the loaded data.")

            # --- RAG System Analysis ---
            st.write("### RAG System Analysis")
            with st.spinner("Generating overall analysis using RAG system..."):
                response = rag_system.get_response(query, processed_cases)
                st.info(response)
        
        # Final status update
        with progress_container:
            st.success(f"✅ Processing and Rendering Complete for {st.session_state.cases_counter.value} cases.")
    
    except Exception as e:
        logger.error(f"Error during search or processing: {str(e)}")
        st.error(f"An error occurred during the search: {str(e)}")

# Main app UI
st.title("Arbitrator Challenge Analysis")
st.subheader("Analyze conflicts of interest in international arbitration")

# Initialize thread-safe structures at app startup
if 'cases_counter' not in st.session_state:
    st.session_state.cases_counter = ThreadSafeCounter()
if 'processed_cases' not in st.session_state:
    st.session_state.processed_cases = ThreadSafeDict()
if 'party_info' not in st.session_state:
    st.session_state.party_info = ThreadSafeDict()
if 'decision_info' not in st.session_state:
    st.session_state.decision_info = ThreadSafeDict()
if 'entity_analysis' not in st.session_state:
    st.session_state.entity_analysis = ThreadSafeDict()
if 'all_case_details' not in st.session_state:
    st.session_state.all_case_details = {}
if 'placeholders' not in st.session_state:
    st.session_state.placeholders = {}
if 'status_placeholders' not in st.session_state:
    st.session_state.status_placeholders = {}

# Create tabs
tab1, tab2 = st.tabs(["Search & Analysis", "Knowledge Graph"])

# Initialize session state
if 'searched_cases' not in st.session_state:
    st.session_state.searched_cases = {}
if 'all_cases' not in st.session_state:
    st.session_state.all_cases = []
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None

with tab1:
    # Search interface
    st.subheader("Search for Cases")
    search_container = st.container()
    
    with search_container:
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "Enter search term:",
                placeholder="e.g., company name, country, arbitrator name"
            )
        with col2:
            st.write("")
            st.write("")
            search_button = st.button("Search", type="primary")
    
    # Add filters in sidebar
    st.sidebar.header("Filters")
    status_filter = st.sidebar.multiselect(
        "Case Status",
        ["Pending", "Concluded", "Discontinued", "All"],
        default=["All"]
    )
    
    organization_filter = st.sidebar.multiselect(
        "Organization",
        ["ICSID", "PCA", "ICC", "All"],
        default=["All"]
    )
    
    # Date range filter
    date_col1, date_col2 = st.sidebar.columns(2)
    with date_col1:
        start_date = st.date_input("Start Date", value=None)
    with date_col2:
        end_date = st.date_input("End Date", value=None)
    
    if search_button:
        if not query:
            st.warning("Please enter a search term.")
        else:
            process_search_results(query)

with tab2:
    st.subheader("Knowledge Graph Visualization")
    
    # Add graph filters with expanded defaults
    st.sidebar.header("Graph Filters")
    all_entity_types = list(set([knowledge_graph.graph.nodes[node].get('type', 'unknown') for node in knowledge_graph.graph.nodes()]))
    show_entities = st.sidebar.multiselect(
        "Show Entity Types",
        all_entity_types,
        default=all_entity_types
    )
    
    # Add detailed entity list view
    show_detailed_list = st.checkbox("Show Detailed Entity List", value=False)
    
    # Filter graph based on selected entity types
    filtered_graph = knowledge_graph.graph.copy()
    nodes_to_remove = []
    for node in filtered_graph.nodes():
        node_type = filtered_graph.nodes[node].get('type', '')
        if node_type not in show_entities:
            nodes_to_remove.append(node)
    filtered_graph.remove_nodes_from(nodes_to_remove)
    
    # Display graph
    if filtered_graph.number_of_nodes() > 0:
        st.write(f"### Interactive Graph ({filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges)")
        graph_file = generate_network_graph(filtered_graph)
        if graph_file:
            with open(graph_file, 'r', encoding='utf-8') as f:
                html_data = f.read()
            st.components.v1.html(html_data, height=800)
            try:
                os.unlink(graph_file)
            except Exception:
                pass
        else:
            st.info("No data in the filtered graph. Try adjusting the filters or fetching more cases.")
    else:
        st.info("No data in the filtered graph. Try adjusting the filters or fetching more cases.")
    
    # Show detailed entity list if selected
    if show_detailed_list and filtered_graph.number_of_nodes() > 0:
        st.write("### Detailed Entity List")
        
        # Group by entity type
        entities_by_type = {}
        for node in filtered_graph.nodes():
            node_data = filtered_graph.nodes[node]
            node_type = node_data.get('type', 'unknown')
            if node_type not in entities_by_type:
                entities_by_type[node_type] = []
            
            entity_info = {
                "ID": node,
                "Name": node_data.get('name', 'Unnamed')
            }
            # Add up to 5 additional attributes
            attr_count = 0
            for key, value in node_data.items():
                if key not in ['type', 'name', 'id'] and attr_count < 5:
                    entity_info[key] = value
                    attr_count += 1
            
            entities_by_type[node_type].append(entity_info)
        
        # Display entities by type in expandable sections
        for entity_type, entities in entities_by_type.items():
            with st.expander(f"{entity_type.capitalize()} ({len(entities)})"):
                if entities:
                    # Convert to DataFrame for easier display
                    import pandas as pd
                    df = pd.DataFrame(entities)
                    st.dataframe(df)

# Add debug information in sidebar
with st.sidebar:
    st.title("Debug Information")
    
    # API Status
    st.subheader("API Status")
    try:
        # Try a simple search to check API
        test_result = jusmundi_api.search_cases("test", count=1)
        st.success("✅ Jus Mundi API Connected")
    except Exception as e:
        st.error(f"❌ Jus Mundi API Error: {str(e)}")
    
    try:
        # Try Gemini API
        test_response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="test"
        )
        st.success("✅ Gemini API Connected")
    except Exception as e:
        st.error(f"❌ Gemini API Error: {str(e)}")
    
    # API Info
    st.subheader("API Configuration")
    st.write(f"- Base URL: {jusmundi_api.base_url}")
    
    # Graph Statistics
    st.subheader("Graph Statistics")
    st.write(f"Total Nodes: {knowledge_graph.graph.number_of_nodes()}")
    st.write(f"Total Edges: {knowledge_graph.graph.number_of_edges()}")
    
    # Show node types distribution
    node_types = {}
    for node in knowledge_graph.graph.nodes():
        node_type = knowledge_graph.graph.nodes[node].get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    st.write("Node Types:")
    for ntype, count in node_types.items():
        st.write(f"- {ntype}: {count}")
    
    # Entity Analysis Debug Information
    if 'all_case_details' in st.session_state:
        st.subheader("Entity Analysis Debug")
        for case_id, details in st.session_state.all_case_details.items():
            analysis = details.get('analysis')
            if details['analysis_status'] == 'success' and analysis:
                st.write(f"**Case {case_id}**")
                st.write(f"- Entities: {len(analysis.get('entities', []))}")
                st.write(f"- Relations: {len(analysis.get('relations', []))}")
                st.write(f"- Events: {len(analysis.get('events', []))}")
                st.write(f"- Challenges: {len(analysis.get('challenges', []))}")
                
                # Show entity type counts
                entity_types = {}
                for entity in analysis.get('entities', []):
                    entity_type = entity.get('type', 'unknown')
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                st.write("Entity Types:")
                for etype, count in entity_types.items():
                    st.write(f"  - {etype}: {count}")
            elif details['analysis_status'] in ['fail', 'error']:
                st.error(f"Case {case_id}: {details.get('analysis_error_msg', 'Unknown error')}")
            st.markdown("---")
    
    # Clear Graph button
    if st.button("Clear Knowledge Graph"):
        knowledge_graph.graph.clear()
        st.success("Graph cleared!")
        st.rerun()
    
    # Clear session state button
    if st.button("Clear Session Data"):
        st.session_state.clear()
        st.success("Session data cleared!")
        st.rerun() 