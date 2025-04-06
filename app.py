from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from google import genai
from jusmundi_api import JusMundiAPI
from knowledge_graph import KnowledgeGraph
from rag_system import RAGSystem
from pyvis.network import Network
import networkx as nx

# Load environment variables
load_dotenv()

# Initialize Gemini
genai_client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

app = Flask(__name__)

# Initialize components
jusmundi_api = JusMundiAPI(api_key=os.getenv('JUSMUNDI_API_KEY'))
knowledge_graph = KnowledgeGraph()
rag_system = RAGSystem(knowledge_graph)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    # Create a Pyvis network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Add nodes and edges from the knowledge graph
    for node in knowledge_graph.graph.nodes():
        node_data = knowledge_graph.graph.nodes[node]
        if node_data['type'] == 'case':
            net.add_node(node, 
                        label=node_data.get('title', 'Unknown Case'),
                        color='#FF9999',
                        shape='box')
        elif node_data['type'] == 'arbitrator':
            net.add_node(node,
                        label=node_data.get('name', 'Unknown Arbitrator'),
                        color='#99FF99',
                        shape='ellipse')
        elif node_data['type'] == 'challenge':
            net.add_node(node,
                        label=node_data.get('grounds', 'Unknown Challenge'),
                        color='#9999FF',
                        shape='diamond')
    
    # Add edges
    for edge in knowledge_graph.graph.edges():
        edge_data = knowledge_graph.graph.edges[edge]
        net.add_edge(edge[0], edge[1], 
                    title=edge_data.get('type', ''),
                    color='#CCCCCC')
    
    # Generate the HTML
    net.save_graph("templates/graph.html")
    return render_template('graph.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Get relevant cases from Jus Mundi API
        cases = jusmundi_api.search_cases(query)
        
        # For each case, get related cases and arbitrator challenges
        enriched_cases = []
        for case in cases:
            case_id = case.get('id')
            if case_id:
                # Get related cases
                related_cases = jusmundi_api.get_related_cases(case_id)
                case['related_cases'] = related_cases
                
                # Get arbitrator challenges
                arbitrators = case.get('arbitrators', [])
                for arbitrator in arbitrators:
                    arbitrator_id = arbitrator.get('id')
                    if arbitrator_id:
                        challenges = jusmundi_api.get_arbitrator_challenges(arbitrator_id)
                        arbitrator['challenges'] = challenges
            
            enriched_cases.append(case)
        
        # Update knowledge graph with enriched cases
        knowledge_graph.update_graph(enriched_cases)
        
        # Get RAG-enhanced response
        response = rag_system.get_response(query, enriched_cases)
        
        return jsonify({
            'response': response,
            'cases': enriched_cases
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 