from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from jusmundi_api import JusMundiAPI
from knowledge_graph import KnowledgeGraph
from rag_system import RAGSystem

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize components
jusmundi_api = JusMundiAPI(api_key=os.getenv('JUSMUNDI_API_KEY'))
knowledge_graph = KnowledgeGraph()
rag_system = RAGSystem(knowledge_graph)

@app.route('/')
def index():
    return render_template('index.html')

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