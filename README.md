# Arbitrator Challenge Analysis

A knowledge graph-based RAG system for analyzing conflicts of interest in international arbitration, built for the Jus Mundi Challenge.

## Features

- Integration with Jus Mundi API for case data
- Knowledge graph using Neo4j for relationship analysis
- RAG system for context-aware responses
- Modern web interface for easy interaction

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with the following variables:
```
JUSMUNDI_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

3. Start Neo4j database:
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

4. Run the application:
```bash
python app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Enter your search query in the search box
3. View the analysis and related cases

## Architecture

- `app.py`: Main Flask application
- `jusmundi_api.py`: Jus Mundi API integration
- `knowledge_graph.py`: Neo4j knowledge graph implementation
- `rag_system.py`: RAG system for context-aware responses
- `templates/index.html`: Web interface

## License

MIT
