# LLM x Law: Jus Mundi Confilict Arbitrator Challenge Analysis

A sophisticated knowledge graph based RAG system for analyzing conflicts of interest in international arbitration, leveraging multiple LLMs and interactive visualization. Built for the Stanford CodeX LegalTech Hackathon.

## Key Features

- 🔍 Smart Search: Integration with Jus Mundi API for comprehensive case data
- 🕸️ Knowledge Graph: Advanced relationship analysis and visualization
- 🤖 Multi-LLM RAG System: Context-aware responses using Gemini and Groq
- 📊 Interactive Visualization: Dynamic network graphs using Pyvis
- 🔄 Concurrent Processing: Thread-safe data handling for better performance
- 🎯 Focused Analysis: Specialized in arbitrator challenges and conflicts of interest

## Technical Stack

- **Frontend**: Streamlit for interactive web interface
- **Backend**: 
  - Knowledge Graph for relationship mapping
  - RAG system for intelligent query processing
  - Thread-safe data structures
- **AI Models**:
  - Google Gemini
  - Groq LLM
- **APIs & Libraries**:
  - Jus Mundi API
  - Pyvis for network visualization
  - NetworkX for graph operations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file with the following variables:
```
JUSMUNDI_API_KEY=your_api_key
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

3. Run the application:
```bash
streamlit run streamlit_app.py
```

## Usage

1. Launch the Streamlit application
2. Enter your search query about arbitration cases or challenges
3. Explore the interactive visualization
4. View AI-generated analysis and insights
5. Navigate through related cases and relationships

## Key Components

- `streamlit_app.py`: Main Streamlit application
- `jusmundi_api.py`: Jus Mundi API integration
- `knowledge_graph.py`: Graph-based relationship management
- `rag_system.py`: Multi-LLM RAG implementation

## Features in Detail

### Knowledge Graph
- Entity relationship mapping
- Conflict detection
- Pattern recognition
- Interactive visualization

### RAG System
- Context-aware query processing
- Multi-LLM integration
- Intelligent response generation
- Dynamic knowledge updates

### Thread-Safe Processing
- Concurrent data handling
- Safe state management
- Performance optimization
- Reliable data updates

## Future Enhancements

- Enhanced relationship analysis algorithms
- Additional LLM integrations
- Advanced visualization features
- API endpoint for third-party integration
- Expanded knowledge graph capabilities

## License

MIT

## Team

Created for Stanford CodeX LegalTech Hackathon
