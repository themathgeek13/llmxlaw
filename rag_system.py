from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from typing import List, Dict, Any

class RAGSystem:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Neo4jVector(
            embedding=self.embeddings,
            url="bolt://localhost:7687",
            username="neo4j",
            password="password",  # Replace with your Neo4j credentials
            index_name="case_embeddings",
            node_label="Case",
            text_node_properties=["title", "challenge_grounds", "outcome"],
            embedding_node_property="embedding"
        )
        self.llm = ChatOpenAI(temperature=0)

    def get_response(self, query: str, cases: List[Dict[str, Any]]) -> str:
        """
        Get a context-aware response using RAG
        """
        # Update vector store with new cases
        for case in cases:
            self.vector_store.add_texts(
                texts=[f"{case.get('title', '')} {case.get('challenge_grounds', '')} {case.get('outcome', '')}"],
                metadatas=[{"id": case['id']}]
            )

        # Create retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )

        # Get response
        response = qa_chain.run(query)
        return response

    def get_similar_cases(self, case_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar cases based on embeddings
        """
        case = self.knowledge_graph.get_case_details(case_id)
        if not case:
            return []

        case_text = f"{case.get('title', '')} {case.get('challenge_grounds', '')} {case.get('outcome', '')}"
        similar_docs = self.vector_store.similarity_search(case_text, k=limit)
        
        return [doc.metadata for doc in similar_docs] 