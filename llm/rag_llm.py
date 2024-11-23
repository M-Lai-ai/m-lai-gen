# llm/rag_llm.py

from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json
from .llm import LLM
from .pdf_embedder import PDFEmbedder
from .website_embedder import WebsiteEmbedder

@dataclass
class SearchResult:
    """Represents a search result from embedders."""
    source_type: str  # 'pdf' or 'website'
    content: str
    score: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return {
            "source_type": self.source_type,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }

class RAGLLM:
    def __init__(
        self,
        llm: LLM,
        pdf_embedder: Optional[PDFEmbedder] = None,
        website_embedder: Optional[WebsiteEmbedder] = None,
        num_sources: int = 5,
        score_threshold: float = 0.8,
        history_file: str = "rag_history.json",
        verbose: bool = False
    ):
        """
        Initialize RAG (Retrieval-Augmented Generation) LLM.
        
        Parameters:
        - llm: LLM instance for generation
        - pdf_embedder: PDFEmbedder instance (optional)
        - website_embedder: WebsiteEmbedder instance (optional)
        - num_sources: Number of sources to retrieve
        - score_threshold: Minimum similarity score for sources
        - history_file: File to store generation history
        - verbose: Whether to print detailed information
        """
        if not pdf_embedder and not website_embedder:
            raise ValueError("At least one embedder (PDF or Website) must be provided")
            
        self.llm = llm
        self.pdf_embedder = pdf_embedder
        self.website_embedder = website_embedder
        self.num_sources = num_sources
        self.score_threshold = score_threshold
        self.history_file = history_file
        self.verbose = verbose
        
    def _search_sources(self, query: str) -> List[SearchResult]:
        """Search all available sources."""
        results = []
        
        # Search PDFs if available
        if self.pdf_embedder:
            pdf_results = self.pdf_embedder.search(
                query=query,
                k=self.num_sources,
                return_documents=True
            )
            
            for result in pdf_results:
                if result["score"] >= self.score_threshold:
                    results.append(SearchResult(
                        source_type="pdf",
                        content=result["chunk"]["text"],
                        score=result["score"],
                        metadata={
                            "document": result["document"]["filename"],
                            "page": result["chunk"]["doc_id"],
                            "chunk_index": result["chunk"]["chunk_idx"]
                        }
                    ))
        
        # Search websites if available
        if self.website_embedder:
            web_results = self.website_embedder.search(
                query=query,
                k=self.num_sources,
                return_websites=True
            )
            
            for result in web_results:
                if result["score"] >= self.score_threshold:
                    results.append(SearchResult(
                        source_type="website",
                        content=result["chunk"]["text"],
                        score=result["score"],
                        metadata={
                            "url": result["url"],
                            "domain": result["website"]["domain"],
                            "chunk_index": result["chunk"]["chunk_idx"]
                        }
                    ))
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self.num_sources]
    
    def _create_context_prompt(
        self,
        query: str,
        search_results: List[SearchResult]
    ) -> str:
        """Create prompt with context from search results."""
        prompt_parts = [
            "Please answer the question based on the following context.",
            "\nContext:",
        ]
        
        for i, result in enumerate(search_results, 1):
            source_info = (
                f"Document: {result.metadata['document']}"
                if result.source_type == "pdf"
                else f"Website: {result.metadata['url']}"
            )
            
            prompt_parts.extend([
                f"\nSource {i} ({source_info}):",
                result.content
            ])
        
        prompt_parts.extend([
            "\nQuestion:",
            query,
            "\nPlease provide a detailed answer using the provided context. "
            "If the context doesn't contain relevant information, please indicate that."
        ])
        
        return "\n".join(prompt_parts)
    
    def _save_to_history(
        self,
        query: str,
        sources: List[SearchResult],
        response: str
    ):
        """Save interaction to history file."""
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "sources": [source.to_dict() for source in sources],
            "response": response
        }
        
        history.append(entry)
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2)
    
    def generate(
        self,
        query: str,
        return_sources: bool = False
    ) -> Union[str, Tuple[str, List[SearchResult]]]:
        """
        Generate response using RAG.
        
        Parameters:
        - query: User query
        - return_sources: Whether to return source information
        
        Returns:
        - Response text (and optionally sources)
        """
        if self.verbose:
            print("Searching relevant sources...")
        
        # Search for relevant sources
        search_results = self._search_sources(query)
        
        if self.verbose:
            print(f"Found {len(search_results)} relevant sources")
            for i, result in enumerate(search_results, 1):
                print(f"\nSource {i} (score: {result.score:.4f}):")
                if result.source_type == "pdf":
                    print(f"Document: {result.metadata['document']}")
                else:
                    print(f"URL: {result.metadata['url']}")
        
        # Create prompt with context
        prompt = self._create_context_prompt(query, search_results)
        
        if self.verbose:
            print("\nGenerating response...")
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Save to history
        self._save_to_history(query, search_results, response)
        
        if return_sources:
            return response, search_results
        return response
    
    def get_history(self) -> List[Dict]:
        """Get generation history."""
        try:
            with open(self.history_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

