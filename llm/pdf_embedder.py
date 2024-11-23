# llm/pdf_embedder.py

from typing import List, Dict, Optional, Tuple
import os
import json
from datetime import datetime
import pdfplumber
import tabula
import numpy as np
import faiss
from tqdm import tqdm
import tiktoken
from .embedding import Embedding

class PDFEmbedder:
    def __init__(
        self,
        embedder: Embedding,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        index_path: str = "faiss_index",
        metadata_path: str = "pdf_metadata.json",
        encoding_name: str = "cl100k_base"  # Pour tiktoken
    ):
        """
        Initialize PDF Embedder.
        
        Parameters:
        - embedder: Instance of Embedding class
        - chunk_size: Number of tokens per chunk
        - chunk_overlap: Number of overlapping tokens between chunks
        - index_path: Path to save FAISS index
        - metadata_path: Path to save document metadata
        - encoding_name: Tiktoken encoding to use
        """
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize or load FAISS index and metadata
        self.dimension = self._get_embedding_dimension()
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
        
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension by testing with a sample text."""
        sample_embedding = self.embedder.embed("Sample text")[0]
        return len(sample_embedding)
        
    def _load_or_create_index(self) -> faiss.Index:
        """Load existing FAISS index or create new one."""
        if os.path.exists(f"{self.index_path}.index"):
            return faiss.read_index(f"{self.index_path}.index")
        return faiss.IndexFlatL2(self.dimension)
        
    def _load_metadata(self) -> Dict:
        """Load existing metadata or create new."""
        try:
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"documents": {}, "chunks": []}
            
    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, f"{self.index_path}.index")
        
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text and tables from PDF."""
        text_parts = []
        
        # Extract text with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
                
        # Extract tables with tabula
        tables = tabula.read_pdf(pdf_path, pages='all')
        for table in tables:
            text_parts.append(table.to_string())
            
        return "\n".join(text_parts)
        
    def _create_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Create overlapping chunks from text.
        Returns list of (chunk_text, start_idx, end_idx).
        """
        tokens = self.encoding.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            # Get chunk tokens
            chunk_tokens = tokens[i:i + self.chunk_size]
            
            # Decode chunk
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Store chunk with indices
            chunks.append((
                chunk_text,
                i,
                min(i + self.chunk_size, len(tokens))
            ))
            
            # Move to next chunk considering overlap
            i += self.chunk_size - self.chunk_overlap
            
        return chunks
        
    def embed_pdfs(
        self,
        pdf_paths: List[str],
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict:
        """
        Process PDFs and store embeddings in FAISS.
        
        Parameters:
        - pdf_paths: List of paths to PDF files
        - batch_size: Number of chunks to embed at once
        - verbose: Whether to show progress bars
        
        Returns:
        - Dict with processing statistics
        """
        stats = {
            "processed_pdfs": 0,
            "total_chunks": 0,
            "failed_pdfs": []
        }
        
        # Process each PDF
        for pdf_path in tqdm(pdf_paths, desc="Processing PDFs", disable=not verbose):
            try:
                # Extract text
                text = self._extract_text_from_pdf(pdf_path)
                
                # Create document metadata
                doc_id = str(len(self.metadata["documents"]))
                doc_metadata = {
                    "path": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "processed_at": datetime.now().isoformat(),
                    "num_tokens": len(self.encoding.encode(text)),
                    "chunks": []
                }
                
                # Create chunks
                chunks = self._create_chunks(text)
                current_batch = []
                chunk_metadata = []
                
                # Process chunks in batches
                for chunk_idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
                    current_batch.append(chunk_text)
                    chunk_metadata.append({
                        "doc_id": doc_id,
                        "chunk_idx": chunk_idx,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "text": chunk_text
                    })
                    
                    # Process batch
                    if len(current_batch) >= batch_size or chunk_idx == len(chunks) - 1:
                        # Generate embeddings
                        embeddings = self.embedder.embed(current_batch)
                        
                        # Add to FAISS index
                        self.index.add(np.array(embeddings, dtype=np.float32))
                        
                        # Update metadata
                        for meta in chunk_metadata:
                            self.metadata["chunks"].append(meta)
                            doc_metadata["chunks"].append(len(self.metadata["chunks"]) - 1)
                            
                        # Clear batch
                        current_batch = []
                        chunk_metadata = []
                        
                # Save document metadata
                self.metadata["documents"][doc_id] = doc_metadata
                
                stats["processed_pdfs"] += 1
                stats["total_chunks"] += len(chunks)
                
            except Exception as e:
                stats["failed_pdfs"].append({
                    "path": pdf_path,
                    "error": str(e)
                })
                
            # Save progress
            self._save_index()
            self._save_metadata()
            
        return stats
    
    def search(
        self,
        query: str,
        k: int = 5,
        return_documents: bool = True
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Parameters:
        - query: Search query
        - k: Number of results to return
        - return_documents: Whether to include document metadata
        
        Returns:
        - List of results with scores and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed(query)[0]
        
        # Search in FAISS
        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata["chunks"]):
                continue
                
            chunk_meta = self.metadata["chunks"][idx]
            result = {
                "score": float(score),
                "chunk": chunk_meta
            }
            
            if return_documents:
                doc_meta = self.metadata["documents"][chunk_meta["doc_id"]]
                result["document"] = doc_meta
                
            results.append(result)
            
        return results

