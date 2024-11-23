# llm/website_embedder.py

from typing import List, Dict, Optional, Tuple, Set
import os
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import numpy as np
import faiss
from tqdm import tqdm
import tiktoken
import time
from concurrent.futures import ThreadPoolExecutor
from .embedding import Embedding

class WebsiteEmbedder:
    def __init__(
        self,
        embedder: Embedding,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_pages: int = 100,
        index_path: str = "website_faiss_index",
        metadata_path: str = "website_metadata.json",
        encoding_name: str = "cl100k_base",
        user_agent: str = "Mozilla/5.0 WebsiteEmbedder Bot",
        request_delay: float = 1.0
    ):
        """
        Initialize Website Embedder.
        
        Parameters:
        - embedder: Instance of Embedding class
        - chunk_size: Number of tokens per chunk
        - chunk_overlap: Number of overlapping tokens
        - max_pages: Maximum number of pages to crawl per domain
        - index_path: Path to save FAISS index
        - metadata_path: Path to save website metadata
        - encoding_name: Tiktoken encoding name
        - user_agent: User agent for web requests
        - request_delay: Delay between requests in seconds
        """
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_pages = max_pages
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.user_agent = user_agent
        self.request_delay = request_delay
        
        # Initialize or load FAISS index and metadata
        self.dimension = self._get_embedding_dimension()
        self.index = self._load_or_create_index()
        self.metadata = self._load_metadata()
        
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension by testing."""
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
            return {"websites": {}, "pages": {}, "chunks": []}
            
    def _save_index(self):
        """Save FAISS index to disk."""
        faiss.write_index(self.index, f"{self.index_path}.index")
        
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_domain(self, url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove very short lines
        lines = [line for line in text.split('\n') if len(line.strip()) > 20]
        return '\n'.join(lines)

    def _extract_content(self, url: str) -> Tuple[str, List[str]]:
        """
        Extract content and links from webpage.
        Returns (cleaned_text, list_of_links)
        """
        headers = {'User-Agent': self.user_agent}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer']):
                element.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n')
            cleaned_text = self._clean_text(text)
            
            # Extract links
            links = []
            for a in soup.find_all('a', href=True):
                link = urljoin(url, a['href'])
                if self._get_domain(link) == self._get_domain(url):
                    links.append(link)
                    
            return cleaned_text, list(set(links))
            
        except Exception as e:
            raise Exception(f"Error extracting content from {url}: {str(e)}")

    def _create_chunks(self, text: str) -> List[Tuple[str, int, int]]:
        """Create overlapping chunks from text."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        i = 0
        while i < len(tokens):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            chunks.append((
                chunk_text,
                i,
                min(i + self.chunk_size, len(tokens))
            ))
            
            i += self.chunk_size - self.chunk_overlap
            
        return chunks

    def crawl_and_embed(
        self,
        start_urls: List[str],
        batch_size: int = 32,
        max_workers: int = 4,
        verbose: bool = True
    ) -> Dict:
        """
        Crawl websites and create embeddings.
        
        Parameters:
        - start_urls: List of starting URLs
        - batch_size: Number of chunks to embed at once
        - max_workers: Maximum number of concurrent crawlers
        - verbose: Whether to show progress
        
        Returns:
        - Statistics about the crawling process
        """
        stats = {
            "processed_websites": 0,
            "processed_pages": 0,
            "total_chunks": 0,
            "failed_urls": []
        }
        
        for start_url in start_urls:
            try:
                domain = self._get_domain(start_url)
                
                # Create website metadata
                website_id = str(len(self.metadata["websites"]))
                website_metadata = {
                    "domain": domain,
                    "start_url": start_url,
                    "crawled_at": datetime.now().isoformat(),
                    "pages": []
                }
                
                # Initialize crawling
                to_visit = {start_url}
                visited = set()
                current_batch = []
                chunk_metadata = []
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    while to_visit and len(visited) < self.max_pages:
                        url = to_visit.pop()
                        if url in visited:
                            continue
                            
                        # Extract content
                        try:
                            content, links = self._extract_content(url)
                            visited.add(url)
                            
                            # Update URLs to visit
                            to_visit.update(links - visited)
                            
                            # Create page metadata
                            page_id = str(len(self.metadata["pages"]))
                            page_metadata = {
                                "url": url,
                                "website_id": website_id,
                                "processed_at": datetime.now().isoformat(),
                                "num_tokens": len(self.encoding.encode(content)),
                                "chunks": []
                            }
                            
                            # Process chunks
                            chunks = self._create_chunks(content)
                            for chunk_idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
                                current_batch.append(chunk_text)
                                chunk_metadata.append({
                                    "website_id": website_id,
                                    "page_id": page_id,
                                    "chunk_idx": chunk_idx,
                                    "start_idx": start_idx,
                                    "end_idx": end_idx,
                                    "text": chunk_text,
                                    "url": url
                                })
                                
                                # Process batch if full
                                if len(current_batch) >= batch_size:
                                    embeddings = self.embedder.embed(current_batch)
                                    self.index.add(np.array(embeddings, dtype=np.float32))
                                    
                                    # Update metadata
                                    for meta in chunk_metadata:
                                        self.metadata["chunks"].append(meta)
                                        page_metadata["chunks"].append(
                                            len(self.metadata["chunks"]) - 1
                                        )
                                    
                                    current_batch = []
                                    chunk_metadata = []
                            
                            # Save page metadata
                            self.metadata["pages"][page_id] = page_metadata
                            website_metadata["pages"].append(page_id)
                            
                            stats["processed_pages"] += 1
                            stats["total_chunks"] += len(chunks)
                            
                            # Delay between requests
                            time.sleep(self.request_delay)
                            
                        except Exception as e:
                            stats["failed_urls"].append({
                                "url": url,
                                "error": str(e)
                            })
                
                # Process remaining batch
                if current_batch:
                    embeddings = self.embedder.embed(current_batch)
                    self.index.add(np.array(embeddings, dtype=np.float32))
                    
                    for meta in chunk_metadata:
                        self.metadata["chunks"].append(meta)
                        self.metadata["pages"][meta["page_id"]]["chunks"].append(
                            len(self.metadata["chunks"]) - 1
                        )
                
                # Save website metadata
                self.metadata["websites"][website_id] = website_metadata
                stats["processed_websites"] += 1
                
                # Save progress
                self._save_index()
                self._save_metadata()
                
            except Exception as e:
                stats["failed_urls"].append({
                    "url": start_url,
                    "error": str(e)
                })
                
        return stats

    def search(
        self,
        query: str,
        k: int = 5,
        return_websites: bool = True
    ) -> List[Dict]:
        """
        Search for similar chunks.
        
        Parameters:
        - query: Search query
        - k: Number of results
        - return_websites: Whether to include website metadata
        
        Returns:
        - List of results with scores and metadata
        """
        query_embedding = self.embedder.embed(query)[0]
        
        scores, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            k
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata["chunks"]):
                continue
                
            chunk_meta = self.metadata["chunks"][idx]
            result = {
                "score": float(score),
                "chunk": chunk_meta,
                "url": chunk_meta["url"]
            }
            
            if return_websites:
                website_meta = self.metadata["websites"][chunk_meta["website_id"]]
                page_meta = self.metadata["pages"][chunk_meta["page_id"]]
                result["website"] = website_meta
                result["page"] = page_meta
                
            results.append(result)
            
        return results

# Example usage:
if __name__ == "__main__":
    # Initialize embedding
    embedder = Embedding(
        provider="openai",
        model="text-embedding-ada-002"
    )
    
    # Initialize website embedder
    website_embedder = WebsiteEmbedder(
        embedder=embedder,
        chunk_size=500,
        chunk_overlap=50,
        max_pages=100
    )
    
    # Crawl and embed websites
    start_urls = [
        "https://example.com",
        "https://another-example.com"
    ]
    
    stats = website_embedder.crawl_and_embed(
        start_urls,
        verbose=True
    )
    
    print("\nCrawling complete!")
    print(f"Processed websites: {stats['processed_websites']}")
    print(f"Processed pages: {stats['processed_pages']}")
    print(f"Total chunks: {stats['total_chunks']}")
    
    if stats['failed_urls']:
        print("\nFailed URLs:")
        for failure in stats['failed_urls']:
            print(f"- {failure['url']}: {failure['error']}")
    
    # Search example
    results = website_embedder.search(
        query="What is your privacy policy?",
        k=3
    )
    
    print("\nSearch results:")
    for result in results:
        print(f"\nScore: {result['score']:.4f}")
        print(f"URL: {result['url']}")
        print(f"Text: {result['chunk']['text'][:200]}...")
