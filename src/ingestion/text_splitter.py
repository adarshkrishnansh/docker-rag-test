from typing import List, Optional
from dataclasses import dataclass
import re


@dataclass
class TextChunk:
    content: str
    metadata: dict
    start_index: int
    end_index: int


class TextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        secondary_separator: str = "\n",
        length_function: callable = len
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.secondary_separator = secondary_separator
        self.length_function = length_function
    
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[TextChunk]:
        if not text:
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # First, try to split by primary separator
        splits = self._split_by_separator(text, self.separator)
        
        # Process each split
        current_chunks = []
        current_length = 0
        start_index = 0
        
        for split in splits:
            split_length = self.length_function(split)
            
            # If adding this split would exceed chunk size
            if current_length + split_length > self.chunk_size and current_chunks:
                # Create chunk from current chunks
                chunk_content = self.separator.join(current_chunks)
                end_index = start_index + len(chunk_content)
                
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata={**metadata, 'chunk_index': len(chunks)},
                    start_index=start_index,
                    end_index=end_index
                ))
                
                # Keep overlap
                overlap_chunks = self._get_overlap_chunks(current_chunks, self.chunk_overlap)
                current_chunks = overlap_chunks
                current_length = sum(self.length_function(c) for c in current_chunks)
                start_index = end_index - sum(len(c) + len(self.separator) for c in overlap_chunks[:-1]) - len(overlap_chunks[-1]) if overlap_chunks else end_index
            
            # If split itself is too large, split it further
            if split_length > self.chunk_size:
                sub_chunks = self._split_large_text(split, start_index)
                chunks.extend(sub_chunks)
                current_chunks = []
                current_length = 0
                if chunks:
                    start_index = chunks[-1].end_index
            else:
                current_chunks.append(split)
                current_length += split_length
        
        # Don't forget the last chunk
        if current_chunks:
            chunk_content = self.separator.join(current_chunks)
            chunks.append(TextChunk(
                content=chunk_content,
                metadata={**metadata, 'chunk_index': len(chunks)},
                start_index=start_index,
                end_index=start_index + len(chunk_content)
            ))
        
        return chunks
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        if separator:
            splits = text.split(separator)
            # Keep the separator with the text for context
            return [s for s in splits if s.strip()]
        return [text]
    
    def _split_large_text(self, text: str, start_index: int) -> List[TextChunk]:
        # For text that's too large for a single chunk, split by secondary separator
        if self.secondary_separator and self.secondary_separator != self.separator:
            secondary_splits = self._split_by_separator(text, self.secondary_separator)
            
            chunks = []
            current_text = ""
            chunk_start = start_index
            
            for split in secondary_splits:
                if self.length_function(current_text + split) <= self.chunk_size:
                    current_text += self.secondary_separator + split if current_text else split
                else:
                    if current_text:
                        chunks.append(TextChunk(
                            content=current_text,
                            metadata={'chunk_index': 0},  # Will be updated
                            start_index=chunk_start,
                            end_index=chunk_start + len(current_text)
                        ))
                        chunk_start = chunk_start + len(current_text) + len(self.secondary_separator)
                    current_text = split
            
            if current_text:
                chunks.append(TextChunk(
                    content=current_text,
                    metadata={'chunk_index': 0},
                    start_index=chunk_start,
                    end_index=chunk_start + len(current_text)
                ))
            
            return chunks
        
        # If no secondary separator or still too large, split by character count
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk_end = min(i + self.chunk_size, len(text))
            chunks.append(TextChunk(
                content=text[i:chunk_end],
                metadata={'chunk_index': 0},
                start_index=start_index + i,
                end_index=start_index + chunk_end
            ))
        
        return chunks
    
    def _get_overlap_chunks(self, chunks: List[str], target_overlap: int) -> List[str]:
        if not chunks or target_overlap <= 0:
            return []
        
        overlap_text = ""
        overlap_chunks = []
        
        # Work backwards through chunks to get overlap
        for chunk in reversed(chunks):
            if self.length_function(overlap_text + chunk) <= target_overlap:
                overlap_chunks.insert(0, chunk)
                overlap_text = self.separator.join(overlap_chunks)
            else:
                break
        
        return overlap_chunks