from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from .validation import (
    validate_chunk_parameters,
    validate_and_default_string,
    validate_and_default_iterable,
    log_parameter_correction
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextChunk:
    chunk_id: int
    source_id: str
    text: str
    
    def __post_init__(self) -> None:
        if self.chunk_id < 0:
            raise ValueError(
                f"TextChunk validation failed: chunk_id must be non-negative. "
                f"Got: {self.chunk_id} (type: {type(self.chunk_id).__name__}). "
                f"Expected: integer >= 0. "
                f"Chunk details: source_id='{self.source_id}', text_length={len(self.text)}. "
                f"Suggestion: Ensure chunk_id is assigned as a sequential non-negative integer starting from 0."
            )
        if not self.source_id or not self.source_id.strip():
            raise ValueError(
                f"TextChunk validation failed: source_id must be non-empty. "
                f"Got: '{self.source_id}' (type: {type(self.source_id).__name__}, length: {len(self.source_id) if self.source_id else 0}). "
                f"Expected: non-empty string with meaningful identifier. "
                f"Chunk details: chunk_id={self.chunk_id}, text_length={len(self.text)}. "
                f"Suggestion: Provide a meaningful source identifier like 'doc_1', 'article_123', or filename."
            )


def chunk_text(text: str, chunk_size: int = 4000, overlap: int = 200) -> list[str]:
    """
    Split text into chunks with parameter validation and default substitution.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (validated to be positive)
        overlap: Overlap between chunks (validated to be non-negative and < chunk_size)
        
    Returns:
        List of text chunks
    """
    # Validate text parameter
    original_text = text
    validated_text = validate_and_default_string(
        text, "", "text", "TextChunker.chunk_text", allow_empty=True
    )
    if validated_text != original_text:
        log_parameter_correction(original_text, validated_text, "text", "TextChunker.chunk_text", "invalid text")
    
    if not validated_text:
        return []
    
    # Validate chunk parameters with interdependent constraints
    original_chunk_size, original_overlap = chunk_size, overlap
    validated_chunk_size, validated_overlap = validate_chunk_parameters(
        chunk_size, overlap, "TextChunker.chunk_text"
    )
    
    if validated_chunk_size != original_chunk_size:
        log_parameter_correction(original_chunk_size, validated_chunk_size, "chunk_size", "TextChunker.chunk_text", "invalid chunk_size")
    if validated_overlap != original_overlap:
        log_parameter_correction(original_overlap, validated_overlap, "overlap", "TextChunker.chunk_text", "invalid overlap")

    chunks: list[str] = []
    start = 0
    length = len(validated_text)
    step = validated_chunk_size - validated_overlap

    while start < length:
        end = min(length, start + validated_chunk_size)
        chunk = validated_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start += step

    return chunks


def chunk_records(records: Iterable[tuple[str, str]], chunk_size: int = 4000, overlap: int = 200) -> list[TextChunk]:
    """
    Chunk multiple records with comprehensive parameter validation and error handling.
    
    Args:
        records: Iterable of (source_id, text) tuples
        chunk_size: Size of each chunk (validated to be positive)
        overlap: Overlap between chunks (validated to be non-negative and < chunk_size)
        
    Returns:
        List of TextChunk objects
    """
    # Validate records parameter
    validated_records = validate_and_default_iterable(
        records, [], "records", "TextChunker.chunk_records", allow_empty=True
    )

    
    # Validate chunk parameters
    original_chunk_size, original_overlap = chunk_size, overlap
    validated_chunk_size, validated_overlap = validate_chunk_parameters(
        chunk_size, overlap, "TextChunker.chunk_records"
    )
    
    if validated_chunk_size != original_chunk_size:
        log_parameter_correction(original_chunk_size, validated_chunk_size, "chunk_size", "TextChunker.chunk_records", "invalid chunk_size")
    if validated_overlap != original_overlap:
        log_parameter_correction(original_overlap, validated_overlap, "overlap", "TextChunker.chunk_records", "invalid overlap")

    chunks: list[TextChunk] = []
    chunk_id = 0
    processed_records = 0
    skipped_records = 0
    total_text_length = 0
    
    for record_index, record in enumerate(validated_records):
        try:
            # Validate record structure
            if not isinstance(record, (tuple, list)) or len(record) != 2:
                logger.warning(
                    f"TextChunker.chunk_records: Invalid record format at index {record_index}. "
                    f"Expected: tuple/list with 2 elements (source_id, text). "
                    f"Got: {type(record).__name__} with {len(record) if hasattr(record, '__len__') else 'unknown'} elements. "
                    f"Record value: {record}. "
                    f"Skipping this record and continuing with next. "
                    f"Suggestion: Ensure each record is a (source_id, text) tuple."
                )
                skipped_records += 1
                continue
                
            source_id, text = record
            
            # Validate source_id and text
            validated_source_id = validate_and_default_string(
                source_id, f"unknown_source_{record_index}", "source_id", "TextChunker.chunk_records"
            )
            validated_text = validate_and_default_string(
                text, "", "text", "TextChunker.chunk_records", allow_empty=True
            )
            
            if validated_source_id != source_id:
                log_parameter_correction(source_id, validated_source_id, "source_id", "TextChunker.chunk_records", "invalid source_id")
            if validated_text != text:
                log_parameter_correction(text, validated_text, "text", "TextChunker.chunk_records", "invalid text")
            
            # Skip empty texts
            if not validated_text.strip():
                logger.debug(
                    f"TextChunker.chunk_records: Skipping empty text for source '{validated_source_id}' at index {record_index}. "
                    f"Text length: {len(validated_text)}, "
                    f"Text content: '{validated_text}'. "
                    f"Empty documents are automatically excluded from chunking."
                )
                skipped_records += 1
                continue
            
            # Process text into chunks
            record_chunks_before = len(chunks)
            text_chunks = chunk_text(validated_text, chunk_size=validated_chunk_size, overlap=validated_overlap)
            
            for chunk_text_content in text_chunks:
                try:
                    chunk = TextChunk(chunk_id=chunk_id, source_id=validated_source_id, text=chunk_text_content)
                    chunks.append(chunk)
                    chunk_id += 1
                except ValueError as chunk_error:
                    logger.error(
                        f"TextChunker.chunk_records: Failed to create TextChunk. "
                        f"Chunk ID: {chunk_id}, "
                        f"Source ID: '{validated_source_id}', "
                        f"Text length: {len(chunk_text_content)}, "
                        f"Text preview: '{chunk_text_content[:100]}...'. "
                        f"Validation error: {str(chunk_error)}. "
                        f"Skipping this chunk and continuing."
                    )
                    continue
            
            record_chunks_created = len(chunks) - record_chunks_before
            total_text_length += len(validated_text)
            processed_records += 1
            
            logger.debug(
                f"TextChunker.chunk_records: Processed record {record_index} ('{validated_source_id}'). "
                f"Text length: {len(validated_text)}, "
                f"Chunks created: {record_chunks_created}, "
                f"Total chunks so far: {len(chunks)}"
            )
                
        except Exception as e:
            logger.error(
                f"TextChunker.chunk_records: Unexpected error processing record at index {record_index}. "
                f"Record: {record}, "
                f"Error type: {type(e).__name__}, "
                f"Error message: {str(e)}. "
                f"Skipping this record and continuing with next. "
                f"Suggestion: Check record format and content validity."
            )
            skipped_records += 1
            continue
    
    # Log summary of chunking operation
    logger.info(
        f"TextChunker.chunk_records: Chunking operation completed. "
        f"Input records: {len(validated_records)}, "
        f"Processed records: {processed_records}, "
        f"Skipped records: {skipped_records}, "
        f"Total chunks created: {len(chunks)}, "
        f"Total text processed: {total_text_length} characters, "
        f"Average chunks per record: {len(chunks) / max(processed_records, 1):.1f}, "
        f"Chunk size: {validated_chunk_size}, "
        f"Overlap: {validated_overlap}"
    )
            
    return chunks
