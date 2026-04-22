"""Layered document-processing architecture primitives.

This package intentionally covers only low-level extraction primitives and
mid-level composition helpers. High-level LLM orchestration belongs above this
package.
"""

from .schemas import (
    AssetRef,
    DocumentMap,
    DocumentMetadata,
    EvidenceChunk,
    ExtractedBlock,
    ExtractedDocument,
    Provenance,
    SourceInfo,
)

__all__ = [
    "AssetRef",
    "DocumentMap",
    "DocumentMetadata",
    "EvidenceChunk",
    "ExtractedBlock",
    "ExtractedDocument",
    "Provenance",
    "SourceInfo",
]
