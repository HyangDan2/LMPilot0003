"""Layered document-processing architecture primitives.

This package intentionally covers only low-level extraction primitives and
mid-level composition helpers. High-level LLM orchestration belongs above this
package.
"""

from .schemas import (
    AssetRef,
    ChunkSummary,
    DocumentMap,
    DocumentMetadata,
    EvidenceChunk,
    ExtractedBlock,
    ExtractedDocument,
    LLMReportResult,
    OutputPlan,
    OutputPlanSection,
    Provenance,
    SelectedEvidence,
    SelectedEvidenceBlock,
    SourceInfo,
    SectionSummary,
)

__all__ = [
    "AssetRef",
    "ChunkSummary",
    "DocumentMap",
    "DocumentMetadata",
    "EvidenceChunk",
    "ExtractedBlock",
    "ExtractedDocument",
    "LLMReportResult",
    "OutputPlan",
    "OutputPlanSection",
    "Provenance",
    "SelectedEvidence",
    "SelectedEvidenceBlock",
    "SectionSummary",
    "SourceInfo",
]
