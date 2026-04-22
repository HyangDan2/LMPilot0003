from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Asset:
    """Metadata for a non-text item discovered in a source document."""

    asset_id: str
    kind: str
    source_file: str
    page_or_slide: int | str | None = None
    path: str = ""
    caption: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Section:
    """A normalized section-like unit extracted from a source document."""

    section_id: str
    title: str
    level: int = 1
    text: str = ""
    page_or_slide: int | str | None = None
    assets: list[Asset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["assets"] = [asset.to_dict() for asset in self.assets]
        return payload


@dataclass(frozen=True)
class ParsedDocument:
    """Common JSON-ready representation for all supported input formats."""

    doc_id: str
    file_path: str
    file_type: str
    title: str
    text: str
    sections: list[Section] = field(default_factory=list)
    assets: list[Asset] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [section.to_dict() for section in self.sections]
        payload["assets"] = [asset.to_dict() for asset in self.assets]
        return payload


@dataclass(frozen=True)
class SlidePlan:
    """A single planned slide returned by the LLM planner."""

    slide_title: str
    purpose: str
    source_refs: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PresentationPlan:
    """Validated presentation plan used by the deterministic PPTX renderer."""

    output_type: str
    title: str
    target_audience: str
    slides: list[SlidePlan]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["slides"] = [slide.to_dict() for slide in self.slides]
        return payload

