import unittest

from src.document_pipeline.low_level import normalize_text
from src.document_pipeline.mid_level import build_doc_map, chunk_sections
from src.document_pipeline.schemas import (
    DocumentMetadata,
    ExtractedBlock,
    ExtractedDocument,
    Provenance,
    SourceInfo,
)


class DocumentPipelineArchitectureTests(unittest.TestCase):
    def test_normalize_text_preserves_content_while_cleaning_spacing(self) -> None:
        self.assertEqual(normalize_text("  Revenue\t grew\r\nby 10%.\x00  "), "Revenue grew\nby 10%.")

    def test_doc_map_and_chunks_preserve_provenance(self) -> None:
        provenance = Provenance(source_path="work/report.pptx", location_type="slide", slide=2)
        document = ExtractedDocument(
            schema_version="0.1",
            document_id="doc_report",
            source=SourceInfo(
                path="work/report.pptx",
                filename="report.pptx",
                extension=".pptx",
                mime_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                size_bytes=100,
                sha256="abc",
            ),
            metadata=DocumentMetadata(title="Report"),
            blocks=[
                ExtractedBlock(
                    block_id="blk_001",
                    document_id="doc_report",
                    type="text",
                    role="section",
                    order=0,
                    text="Revenue grew by 10%.",
                    normalized_text="Revenue grew by 10%.",
                    provenance=provenance,
                )
            ],
        )

        doc_map = build_doc_map([document])
        chunks = chunk_sections([document])

        self.assertEqual(doc_map.documents[0]["document_id"], "doc_report")
        self.assertEqual(doc_map.blocks[0]["provenance"]["slide"], 2)
        self.assertEqual(chunks[0].block_ids, ["blk_001"])
        self.assertEqual(chunks[0].provenance_refs[0].slide, 2)


if __name__ == "__main__":
    unittest.main()
