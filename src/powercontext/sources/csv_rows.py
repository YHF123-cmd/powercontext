from __future__ import annotations

import csv
import io
import json
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict

from powercontext.sources import (
    AdapterSourceDefinition,
    SourceAdapter,
    Source,
    SourceMaterialization,
    TextEvidence,
    TEXT_EVIDENCE_PROJECTION_KEY,
)


class CSVDocumentInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    model_config = ConfigDict(frozen=True)
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


class CSVRowSource(Source):
    model_config = ConfigDict(frozen=True)
    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    def project(self, key: str) -> dict[str, Any]:
        if key != TEXT_EVIDENCE_PROJECTION_KEY.name:
            raise ValueError(f"Unsupported projection: {key}")
        
        row_mapping = dict(zip(self.columns, self.values))
        content = json.dumps(row_mapping, ensure_ascii=False, separators=(",", ":"))
        
        return TextEvidence(
            source_type="csv-row",
            source_id=self.source_id,
            content=content,
            metadata={
                "document_id": self.document_id,
                "primary_key": self.primary_key,
                "primary_key_value": self.primary_key_value,
                "columns": list(self.columns),
                "row": row_mapping,
            },
        ).model_dump()


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    if not document_id or document_id != document_id.strip():
        raise ValueError("Invalid document_id")
    if len(primary_key_value) > 256:
        raise ValueError("Primary key value too long")
    return f"csv:{quote(document_id, safe='')}:{quote(primary_key_value, safe='')}"


class CSVRowAdapter(SourceAdapter[CSVRowInput, CSVRowSource, CSVRowInput]):
    name: str = "csv-row"
    input_class: type[CSVRowInput] = CSVRowInput
    source_class: type[CSVRowSource] = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        sid = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=sid,
            source_id=sid,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    name: str = TEXT_EVIDENCE_PROJECTION_KEY.name
    version: str = "1"
    source_class: type[CSVRowSource] = CSVRowSource
    output_class: type[TextEvidence] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        return TextEvidence.model_validate(source.project(self.name))


CSV_ROW_SOURCE_DEFINITION = AdapterSourceDefinition(
    adapter=CSVRowAdapter(),
    projections=(CSVRowTextEvidenceProjection(),),
)


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    if not value.document_id or value.document_id != value.document_id.strip():
        raise ValueError("Invalid document_id")
    
    if not value.content:
        raise ValueError("Empty file")

    try:
        text = value.content.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ValueError("Invalid UTF-8") from e

    if not text.strip():
        raise ValueError("Empty content")

    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise ValueError("Empty header")

    if not header or any(not col or not col.strip() for col in header) or len(set(header)) != len(header):
        raise ValueError("Invalid header")

    if value.primary_key not in header:
        raise ValueError("Primary key not found in header")
    
    pk_idx = header.index(value.primary_key)
    rows = []
    seen_pks = set()

    try:
        for row_data in reader:
            if not row_data or all(not c.strip() for c in row_data):
                continue
                
            if len(row_data) != len(header):
                raise ValueError("Inconsistent column count")

            pk_val = row_data[pk_idx]
            if not pk_val.strip():
                raise ValueError("Empty primary key")
            if pk_val in seen_pks:
                raise ValueError("Duplicate primary key")
            seen_pks.add(pk_val)

            rows.append(
                CSVRowInput(
                    document_id=value.document_id,
                    primary_key=value.primary_key,
                    primary_key_value=pk_val,
                    columns=tuple(header),
                    values=tuple(row_data),
                )
            )
    except csv.Error as e:
        raise ValueError(f"CSV parsing error: {e}") from e

    return tuple(rows)
