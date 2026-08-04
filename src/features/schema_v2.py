"""Canonical feature contract for remediation schema version 2.0.0.

Column count alone is never accepted as evidence of schema compatibility.  Each
feature has a stable name, unit, type, domain, extraction definition, and source
aliases.  Source aliases are resolved explicitly and ambiguities fail closed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Optional


FEATURE_SCHEMA_VERSION = "2.0.0"


class FeatureSchemaError(ValueError):
    """Raised when a source cannot be mapped unambiguously to schema v2."""


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    unit: str
    kind: str
    minimum: Optional[float]
    maximum: Optional[float]
    allow_missing: bool
    formula: str
    rationale: str
    aliases: tuple[str, ...] = ()
    train_available: bool = True
    inference_available: bool = True
    backward_compatible_from: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _count(
    name: str,
    *,
    category: str = "structural",
    token: Optional[str] = None,
    rationale: str,
    aliases: tuple[str, ...] = (),
) -> FeatureSpec:
    marker = token or name
    return FeatureSpec(
        name=name,
        category=category,
        unit="occurrences",
        kind="count",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula=f"Number of canonicalized occurrences of {marker}.",
        rationale=rationale,
        aliases=aliases,
    )


BASE_FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    _count(
        "obj_count",
        token="indirect object declarations (`n g obj`)",
        rationale="Physical object declarations support consistency checks.",
        aliases=("obj", "object_declaration_count"),
    ),
    _count(
        "endobj_count",
        token="`endobj`",
        rationale="Mismatch with object declarations indicates malformed structure.",
        aliases=("endobj",),
    ),
    _count(
        "stream_count",
        token="`stream`",
        rationale="Streams carry rendered or embedded content.",
        aliases=("stream", "streams"),
    ),
    _count(
        "endstream_count",
        token="`endstream`",
        rationale="Mismatch with stream starts indicates malformed structure.",
        aliases=("endstream", "endstreams"),
    ),
    _count(
        "xref_count",
        token="xref tables",
        rationale="Cross-reference structure is frequently manipulated by evasions.",
        aliases=("xref",),
    ),
    _count(
        "trailer_count",
        token="trailers",
        rationale="Multiple trailers can indicate incremental updates.",
        aliases=("trailer",),
    ),
    _count(
        "startxref_count",
        token="`startxref`",
        rationale="Supports xref/trailer consistency checks.",
        aliases=("startxref",),
    ),
    _count(
        "js_count",
        token="PDF name `/JS`",
        rationale="Signals JavaScript-bearing actions after name canonicalization.",
        aliases=("JS", "js"),
    ),
    _count(
        "javascript_count",
        token="PDF name `/JavaScript`",
        rationale="Signals explicit JavaScript actions.",
        aliases=("JavaScript", "javascript"),
    ),
    _count(
        "action_count",
        token="PDF name `/Action`",
        rationale="Actions can trigger risky document behavior.",
        aliases=("Action", "action"),
    ),
    _count(
        "openaction_count",
        token="PDF name `/OpenAction`",
        rationale="Automatic open actions are operationally high risk.",
        aliases=("OpenAction", "open_action"),
    ),
    _count(
        "aa_count",
        token="PDF name `/AA`",
        rationale="Additional actions can trigger on document events.",
        aliases=("AA", "additional_action_count"),
    ),
    _count(
        "launch_count",
        token="PDF name `/Launch`",
        rationale="Launch actions can invoke external applications.",
        aliases=("Launch", "launch"),
    ),
    _count(
        "uri_count",
        token="PDF name `/URI`",
        rationale="URI actions can connect document interaction to external targets.",
        aliases=("URI", "uri"),
    ),
    _count(
        "submitform_count",
        token="PDF name `/SubmitForm`",
        rationale="Form submission can exfiltrate entered information.",
        aliases=("SubmitForm", "submit_form"),
    ),
    _count(
        "acroform_count",
        token="PDF name `/AcroForm`",
        rationale="Interactive forms can expose scripting surfaces.",
        aliases=("Acroform", "AcroForm", "acro_form"),
    ),
    _count(
        "xfa_count",
        token="PDF name `/XFA`",
        rationale="XFA documents expose additional scripting and parsing complexity.",
        aliases=("XFA", "xfa"),
    ),
    _count(
        "richmedia_count",
        token="PDF name `/RichMedia`",
        rationale="Rich media introduces embedded active content.",
        aliases=("RichMedia", "rich_media"),
    ),
    _count(
        "jbig2decode_count",
        token="PDF name `/JBIG2Decode` or `/JBig2Decode`",
        rationale="JBIG2 is security-relevant and historically exploit-prone.",
        aliases=("JBIG2Decode", "JBig2Decode", "jbig2"),
    ),
    _count(
        "colors_count",
        token="PDF name `/Colors`",
        rationale="Preserved for compatibility with the published PDFMal schema.",
        aliases=("Colors", "colors"),
    ),
    _count(
        "objstm_count",
        token="PDF name `/ObjStm`",
        rationale="Object streams can hide logical objects from shallow scanners.",
        aliases=("ObjStm", "object_stream_count"),
    ),
    _count(
        "filter_count",
        token="PDF name `/Filter`",
        rationale="Filter usage captures encoding and compression complexity.",
        aliases=("Filter", "filters"),
    ),
    _count(
        "obfuscation_count",
        token="PDF name `#XX` escapes",
        rationale="Name escaping is a direct static-analysis evasion technique.",
        aliases=("obfuscations", "name_obfuscation_count"),
    ),
    FeatureSpec(
        name="avg_stream_size",
        category="structural",
        unit="bytes",
        kind="continuous",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula="Mean bounded byte length of lexically identified stream bodies.",
        rationale="Distinguishes stream-heavy and unusually packed documents.",
        aliases=("average_stream_size", "stream_avg_size"),
    ),
    _count(
        "indirect_obj_count",
        token="indirect references (`n g R`)",
        rationale="Captures graph connectivity and indirection complexity.",
        aliases=("indirect_objects", "indirect_object_count"),
    ),
    FeatureSpec(
        name="pdf_size",
        category="general",
        unit="bytes",
        kind="continuous",
        minimum=0.0,
        maximum=None,
        allow_missing=False,
        formula="Exact input byte length.",
        rationale="Base size used mainly to normalize count features.",
        aliases=("pdfsize", "file_size", "PDF size"),
    ),
    FeatureSpec(
        name="title_chars",
        category="general",
        unit="characters",
        kind="count",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula="Unicode character length of the document title metadata.",
        rationale="Captures presence and richness of descriptive metadata.",
        aliases=("title_length", "title characters"),
    ),
    FeatureSpec(
        name="is_encrypted",
        category="general",
        unit="boolean",
        kind="boolean",
        minimum=0.0,
        maximum=1.0,
        allow_missing=True,
        formula="1 when the parser reports encryption, otherwise 0.",
        rationale="Encryption changes parser visibility and analysis confidence.",
        aliases=("isEncrypted", "encryption", "encrypted"),
    ),
    FeatureSpec(
        name="metadata_size",
        category="general",
        unit="bytes",
        kind="continuous",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula="Serialized metadata byte/character size under the versioned parser.",
        rationale="Supports metadata-to-document ratio and anomaly features.",
        aliases=("metadata size", "meta_size"),
    ),
    FeatureSpec(
        name="page_count",
        category="general",
        unit="pages",
        kind="count",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula="Number of pages reachable from the parsed page tree.",
        rationale="Normalizes content richness and identifies sparse documents.",
        aliases=("pages", "page number"),
    ),
    FeatureSpec(
        name="has_text",
        category="general",
        unit="boolean",
        kind="boolean",
        minimum=0.0,
        maximum=1.0,
        allow_missing=True,
        formula=(
            "1 when bounded static inspection finds font resources or an approved "
            "provider's equivalent text-presence signal; no stream is executed."
        ),
        rationale="Separates content-bearing documents from sparse carriers.",
        aliases=("contains_text", "text", "hasText"),
    ),
    _count(
        "image_count",
        category="general",
        token="parsed image XObjects",
        rationale="Supports per-page content richness features.",
        aliases=("images", "image number"),
    ),
    _count(
        "obj_count_total",
        category="general",
        token="total parsed/declared objects",
        rationale="Provides an independent object total for parser disagreement checks.",
        aliases=("object_number", "total_objects", "object count"),
    ),
    _count(
        "font_obj_count",
        category="general",
        token="font resources",
        rationale="Supports content-richness and sparse-document features.",
        aliases=("font_objects", "font count"),
    ),
    _count(
        "embedded_file_count",
        category="general",
        token="embedded file specifications",
        rationale="Embedded files are a direct document attack surface.",
        aliases=("embedded_files", "EmbeddedFile"),
    ),
    FeatureSpec(
        name="avg_embedded_media_size",
        category="general",
        unit="bytes",
        kind="continuous",
        minimum=0.0,
        maximum=None,
        allow_missing=True,
        formula="Mean declared/decoded size of bounded embedded media entries.",
        rationale="Distinguishes token presence from substantial embedded content.",
        aliases=("average_embedded_media_size", "embedded_media_avg_size"),
    ),
    FeatureSpec(
        name="header_valid",
        category="general",
        unit="boolean",
        kind="boolean",
        minimum=0.0,
        maximum=1.0,
        allow_missing=False,
        formula="1 when `%PDF-x.y` occurs in the permitted header window, else 0.",
        rationale="Header corruption and displacement are common parser-evasion signals.",
        aliases=("header", "valid_header", "header validity"),
    ),
)

BASE_FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.name for spec in BASE_FEATURE_SPECS)
FEATURE_SPEC_BY_NAME: Mapping[str, FeatureSpec] = {
    spec.name: spec for spec in BASE_FEATURE_SPECS
}

if len(BASE_FEATURE_COLUMNS) != 37 or len(set(BASE_FEATURE_COLUMNS)) != 37:
    raise RuntimeError("Schema v2 must contain exactly 37 unique base features.")


def _normalized(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def resolve_source_columns(
    columns: Iterable[str],
    *,
    overrides: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Map canonical feature names to source columns and reject ambiguity.

    ``overrides`` maps canonical name -> exact source column and is intended for
    an approved source manifest.  Automatic alias resolution is conservative.
    """
    source_columns = list(columns)
    normalized_lookup: dict[str, list[str]] = {}
    for column in source_columns:
        normalized_lookup.setdefault(_normalized(column), []).append(column)

    mapping: dict[str, str] = {}
    explicit = dict(overrides or {})
    for spec in BASE_FEATURE_SPECS:
        if spec.name in explicit:
            source = explicit[spec.name]
            if source not in source_columns:
                raise FeatureSchemaError(
                    f"Override for {spec.name!r} names absent source column {source!r}."
                )
            mapping[spec.name] = source
            continue

        candidates: set[str] = set()
        for alias in (spec.name, *spec.aliases):
            candidates.update(normalized_lookup.get(_normalized(alias), []))
        if len(candidates) == 1:
            mapping[spec.name] = next(iter(candidates))
        elif not candidates:
            raise FeatureSchemaError(
                f"No source column maps to required feature {spec.name!r}."
            )
        else:
            raise FeatureSchemaError(
                f"Ambiguous source mapping for {spec.name!r}: {sorted(candidates)}. "
                "Declare an explicit source-manifest override."
            )
    return mapping


def schema_dictionary() -> dict:
    """Return a serializable feature dictionary for documentation and manifests."""
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(BASE_FEATURE_SPECS),
        "compatibility_policy": (
            "Version 1 artifacts are incompatible by default. Compatibility requires "
            "an explicit source-column map, identical units, and a new sidecar."
        ),
        "audited_non_base_fields": {
            "xref_entry_count": "Not silently substituted for xref_count.",
            "nested_filter_count": "Not silently substituted for filter_count.",
            "indirect_object_count": "Mapped only to indirect_obj_count by explicit alias.",
            "object_endobject_counts": "Kept as separate obj_count/endobj_count fields.",
            "general_metadata": "Each field requires its own named mapping; position is ignored.",
        },
        "features": [spec.to_dict() for spec in BASE_FEATURE_SPECS],
    }
