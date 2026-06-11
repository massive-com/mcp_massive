import json
import logging
import re
import ssl
from typing import Any, Literal

import certifi
import httpx
from pydantic import BaseModel, ConfigDict, Field

from .constants import _CROSS_MARKET_PATHS, _EXCLUDED_PATH_PREFIXES
from .models import Endpoint, QueryParam, ResponseAttribute

logger = logging.getLogger(__name__)


# Tag → market hints for OpenAPI paths that don't have a market-bearing
# path prefix (e.g. /v1/marketstatus/*, /v2/aggs/*, /v3/reference/*).
# Keys are OpenAPI tag prefixes; values are the markdown "## Market"
# section those endpoints belong under.  Tags like ``crypto:aggregates``
# match prefix ``crypto:`` → ``Crypto``.
_TAG_PREFIX_MARKETS: dict[str, str] = {
    "crypto": "Crypto",
    "fx": "Forex",
    "global_forex": "Forex",
    "global_crypto": "Crypto",
    "stocks": "Stocks",
    "us_stocks": "Stocks",
    "options": "Options",
    "us_options": "Options",
    "indices": "Indices",
    "futures": "Futures",
    "us_futures": "Futures",
    "fed": "Economy",
    "benzinga": "Partners",
    "tmx": "Partners",
    "etfglobal": "Partners",
    "fable": "Alternative",
    "financials": "Stocks",
}


# Format-string template for synthesizing a market when neither the
# path prefix nor the tags identify one.  Kept narrow on purpose so a
# stale OA tag doesn't quietly bucket new endpoints under the wrong
# market.
_PATH_PREFIX_MARKETS: tuple[tuple[str, str], ...] = (
    ("/crypto/", "Crypto"),
    ("/options/", "Options"),
    ("/futures/", "Futures"),
    ("/fx/", "Forex"),
    ("/stocks/", "Stocks"),
    ("/benzinga/", "Partners"),
    ("/tmx/", "Partners"),
    ("/etf-global/", "Partners"),
    ("/fed/", "Economy"),
    ("/consumer-spending/", "Alternative"),
)

# Path-placeholder name → market.  The OpenAPI spec occasionally tags
# an endpoint with the wrong market top-level (e.g. /v1/open-close/{indicesTicker}/{date}
# carries tag ``stocks:open-close`` even though the placeholder names
# the indices market).  When a placeholder is unambiguous about which
# market the endpoint serves, it wins over the tag heuristic.
_PLACEHOLDER_MARKETS: dict[str, str] = {
    "indicesticker": "Indices",
    "cryptoticker": "Crypto",
    "forexticker": "Forex",
    "fxticker": "Forex",
    "optionsticker": "Options",
    "optionticker": "Options",
}


def _market_for(path: str, tags: list[str]) -> str:
    """Derive the markdown market label for an OpenAPI GET endpoint.

    Path prefix is checked first since it's unambiguous when present
    (e.g. ``/crypto/*`` always means Crypto).  Tag prefixes provide the
    fallback for paths that don't carry a market in their prefix (the
    ``/v1/*``, ``/v2/*``, ``/v3/*`` family).  Reference-tagged paths
    (``reference:*``) default to Stocks unless overridden by a more
    specific tag — Stocks dominates the reference catalog and matches
    how the markdown organises these sections.
    """
    for prefix, market in _PATH_PREFIX_MARKETS:
        if path.startswith(prefix):
            return market
    for ph in re.findall(r"\{(\w+)\}", path):
        m = _PLACEHOLDER_MARKETS.get(ph.lower())
        if m:
            return m
    for tag in tags:
        head = tag.split(":", 1)[0]
        if head in _TAG_PREFIX_MARKETS:
            return _TAG_PREFIX_MARKETS[head]
        # ``reference:stocks:*`` and similar — second segment can name a market.
        if head == "reference" and ":" in tag:
            sub = tag.split(":", 2)[1]
            if sub in _TAG_PREFIX_MARKETS:
                return _TAG_PREFIX_MARKETS[sub]
    return "Stocks"


def _format_schema_type(schema: dict[str, Any]) -> str:
    """Return the markdown-style type label for an OpenAPI schema node.

    Booleans/integers/numbers/strings line up directly; arrays nest
    inside ``array[...]``; everything else falls back to the legacy
    ``N/A`` sentinel so type handling matches what the markdown source
    produced.
    """
    t = schema.get("type")
    if t == "array":
        items = schema.get("items", {})
        inner = _format_schema_type(items) if isinstance(items, dict) else "object"
        return f"array[{inner}]"
    if t in ("integer", "number", "boolean", "string", "object"):
        return t
    return "N/A"


def _flatten_response_schema(
    schema: dict[str, Any], prefix: str = ""
) -> list[ResponseAttribute]:
    """Walk a JSON schema into the flat ``name (type): description`` form
    used by :class:`ResponseAttribute`.

    Mirrors the markdown convention: array items are addressed via
    ``parent[]`` (so ``results`` → ``results[].close``), nested objects
    via dotted paths (``status.code``).  The walk only descends into
    ``object`` and ``array[object]`` nodes; primitive leaves emit a
    single entry, matching what the markdown tables produced.

    ``allOf``/``oneOf``/``anyOf`` composers are flattened by walking each
    branch in turn. Our spec uses ``allOf`` to assemble response
    envelopes from multiple properties (ticker + metadata + results),
    so concatenating the branches reproduces the same flat attribute
    list the markdown tables enumerates.
    """
    out: list[ResponseAttribute] = []
    if not isinstance(schema, dict):
        return out

    for key in ("allOf", "oneOf", "anyOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for sub in branches:
                if isinstance(sub, dict):
                    out.extend(_flatten_response_schema(sub, prefix))

    t = schema.get("type")
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties", {})
        for name, sub in props.items():
            if not isinstance(sub, dict):
                continue
            full = f"{prefix}.{name}" if prefix else name
            type_label = _format_schema_type(sub)
            desc = sub.get("description", "") or ""
            out.append(ResponseAttribute(name=full, type=type_label, description=desc))
            if sub.get("type") == "object":
                out.extend(_flatten_response_schema(sub, full))
            elif sub.get("type") == "array":
                items = sub.get("items", {})
                if isinstance(items, dict) and items.get("type") == "object":
                    out.extend(_flatten_response_schema(items, f"{full}[]"))
    elif t == "array":
        items = schema.get("items", {})
        if isinstance(items, dict) and items.get("type") == "object":
            out.extend(_flatten_response_schema(items, f"{prefix}[]"))
    return out


def _serialise_sample(example: Any) -> str:
    """Render an OpenAPI ``example`` value as a JSON string."""
    if example is None:
        return ""
    if isinstance(example, str):
        return example
    try:
        return json.dumps(example, indent=2, sort_keys=False)
    except (TypeError, ValueError):
        return str(example)


def _title_from_path(path: str) -> str:
    """Synthesise a human-readable title from a path when ``summary`` is missing.

    Drops the leading version segment and capitalises the final
    placeholder-free segment (e.g. ``/consumer-spending/eu/v1/merchant-aggregates``
    → ``Merchant Aggregates``).  Only used as a last-resort fallback —
    the markdown enrichment pass overrides this for any path it knows.
    """
    parts = [p for p in path.split("/") if p and not p.startswith("{")]
    if not parts:
        return path
    last = parts[-1]
    return " ".join(w.capitalize() for w in re.split(r"[-_]", last) if w) or path


# HTML fragments embedded in the spec's summary/description text.
# ``<br />`` runs are paragraph breaks; everything else (anchor tags)
# is wrapping noise around text we keep.  Stripping happens at parse
# time so neither the BM25 corpus nor the rendered search results see
# tag tokens like ``br``.
_HTML_BR_RE = re.compile(r"(?:\s*<br\s*/?>\s*)+", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Replace ``<br />`` runs with paragraph breaks and drop other tags."""
    text = _HTML_BR_RE.sub("\n\n", text)
    return _HTML_TAG_RE.sub("", text).strip()


def _clean_title(summary: str, path: str) -> str:
    """Normalise an OpenAPI ``summary`` into a display title.

    Some spec summaries are machine-generated route names rather than
    prose ("futures contracts API", "futures_snapshot_v1 API").  Strip
    the redundant " API" suffix, fall back to the path-derived title
    when what remains is a code identifier, and title-case fully
    lowercase leftovers so titles read uniformly in search results.
    """
    title = re.sub(r"\s+API\s*$", "", summary).strip()
    if not title or "_" in title:
        return _title_from_path(path)
    if title == title.lower():
        title = " ".join(w.capitalize() for w in title.split())
    return title


class OpenAPIParameter(BaseModel):
    """One parameter from an OpenAPI 3 operation.

    Only ``path`` and ``query`` parameters surface as Endpoint query
    params — header/cookie parameters exist in the spec but aren't
    part of the API the MCP server exposes.
    """

    name: str = Field(min_length=1)
    location: Literal["path", "query", "header", "cookie"] = Field(alias="in")
    # ``required`` is ``None`` when the spec omits it; OpenAPI 3
    # mandates ``required: true`` for path params so :attr:`is_required`
    # fills that in defensively.
    required: bool | None = None
    description: str = ""
    parameter_schema: dict[str, Any] = Field(default_factory=dict, alias="schema")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def is_required(self) -> bool:
        """Effective required flag.

        Path params default to ``True`` per the OpenAPI 3 spec; query
        params default to ``False`` when ``required`` is omitted.
        """
        if self.required is not None:
            return self.required
        return self.location == "path"

    def to_query_param(self) -> QueryParam | None:
        """Render this parameter as the markdown-flavour QueryParam the
        Endpoint model uses.  Returns ``None`` for header/cookie params,
        which the API layer doesn't expose to MCP callers.
        """
        if self.location not in ("path", "query"):
            return None
        return QueryParam(
            name=self.name,
            type=_format_schema_type(self.parameter_schema),
            required=self.is_required,
            description=self.description,
        )


class OpenAPIOperation(BaseModel):
    """A single HTTP operation under an OpenAPI 3 path.

    Only the fields the index actually consumes are typed — the rest
    are tolerated via ``extra="ignore"`` so the wrapper survives spec
    additions (servers, security, callbacks, etc.) without churn.
    """

    summary: str = ""
    description: str = ""
    tags: list[str] = []
    deprecated: bool = False
    parameters: list[OpenAPIParameter] = []
    # Response objects are kept untyped because the spec uses ``$ref``
    # under ``content[*].schema`` and we walk the result with the
    # legacy schema-flattener below.
    responses: dict[str, Any] = {}

    model_config = ConfigDict(extra="ignore")

    def to_query_params(self) -> list[QueryParam]:
        """Path + query params in declaration order — both share one
        namespace from the caller's perspective.
        """
        return [qp for qp in (p.to_query_param() for p in self.parameters) if qp]

    def _json_200(self) -> dict[str, Any]:
        ok = self.responses.get("200")
        if not isinstance(ok, dict):
            return {}
        content = ok.get("content", {})
        if not isinstance(content, dict):
            return {}
        json_content = content.get("application/json", {})
        return json_content if isinstance(json_content, dict) else {}

    @property
    def response_schema(self) -> dict[str, Any]:
        """The ``application/json`` schema under the 200 response.

        Returns an empty dict when the response has no schema — that's
        treated by :func:`_flatten_response_schema` as ``[]`` attrs.
        """
        schema = self._json_200().get("schema")
        return schema if isinstance(schema, dict) else {}

    @property
    def response_example(self) -> Any:
        """The ``example`` payload under the 200 response, if any."""
        return self._json_200().get("example")


class OpenAPISpec(BaseModel):
    """Validated wrapper around an OpenAPI 3 spec document.

    Construct via :meth:`from_url` (fetches and parses) or
    :meth:`from_text` (parses pre-fetched JSON).  The
    :attr:`endpoints` property returns the flat ``list[Endpoint]``
    consumed by the rest of the index module.

    The raw spec is kept on :attr:`raw` so callers that need upstream
    fields beyond what :class:`Endpoint` surfaces (servers, info,
    components/schemas) can reach in without re-fetching.
    """

    url: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_text(cls, spec_text: str, url: str | None = None) -> "OpenAPISpec":
        """Parse a JSON-encoded OpenAPI 3 document.

        A malformed document logs the parse error and yields a spec
        with an empty :attr:`raw` (so :attr:`endpoints` == ``[]``);
        callers can detect this by checking ``bool(spec.raw)``.
        """
        try:
            raw = json.loads(spec_text)
        except json.JSONDecodeError:
            logger.exception("Failed to parse OpenAPI JSON from %s", url or "<text>")
            raw = {}
        if not isinstance(raw, dict):
            raw = {}
        return cls(url=url, raw=raw)

    @classmethod
    async def from_url(
        cls,
        url: str,
        client: httpx.AsyncClient | None = None,
    ) -> "OpenAPISpec":
        """Fetch and parse an OpenAPI document from ``url``.

        Pass an existing ``client`` to share its connection pool and
        TLS context with other fetches; otherwise a short-lived
        AsyncClient is created with the project's certifi bundle.
        Network/HTTP errors propagate to the caller — the wrapper
        doesn't decide policy on failure.
        """
        if client is not None:
            resp = await client.get(url, follow_redirects=True)
            resp.raise_for_status()
            return cls.from_text(resp.text, url=url)
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        async with httpx.AsyncClient(timeout=30.0, verify=ssl_ctx) as owned:
            resp = await owned.get(url, follow_redirects=True)
            resp.raise_for_status()
            return cls.from_text(resp.text, url=url)

    @property
    def operations(self) -> list[tuple[str, OpenAPIOperation]]:
        """Validated GET operations as ``(path, op)`` pairs.

        Deprecated operations, known legacy/internal paths
        (:data:`_EXCLUDED_PATH_PREFIXES`), and entries that fail
        validation are skipped — one malformed entry doesn't block the
        rest of the index from building.
        """
        out: list[tuple[str, OpenAPIOperation]] = []
        paths = self.raw.get("paths", {})
        if not isinstance(paths, dict):
            return out
        for path, ops in paths.items():
            if not isinstance(ops, dict):
                continue
            if path.startswith(_EXCLUDED_PATH_PREFIXES):
                logger.info("Skipping excluded legacy/internal path: %s", path)
                continue
            op = ops.get("get")
            if not isinstance(op, dict):
                continue
            try:
                operation = OpenAPIOperation.model_validate(op)
            except Exception:
                logger.exception("Skipping invalid OpenAPI operation at %s", path)
                continue
            if operation.deprecated:
                continue
            out.append((path, operation))
        return out

    @property
    def endpoints(self) -> list[Endpoint]:
        """The spec's GET operations rendered as :class:`Endpoint` objects.

        Cross-market paths (see :data:`_CROSS_MARKET_PATHS`) are
        duplicated once per applicable market so the BM25 ranker can
        preserve the per-market boosts the markdown corpus used to
        provide implicitly.
        """
        out: list[Endpoint] = []
        for path, op in self.operations:
            title = _clean_title(_strip_html(op.summary), path)
            description = _strip_html(op.description)
            schema = op.response_schema
            response_attrs = _flatten_response_schema(schema) if schema else []
            sample_response = _serialise_sample(op.response_example)
            markets = _CROSS_MARKET_PATHS.get(path) or {_market_for(path, op.tags)}
            query_params = op.to_query_params()
            for market in markets:
                out.append(
                    Endpoint(
                        title=title,
                        path=path,
                        market=market,
                        description=description,
                        query_params=query_params,
                        response_attributes=response_attrs,
                        sample_response=sample_response,
                    )
                )
        return out
