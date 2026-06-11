import re

from pydantic import BaseModel, Field

# Trailing "Use Cases:" sentence that the docs append to nearly every
# endpoint description.  It's marketing copy aimed at human readers
# ("Cross-border transactions, currency hedging, ...") and adds no
# signal for endpoint selection — strip it from the formatted output
# while leaving the FTS-indexed text untouched so any rare matches
# still rank.
_USE_CASES_RE = re.compile(r"\s*Use Cases?:.*\Z", re.DOTALL)

# Filter-operator suffixes that the API consistently appends to base
# query-param names (e.g. ``market_cap.gt``, ``ticker.any_of``).  Their
# descriptions are pure boilerplate — collapsing them into a per-field
# annotation in ``more`` mode is a large token win without losing
# information: the LLM still sees which fields support filtering and
# which operators are available.
_FILTER_OP_SUFFIXES: frozenset[str] = frozenset(
    {"gt", "gte", "lt", "lte", "any_of", "all_of"}
)


class QueryParam(BaseModel):
    name: str
    type: str
    required: bool
    description: str


class ResponseAttribute(BaseModel):
    name: str
    type: str
    description: str


def _collapse_filter_operators(
    params: list["QueryParam"],
) -> list[tuple["QueryParam", list[str]]]:
    """Group ``field`` and ``field.<op>`` params for compact rendering.

    Returns a list of ``(base_param, ops)`` pairs in original order,
    where ``ops`` is the list of filter-operator suffixes seen for the
    base.  Operator-suffix params that don't match a base param fall
    through unchanged (with empty ``ops``) so we never lose data.
    """
    by_name = {qp.name: qp for qp in params}
    grouped: list[tuple[QueryParam, list[str]]] = []
    seen_bases: set[str] = set()
    for qp in params:
        if "." in qp.name:
            base, suffix = qp.name.split(".", 1)
            if suffix in _FILTER_OP_SUFFIXES and base in by_name:
                # Operator variant of a base we've already emitted (or
                # will emit on its own row) — skip; we'll annotate the
                # base row with the suffix list below.
                continue
        if qp.name in seen_bases:
            continue
        seen_bases.add(qp.name)
        ops = sorted(
            {
                p.name.split(".", 1)[1]
                for p in params
                if p.name.startswith(qp.name + ".")
                and p.name.split(".", 1)[1] in _FILTER_OP_SUFFIXES
            }
        )
        grouped.append((qp, ops))
    return grouped


class Endpoint(BaseModel):
    title: str = Field(min_length=1)
    path: str
    market: str
    description: str
    query_params: list[QueryParam] = []
    response_attributes: list[ResponseAttribute] = []
    sample_response: str = ""
    # Set during indexing
    path_prefix: str = ""

    @property
    def display_description(self) -> str:
        """Description with the "Use Cases:" marketing trailer removed."""
        return _USE_CASES_RE.sub("", self.description).rstrip()

    def format(self, detail: str = "default", counter: int = 1) -> str:
        """Format this endpoint for search results at the given detail level."""
        parts = [f"{counter}. {self.title} [{self.market}]"]
        parts.append(f"   GET {self.path}")
        parts.append(f"   {self.display_description}")

        if detail == "more" and self.query_params:
            parts.append("   Query Parameters:")
            for qp, ops in _collapse_filter_operators(self.query_params):
                req = "required" if qp.required else "optional"
                ops_note = f" [filters: {', '.join(ops)}]" if ops else ""
                parts.append(
                    f"   - {qp.name} ({qp.type}, {req}){ops_note}: {qp.description}"
                )

        if detail == "verbose":
            # Verbose keeps every parameter line — callers explicitly
            # opted into full detail.
            if self.query_params:
                parts.append("   Query Parameters:")
                for qp in self.query_params:
                    req = "required" if qp.required else "optional"
                    parts.append(f"   - {qp.name} ({qp.type}, {req}): {qp.description}")
            if self.response_attributes:
                parts.append("   Response Attributes:")
                for ra in self.response_attributes:
                    parts.append(f"   - {ra.name} ({ra.type}): {ra.description}")
            if self.sample_response:
                parts.append(
                    f"   Sample Response:\n   ```json\n   {self.sample_response}\n   ```"
                )

        return "\n".join(parts)
