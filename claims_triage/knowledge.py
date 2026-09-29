"""Knowledge base: underwriting rules, coverage rules and fraud indicators.

THE KEY IDEA
    The markdown files in knowledge/ are the single source of truth for business rules. Each rule
    looks like this:

        ### UW-01 · Early-inception claims
        - **Condition:** `days_since_inception <= 7`      <- machine-checkable expression over claim facts
        - **Action:** route_to_investigator                <- what to recommend if it fires
        - **Severity:** high
        Free-text rationale that the LLM reads...

    The same files are used in two ways:
      1. LIVE MODE: uploaded to a Foundry vector store. The Anomaly & Coverage agent retrieves them with
         file_search (this is the "knowledge grounding" requirement).
      2. HERE: parsed into Rule objects so the *deterministic verifier* (guardrails.py) can re-check
         every rule against the facts and catch a model that skipped or invented a rule. In mock mode
         the same evaluator also stands in for the model's reasoning.

    Changing a threshold in the markdown changes behaviour in both paths, with no code change needed.

WHAT'S IN THIS FILE
    * Rule dataclass + load_rules()           parse the markdown files
    * evaluate_condition() / evaluate_rules() a SAFE mini-interpreter for rule conditions (no eval!)
    * most_severe() / action_rank()           combine several rule actions into one recommendation
    * LocalKnowledgeSearch                    keyword search used only by the mock (stand-in for file_search)
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Severity ordering of actions. Higher number = more cautious. "note" is informational and never
# escalates a claim on its own.
ACTION_RANK = {"auto_approve": 0, "note": 0, "request_documentation": 1, "route_to_investigator": 2}
# The only three recommendations the system may give a claim (from the assignment brief).
FINAL_ACTIONS = ("auto_approve", "request_documentation", "route_to_investigator")


@dataclass(frozen=True)
class Rule:
    rule_id: str               # e.g. "UW-01"
    title: str                 # e.g. "Early-inception claims"
    condition: Optional[str]   # e.g. "days_since_inception <= 7"; None = judgement-only rule
    action: str                # note | auto_approve | request_documentation | route_to_investigator
    severity: str              # info | low | medium | high
    text: str                  # full markdown body (what the LLM reads)
    source: str                # file name the rule came from

    @property
    def machine_checkable(self) -> bool:
        """Rules without a Condition (e.g. FI-06) can only be applied by LLM judgement."""
        return self.condition is not None


# Regexes that pick the pieces out of a rule's markdown.
_HEADER = re.compile(r"^###\s+([A-Z]{2,3}-\d{2})\s*[·\-—:]\s*(.+?)\s*$")   # "### UW-01 · Title"
_COND = re.compile(r"\*\*Condition:\*\*\s*`([^`]+)`")
_ACTION = re.compile(r"\*\*Action:\*\*\s*([a-z_]+)")
_SEVERITY = re.compile(r"\*\*Severity:\*\*\s*([a-z]+)")


def load_rules(knowledge_dir: Path) -> List[Rule]:
    """Parse every knowledge/*.md file into Rule objects (one per '### ID · Title' section)."""
    rules: List[Rule] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        current: Optional[List[str]] = None   # body lines of the rule being read
        header = None                         # (rule_id, title) of the rule being read
        # A sentinel header at the end flushes the last real rule.
        for line in path.read_text(encoding="utf-8").splitlines() + ["### END-00 · sentinel"]:
            m = _HEADER.match(line)
            if m:
                if header and current is not None:
                    rules.append(_build_rule(header, current, path.name))
                header, current = (m.group(1), m.group(2)), []
            elif current is not None:
                current.append(line)
    return [r for r in rules if r.rule_id != "END-00"]


def _build_rule(header, body_lines: List[str], source: str) -> Rule:
    """Turn a header + its body lines into a Rule (missing Action defaults to 'note')."""
    body = "\n".join(body_lines).strip()
    cond = _COND.search(body)
    action = _ACTION.search(body)
    sev = _SEVERITY.search(body)
    return Rule(
        rule_id=header[0],
        title=header[1],
        condition=cond.group(1).strip() if cond else None,
        action=action.group(1) if action else "note",
        severity=sev.group(1) if sev else "low",
        text=body,
        source=source,
    )


# ==================================================================================================
# Safe condition evaluator
#
# Rule conditions look like Python ("claim_amount > 10000 and "flood" in description"), but we
# NEVER call eval() on them. Knowledge files are business content and must not be able to run code.
# Instead we parse the expression into an AST and walk it ourselves, allowing only:
#   comparisons (== != < <= > >= in, not in), and/or/not, + - * / %, numbers/strings/lists,
#   and names that exist in the facts dict.
# Anything else (function calls, attribute access, imports...) raises UnsafeExpression.
# ==================================================================================================
class UnsafeExpression(ValueError):
    pass


# Whitelisted comparison and arithmetic operators -> the Python function that implements them.
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod}


@lru_cache(maxsize=256)
def _parse(expr: str) -> ast.Expression:
    """Parse once and cache: the same few rule conditions are evaluated for every claim."""
    return ast.parse(expr, mode="eval")


def _eval(node: ast.AST, facts: Dict[str, Any]) -> Any:
    """Recursively evaluate one AST node against the facts dict."""
    if isinstance(node, ast.Expression):
        return _eval(node.body, facts)
    if isinstance(node, ast.BoolOp):                       # a and b / a or b
        if isinstance(node.op, ast.And):
            return all(_eval(v, facts) for v in node.values)
        return any(_eval(v, facts) for v in node.values)
    if isinstance(node, ast.UnaryOp):                      # not x / -x
        val = _eval(node.operand, facts)
        if isinstance(node.op, ast.Not):
            return not val
        if isinstance(node.op, ast.USub):
            return -val
        raise UnsafeExpression(type(node.op).__name__)
    if isinstance(node, ast.Compare):                      # a < b, a in b, chained a < b <= c
        left = _eval(node.left, facts)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, facts)
            fn = _CMP.get(type(op))
            if fn is None:
                raise UnsafeExpression(type(op).__name__)
            if left is None or right is None:
                if isinstance(op, (ast.Eq, ast.NotEq)):
                    pass  # comparing with None is well defined for ==/!=
                else:
                    # A needed fact is missing (e.g. no loss date): signal "can't evaluate".
                    raise TypeError("missing fact")
            if not fn(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BinOp):                        # arithmetic, e.g. claim_amount % 5000
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise UnsafeExpression(type(node.op).__name__)
        return fn(_eval(node.left, facts), _eval(node.right, facts))
    if isinstance(node, ast.Name):                         # a fact name, e.g. days_since_inception
        if node.id not in facts:
            raise UnsafeExpression(f"unknown fact '{node.id}'")
        return facts[node.id]
    if isinstance(node, ast.Constant):                     # literals: 7, "flood", True
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):            # ["fire", "theft", ...]
        return [_eval(e, facts) for e in node.elts]
    # Everything else (calls, attributes, subscripts, lambdas...) is forbidden.
    raise UnsafeExpression(type(node).__name__)


def evaluate_condition(expr: str, facts: Dict[str, Any]) -> Optional[bool]:
    """True/False, or None if a needed fact is missing (rule can't be evaluated for this claim)."""
    try:
        return bool(_eval(_parse(expr), facts))
    except TypeError:
        return None


def facts_referenced(expr: str) -> List[str]:
    """Names used in a condition, used to print evidence like 'days_since_inception=3'."""
    return sorted({n.id for n in ast.walk(_parse(expr)) if isinstance(n, ast.Name)})


def evaluate_rules(rules: Iterable[Rule], facts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every machine-checkable rule whose condition is TRUE for these facts, with evidence."""
    fired = []
    for rule in rules:
        if not rule.machine_checkable:
            continue
        if evaluate_condition(rule.condition, facts):
            # Evidence = the condition + the actual values that made it true (description omitted: too long).
            evidence = ", ".join(f"{name}={facts.get(name)!r}" for name in facts_referenced(rule.condition)
                                 if name not in ("description",))
            fired.append({
                "rule_id": rule.rule_id,
                "title": rule.title,
                "action": rule.action,
                "severity": rule.severity,
                "evidence": f"{rule.condition}  [{evidence}]",
                "source": rule.source,
            })
    return fired


def most_severe(actions: Iterable[str]) -> str:
    """Combine rule actions into one final recommendation (most cautious wins; default auto_approve)."""
    best = "auto_approve"
    for a in actions:
        if a in ACTION_RANK and ACTION_RANK[a] > ACTION_RANK[best]:
            best = a
    return best


def action_rank(action: str) -> int:
    """Numeric severity of an action (unknown actions count as 0)."""
    return ACTION_RANK.get(action, 0)


# ==================================================================================================
# Local keyword retrieval. Used ONLY in mock mode as a stand-in for Foundry file_search.
# It scores each rule by how many query words appear in it. Crude, but enough for a demo.
# ==================================================================================================
_WORD = re.compile(r"[a-z0-9_]+")
_STOP = {"the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "is", "are", "with", "claim", "claims", "rules", "rule"}


class LocalKnowledgeSearch:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
        # Pre-tokenise each rule once (ID + title + condition + body text).
        self._index = [(r, self._tokens(f"{r.rule_id} {r.title} {r.condition or ''} {r.text}")) for r in rules]

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [t for t in _WORD.findall(text.lower()) if t not in _STOP]

    def search(self, query: str, k: int = 5) -> List[Rule]:
        """Return up to k rules with the most query-word matches (ties broken by rule ID)."""
        q = set(self._tokens(query))
        scored = []
        for rule, toks in self._index:
            score = sum(1 for t in toks if t in q)
            if score:
                scored.append((score, rule.rule_id, rule))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [r for _, _, r in scored[:k]]
