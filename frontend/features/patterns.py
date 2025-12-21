from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class DiscoveredRule:
    parameter: str
    condition: str
    occurrence_pct: float
    avg_return_with: float
    avg_return_without: float
    confidence: float
    description: str
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    rule_id: Optional[int] = None


def get_rule_summary(rules: List[DiscoveredRule]) -> str:
    """Generate a human-readable summary of discovered rules."""
    if not rules:
        return "No rules discovered yet. Run more discovery tests to find patterns."

    lines = [
        "=" * 60,
        "DISCOVERED WINNING PATTERNS",
        "=" * 60,
        "",
    ]

    high_conf = [r for r in rules if r.confidence >= 0.7]
    med_conf = [r for r in rules if 0.4 <= r.confidence < 0.7]
    low_conf = [r for r in rules if r.confidence < 0.4]

    if high_conf:
        lines.append("HIGH CONFIDENCE PATTERNS:")
        lines.append("-" * 40)
        for rule in high_conf[:5]:
            lines.append(f"  • {rule.description}")
            lines.append(f"    Confidence: {rule.confidence:.0%}")
        lines.append("")

    if med_conf:
        lines.append("MEDIUM CONFIDENCE PATTERNS:")
        lines.append("-" * 40)
        for rule in med_conf[:5]:
            lines.append(f"  • {rule.description}")
        lines.append("")

    lines.append("=" * 60)
    lines.append(f"Total rules discovered: {len(rules)}")
    lines.append(f"High confidence: {len(high_conf)}")
    lines.append(f"Medium confidence: {len(med_conf)}")
    lines.append(f"Low confidence: {len(low_conf)}")

    return "\n".join(lines)
