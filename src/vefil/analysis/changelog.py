"""Changelog tracking for veFIL simulation reproducibility.

Tracks configuration changes and simulation runs tied to config hashes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChangelogEntry:
    """A single changelog entry."""
    timestamp: str
    config_hash: str
    action: str  # "run", "config_change", "parameter_update"
    description: str
    details: Optional[Dict[str, Any]] = None
    metrics_snapshot: Optional[Dict[str, float]] = None


@dataclass
class ConfigDiff:
    """Difference between two configurations."""
    param_path: str
    old_value: Any
    new_value: Any
    impact: str  # "high", "medium", "low"


class ChangelogManager:
    """Manage simulation changelog for reproducibility."""

    def __init__(self, max_entries: int = 100):
        """
        Initialize changelog manager.

        Args:
            max_entries: Maximum number of entries to keep
        """
        self.entries: List[ChangelogEntry] = []
        self.max_entries = max_entries
        self._last_config_hash: Optional[str] = None

    def log_run(
        self,
        config_hash: str,
        metrics: Dict[str, Any],
        description: str = "Simulation run"
    ) -> ChangelogEntry:
        """
        Log a simulation run.

        Args:
            config_hash: Hash of the configuration used
            metrics: Final metrics from the run
            description: Optional description

        Returns:
            The created entry
        """
        entry = ChangelogEntry(
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            action="run",
            description=description,
            metrics_snapshot={
                'final_locked': metrics.get('final_locked', 0),
                'final_reserve': metrics.get('final_reserve', 0),
                'reserve_runway_years': metrics.get('reserve_runway_years', 0),
            }
        )

        self._add_entry(entry)
        self._last_config_hash = config_hash
        return entry

    def log_config_change(
        self,
        old_hash: str,
        new_hash: str,
        diffs: List[ConfigDiff],
        description: str = "Configuration updated"
    ) -> ChangelogEntry:
        """
        Log a configuration change.

        Args:
            old_hash: Previous config hash
            new_hash: New config hash
            diffs: List of parameter differences
            description: Optional description

        Returns:
            The created entry
        """
        details = {
            'previous_hash': old_hash,
            'changes': [
                {
                    'param': d.param_path,
                    'from': d.old_value,
                    'to': d.new_value,
                    'impact': d.impact
                }
                for d in diffs
            ]
        }

        entry = ChangelogEntry(
            timestamp=datetime.now().isoformat(),
            config_hash=new_hash,
            action="config_change",
            description=description,
            details=details
        )

        self._add_entry(entry)
        self._last_config_hash = new_hash
        return entry

    def log_parameter_update(
        self,
        config_hash: str,
        param_path: str,
        old_value: Any,
        new_value: Any
    ) -> ChangelogEntry:
        """
        Log a single parameter update.

        Args:
            config_hash: Current config hash
            param_path: Parameter path (e.g., "yield_source.reserve_annual_rate")
            old_value: Previous value
            new_value: New value

        Returns:
            The created entry
        """
        entry = ChangelogEntry(
            timestamp=datetime.now().isoformat(),
            config_hash=config_hash,
            action="parameter_update",
            description=f"Updated {param_path}",
            details={
                'parameter': param_path,
                'from': old_value,
                'to': new_value
            }
        )

        self._add_entry(entry)
        return entry

    def get_entries_for_hash(self, config_hash: str) -> List[ChangelogEntry]:
        """Get all entries for a specific config hash."""
        return [e for e in self.entries if e.config_hash == config_hash]

    def get_recent_entries(self, n: int = 10) -> List[ChangelogEntry]:
        """Get the n most recent entries."""
        return self.entries[-n:] if self.entries else []

    def get_runs_summary(self) -> Dict[str, Any]:
        """Get summary of all runs grouped by config hash."""
        runs_by_hash: Dict[str, List[ChangelogEntry]] = {}

        for entry in self.entries:
            if entry.action == "run":
                if entry.config_hash not in runs_by_hash:
                    runs_by_hash[entry.config_hash] = []
                runs_by_hash[entry.config_hash].append(entry)

        return {
            'total_runs': sum(len(runs) for runs in runs_by_hash.values()),
            'unique_configs': len(runs_by_hash),
            'configs': {
                hash_prefix: {
                    'count': len(runs),
                    'first_run': runs[0].timestamp,
                    'last_run': runs[-1].timestamp,
                    'latest_metrics': runs[-1].metrics_snapshot
                }
                for hash_prefix, runs in runs_by_hash.items()
            }
        }

    def _add_entry(self, entry: ChangelogEntry) -> None:
        """Add entry and trim if needed."""
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize changelog to dict."""
        return {
            'entries': [
                {
                    'timestamp': e.timestamp,
                    'config_hash': e.config_hash,
                    'action': e.action,
                    'description': e.description,
                    'details': e.details,
                    'metrics_snapshot': e.metrics_snapshot
                }
                for e in self.entries
            ],
            'last_config_hash': self._last_config_hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangelogManager':
        """Deserialize changelog from dict."""
        manager = cls()
        for entry_data in data.get('entries', []):
            entry = ChangelogEntry(
                timestamp=entry_data['timestamp'],
                config_hash=entry_data['config_hash'],
                action=entry_data['action'],
                description=entry_data['description'],
                details=entry_data.get('details'),
                metrics_snapshot=entry_data.get('metrics_snapshot')
            )
            manager.entries.append(entry)
        manager._last_config_hash = data.get('last_config_hash')
        return manager


def compare_configs(old_config, new_config) -> List[ConfigDiff]:
    """
    Compare two configurations and return differences.

    Args:
        old_config: Previous Config object
        new_config: New Config object

    Returns:
        List of ConfigDiff objects
    """
    diffs = []

    # High-impact parameters
    high_impact_params = [
        'yield_source.type',
        'yield_source.reserve_annual_rate',
        'reward_curve.k',
        'reward_curve.max_duration_years',
        'simulation.participation_elasticity',
    ]

    # Medium-impact parameters
    medium_impact_params = [
        'yield_source.fee_multiplier',
        'yield_source.hybrid_reserve_ratio',
        'capital_flow.net_new_fraction',
        'market.market_multiplier',
        'market.volatility',
    ]

    def get_nested_value(obj, path: str):
        """Get value from object using dot notation."""
        parts = path.split('.')
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj

    # Check all known parameters
    all_params = high_impact_params + medium_impact_params + [
        'initial_supply.total',
        'initial_supply.circulating',
        'initial_supply.reserve',
        'alternatives.ifil_apy',
        'alternatives.glif_apy',
        'alternatives.defi_apy',
        'cohorts.retail.required_premium',
        'cohorts.institutional.required_premium',
        'cohorts.storage_providers.required_premium',
        'cohorts.treasuries.required_premium',
        'simulation.time_horizon_months',
        'simulation.bootstrap_apy',
        'simulation.adoption_ramp_fraction',
    ]

    for param in all_params:
        old_val = get_nested_value(old_config, param)
        new_val = get_nested_value(new_config, param)

        if old_val != new_val:
            if param in high_impact_params:
                impact = "high"
            elif param in medium_impact_params:
                impact = "medium"
            else:
                impact = "low"

            diffs.append(ConfigDiff(
                param_path=param,
                old_value=old_val,
                new_value=new_val,
                impact=impact
            ))

    return diffs


def format_changelog_markdown(manager: ChangelogManager, max_entries: int = 20) -> str:
    """
    Format changelog as markdown.

    Args:
        manager: ChangelogManager instance
        max_entries: Maximum entries to show

    Returns:
        Markdown formatted string
    """
    lines = ["# Simulation Changelog\n"]

    summary = manager.get_runs_summary()
    lines.append("## Summary\n")
    lines.append(f"- **Total runs**: {summary['total_runs']}")
    lines.append(f"- **Unique configurations**: {summary['unique_configs']}\n")

    recent = manager.get_recent_entries(max_entries)
    if not recent:
        lines.append("*No entries yet.*")
        return "\n".join(lines)

    lines.append("## Recent Activity\n")
    lines.append("| Timestamp | Action | Config Hash | Description |")
    lines.append("|-----------|--------|-------------|-------------|")

    for entry in reversed(recent):
        ts = entry.timestamp[:19]  # Trim microseconds
        hash_short = entry.config_hash[:8]
        action_icon = {
            'run': '▶️',
            'config_change': '🔄',
            'parameter_update': '✏️'
        }.get(entry.action, '•')

        lines.append(f"| {ts} | {action_icon} {entry.action} | `{hash_short}` | {entry.description} |")

    return "\n".join(lines)
