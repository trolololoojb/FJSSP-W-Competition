from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

RESULT_ROOT_NAMES = {"results", "results_old"}


def result_hyperparameter_dir(output_dir: Path) -> Path:
    """Collapse nested run output dirs to their direct results group directory."""
    parts = output_dir.parts
    for index, part in enumerate(parts):
        if part in RESULT_ROOT_NAMES and len(parts) > index + 2:
            return Path(*parts[: index + 2])
    return output_dir


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "None"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _write_section(lines: list[str], title: str, values: Mapping[str, Any]) -> None:
    if not values:
        return
    lines.append(f"[{title}]")
    for key in sorted(values):
        lines.append(f"{key}: {_format_value(values[key])}")
    lines.append("")


def write_hyperparameters_txt(
    output_dir: Path,
    *,
    run_metadata: Mapping[str, Any] | None = None,
    ga_config: Mapping[str, Any] | None = None,
    run_config: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
) -> Path:
    """Write a concise, human-readable hyperparameter summary for a result dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = ["Hyperparameters", "===============", ""]
    if notes:
        lines.append("[notes]")
        lines.extend(notes)
        lines.append("")
    _write_section(lines, "run", run_metadata or {})
    _write_section(lines, "ga_config", ga_config or {})
    _write_section(lines, "run_config", run_config or {})

    path = output_dir / "hyperparameters.txt"
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path
