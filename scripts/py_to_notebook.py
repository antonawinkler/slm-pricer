#!/usr/bin/env python
"""Convert Python files with cell markers (# %%) to Jupyter notebooks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_cells(content: str) -> list[dict]:
    """Parse Python content into notebook cells based on # %% markers."""
    # Split on cell markers, keeping the marker content for cell type detection
    cell_pattern = re.compile(r"^# %%(.*)$", re.MULTILINE)

    # Find all cell boundaries
    matches = list(cell_pattern.finditer(content))

    if not matches:
        # No cell markers, treat entire file as one code cell
        return [create_code_cell(content.strip())]

    cells = []

    # Handle content before first cell marker (if any)
    if matches[0].start() > 0:
        pre_content = content[: matches[0].start()].strip()
        if pre_content:
            cells.append(create_code_cell(pre_content))

    # Process each cell
    for i, match in enumerate(matches):
        # Determine cell end
        if i + 1 < len(matches):
            cell_end = matches[i + 1].start()
        else:
            cell_end = len(content)

        # Get the marker suffix (e.g., " [markdown]" or "")
        marker_suffix = match.group(1).strip()

        # Get cell content (everything after the marker line)
        cell_start = match.end() + 1  # +1 to skip the newline after marker
        cell_content = content[cell_start:cell_end]

        # Strip leading whitespace but preserve trailing newlines structure
        cell_content = cell_content.lstrip("\n").rstrip()

        if not cell_content:
            continue

        # Check if this is a markdown cell
        if marker_suffix.lower() in ("[markdown]", "[md]", "markdown", "md"):
            # For markdown cells, strip leading # from each line
            lines = cell_content.split("\n")
            md_lines = []
            for line in lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            cells.append(create_markdown_cell("\n".join(md_lines)))
        else:
            cells.append(create_code_cell(cell_content))

    return cells


def create_code_cell(source: str) -> dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n") if source else [],
    }


def create_markdown_cell(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if source else [],
    }


def format_cell_source(lines: list[str]) -> list[str]:
    """Format cell source with proper newlines for notebook format."""
    if not lines:
        return []
    # Add newline to all lines except the last
    return [line + "\n" if i < len(lines) - 1 else line for i, line in enumerate(lines)]


def create_notebook(cells: list[dict]) -> dict:
    """Create a notebook structure."""
    # Format cell sources
    for cell in cells:
        cell["source"] = format_cell_source(cell["source"])

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def convert_py_to_notebook(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert a Python file to a Jupyter notebook.

    Args:
        input_path: Path to the .py file
        output_path: Path for the output .ipynb file. If None, uses same name with .ipynb extension.

    Returns:
        Path to the created notebook
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".ipynb")
    else:
        output_path = Path(output_path)

    content = input_path.read_text(encoding="utf-8")
    cells = parse_cells(content)
    notebook = create_notebook(cells)

    output_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Python files with # %% cell markers to Jupyter notebooks"
    )
    parser.add_argument("input", type=Path, help="Input .py file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .ipynb file (default: same name with .ipynb)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    output_path = convert_py_to_notebook(args.input, args.output)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
