#!/usr/bin/env python
"""Convert Jupyter notebooks to Python files with cell markers (# %%)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def notebook_to_py(notebook: dict) -> str:
    """Convert a notebook structure to Python content with cell markers."""
    lines = []

    for i, cell in enumerate(notebook.get("cells", [])):
        cell_type = cell.get("cell_type", "code")
        source = cell.get("source", [])

        # Join source lines (they may or may not have trailing newlines)
        if isinstance(source, list):
            content = "".join(source)
        else:
            content = source

        # Remove trailing whitespace from content
        content = content.rstrip()

        if not content and cell_type == "code":
            # Skip empty code cells
            continue

        # Add cell marker
        if cell_type == "markdown":
            lines.append("# %% [markdown]")
            # Prefix each line with #
            for line in content.split("\n"):
                if line:
                    lines.append(f"# {line}")
                else:
                    lines.append("#")
        else:
            lines.append("# %%")
            lines.append(content)

        # Add blank line between cells (except after last cell)
        if i < len(notebook.get("cells", [])) - 1:
            lines.append("")
            lines.append("")

    return "\n".join(lines) + "\n"


def convert_notebook_to_py(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert a Jupyter notebook to a Python file.

    Args:
        input_path: Path to the .ipynb file
        output_path: Path for the output .py file. If None, uses same name with .py extension.

    Returns:
        Path to the created Python file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".py")
    else:
        output_path = Path(output_path)

    notebook = json.loads(input_path.read_text(encoding="utf-8"))
    content = notebook_to_py(notebook)

    output_path.write_text(content, encoding="utf-8")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Jupyter notebooks to Python files with # %% cell markers"
    )
    parser.add_argument("input", type=Path, help="Input .ipynb file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output .py file (default: same name with .py)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    output_path = convert_notebook_to_py(args.input, args.output)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
