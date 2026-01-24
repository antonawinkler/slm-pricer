"""Tests for notebook <-> Python file conversion round-trips."""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

from notebook_to_py import convert_notebook_to_py, notebook_to_py
from py_to_notebook import convert_py_to_notebook, parse_cells


class TestPyToNotebook:
    """Tests for Python to notebook conversion."""

    def test_single_code_cell(self) -> None:
        content = "# %%\nprint('hello')"
        cells = parse_cells(content)
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "code"
        assert "print('hello')" in cells[0]["source"]

    def test_multiple_code_cells(self) -> None:
        content = "# %%\nprint('a')\n\n# %%\nprint('b')"
        cells = parse_cells(content)
        assert len(cells) == 2
        assert all(c["cell_type"] == "code" for c in cells)

    def test_markdown_cell(self) -> None:
        content = "# %% [markdown]\n# This is markdown\n# With multiple lines"
        cells = parse_cells(content)
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "markdown"
        assert "This is markdown" in cells[0]["source"]

    def test_mixed_cells(self) -> None:
        content = """# %%
import numpy as np

# %% [markdown]
# Documentation here

# %%
print('code')"""
        cells = parse_cells(content)
        assert len(cells) == 3
        assert cells[0]["cell_type"] == "code"
        assert cells[1]["cell_type"] == "markdown"
        assert cells[2]["cell_type"] == "code"

    def test_empty_content_before_marker(self) -> None:
        content = "\n\n# %%\nprint('hello')"
        cells = parse_cells(content)
        assert len(cells) == 1

    def test_content_before_first_marker(self) -> None:
        content = "# Some header comment\n\n# %%\nprint('hello')"
        cells = parse_cells(content)
        assert len(cells) == 2  # Pre-marker content becomes a cell


class TestNotebookToPy:
    """Tests for notebook to Python conversion."""

    def test_single_code_cell(self) -> None:
        notebook: dict[str, Any] = {
            "cells": [{"cell_type": "code", "source": ["print('hello')"]}]
        }
        result = notebook_to_py(notebook)
        assert "# %%" in result
        assert "print('hello')" in result

    def test_markdown_cell(self) -> None:
        notebook: dict[str, Any] = {
            "cells": [{"cell_type": "markdown", "source": ["This is markdown"]}]
        }
        result = notebook_to_py(notebook)
        assert "# %% [markdown]" in result
        assert "# This is markdown" in result

    def test_empty_code_cell_skipped(self) -> None:
        notebook: dict[str, Any] = {
            "cells": [
                {"cell_type": "code", "source": []},
                {"cell_type": "code", "source": ["print('hello')"]},
            ]
        }
        result = notebook_to_py(notebook)
        # Should only have one cell marker
        assert result.count("# %%") == 1

    def test_source_as_list_with_newlines(self) -> None:
        notebook: dict[str, Any] = {
            "cells": [{"cell_type": "code", "source": ["print('a')\n", "print('b')"]}]
        }
        result = notebook_to_py(notebook)
        assert "print('a')" in result
        assert "print('b')" in result


class TestRoundTrip:
    """Tests that conversion round-trips preserve content."""

    def normalize_py_content(self, content: str) -> str:
        """Normalize Python content for comparison.

        Collapses multiple consecutive blank lines into a single blank line,
        since the exact number of blank lines between cells isn't preserved.
        """
        # Remove trailing whitespace from lines
        lines = [line.rstrip() for line in content.strip().split("\n")]
        result = "\n".join(lines)
        # Collapse multiple blank lines into one
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result

    def test_py_to_notebook_to_py(self) -> None:
        """Test: .py -> .ipynb -> .py gives equivalent content."""
        original_py = """# %%
import numpy as np
import pandas as pd

# %%
x = np.array([1, 2, 3])
print(x)

# %% [markdown]
# This is a markdown cell
# With multiple lines

# %%
def foo():
    return 42

result = foo()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write original .py
            py_path = tmppath / "test.py"
            py_path.write_text(original_py, encoding="utf-8")

            # Convert to notebook
            notebook_path = convert_py_to_notebook(py_path)

            # Convert back to .py
            final_py_path = tmppath / "test_final.py"
            convert_notebook_to_py(notebook_path, final_py_path)

            # Compare
            final_content = final_py_path.read_text(encoding="utf-8")

            original_normalized = self.normalize_py_content(original_py)
            final_normalized = self.normalize_py_content(final_content)

            assert original_normalized == final_normalized

    def test_notebook_to_py_to_notebook(self) -> None:
        """Test: .ipynb -> .py -> .ipynb gives equivalent structure."""
        original_notebook: dict[str, Any] = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["import numpy as np\n", "import pandas as pd"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["x = np.array([1, 2, 3])\n", "print(x)"],
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["This is a markdown cell\n", "With multiple lines"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "def foo():\n",
                        "    return 42\n",
                        "\n",
                        "result = foo()",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                }
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Write original notebook
            notebook_path = tmppath / "test.ipynb"
            notebook_path.write_text(json.dumps(original_notebook), encoding="utf-8")

            # Convert to .py
            py_path = convert_notebook_to_py(notebook_path)

            # Convert back to notebook
            final_notebook_path = tmppath / "test_final.ipynb"
            convert_py_to_notebook(py_path, final_notebook_path)

            # Load and compare
            final_notebook: dict[str, Any] = json.loads(
                final_notebook_path.read_text(encoding="utf-8")
            )

            # Compare cell count and types
            assert len(final_notebook["cells"]) == len(original_notebook["cells"])

            for orig_cell, final_cell in zip(
                original_notebook["cells"], final_notebook["cells"]
            ):
                assert orig_cell["cell_type"] == final_cell["cell_type"]

                # Compare source content (normalize whitespace)
                orig_source = "".join(orig_cell["source"]).strip()
                final_source = "".join(final_cell["source"]).strip()
                assert orig_source == final_source

    def test_real_file_round_trip(self) -> None:
        """Test round-trip on a real file if it exists."""
        real_file = Path(
            "c:/Users/anton/projects/slm-pricer/notebooks/06_end_to_end.py"
        )
        if not real_file.exists():
            pytest.skip("Real test file not found")

        original_content = real_file.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Copy original
            py_path = tmppath / "test.py"
            py_path.write_text(original_content, encoding="utf-8")

            # Round trip: .py -> .ipynb -> .py
            notebook_path = convert_py_to_notebook(py_path)
            final_py_path = tmppath / "test_final.py"
            convert_notebook_to_py(notebook_path, final_py_path)

            final_content = final_py_path.read_text(encoding="utf-8")

            # Normalize and compare
            original_normalized = self.normalize_py_content(original_content)
            final_normalized = self.normalize_py_content(final_content)

            assert original_normalized == final_normalized


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_markdown_lines(self) -> None:
        """Test markdown with empty lines."""
        content = """# %% [markdown]
# Title
#
# Paragraph after empty line"""
        cells = parse_cells(content)
        assert len(cells) == 1
        assert cells[0]["cell_type"] == "markdown"
        source = "\n".join(cells[0]["source"])
        assert "Title" in source
        assert "Paragraph after empty line" in source

    def test_multiline_strings(self) -> None:
        """Test code with multiline strings."""
        content = '''# %%
text = """
This is a
multiline string
"""
print(text)'''
        cells = parse_cells(content)
        assert len(cells) == 1
        source = "\n".join(cells[0]["source"])
        assert 'text = """' in source
        assert "multiline string" in source

    def test_comments_in_code(self) -> None:
        """Test that regular comments aren't treated as cell markers."""
        content = """# %%
# This is a regular comment
x = 1  # inline comment
# Another comment
print(x)"""
        cells = parse_cells(content)
        assert len(cells) == 1
        source = "\n".join(cells[0]["source"])
        assert "# This is a regular comment" in source
        assert "# inline comment" in source

    def test_cell_marker_variations(self) -> None:
        """Test different markdown marker styles."""
        for marker in ["# %% [markdown]", "# %% [md]", "# %% markdown", "# %% md"]:
            content = f"{marker}\n# Test content"
            cells = parse_cells(content)
            assert cells[0]["cell_type"] == "markdown", f"Failed for marker: {marker}"

    def test_indented_code(self) -> None:
        """Test that indentation is preserved."""
        content = """# %%
def foo():
    if True:
        for i in range(10):
            print(i)"""
        cells = parse_cells(content)
        source = "\n".join(cells[0]["source"])
        assert "    if True:" in source
        assert "        for i in range(10):" in source
        assert "            print(i)" in source
