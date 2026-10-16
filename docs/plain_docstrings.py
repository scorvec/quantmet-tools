"""Griffe extension for the documentation build: every docstring is shown exactly as written, as preformatted text.

quantmet's docstrings are written for ``help()`` and editors: aligned ``name : meaning`` lists and indented formulas
whose layout carries the meaning. Rendered as Markdown they would reflow into run-on paragraphs.
"""

from __future__ import annotations

from typing import Any

import griffe


class PlainDocstrings(griffe.Extension):
    def on_object(self, *, obj: griffe.Object, **kwargs: Any) -> None:
        if obj.docstring is not None and obj.docstring.value.strip():
            obj.docstring.value = f"```text\n{obj.docstring.value}\n```"
