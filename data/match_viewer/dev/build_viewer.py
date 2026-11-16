#!/usr/bin/env python3
"""
Builds the single file match_viewer.html from the module sources in viewer/.

    python dev/build_viewer.py

The modules stay as they are for the unit tests. This just strips the import/export lines and
puts everything into one <script type="module"> block inside the page template, so the page
needs nothing but itself.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)   # match_viewer/
ORDER = ["data.js", "events.js", "possession.js", "annotations.js", "queue.js", "timeline.js", "slots.js", "app.js"]


def strip_module_syntax(src):
    out = []
    for line in src.split("\n"):
        if re.match(r"^\s*import\s", line):
            continue
        line = re.sub(r"^(\s*)export\s+(const|let|function|class)\s", r"\1\2 ", line)
        out.append(line)
    return "\n".join(out)


def main():
    template = open(os.path.join(HERE, "match_viewer.template.html"), encoding="utf-8").read()
    parts = []
    for name in ORDER:
        src = open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
        parts.append(f"// ===== {name} =====\n" + strip_module_syntax(src))
    bundle = "\n\n".join(parts)
    assert "<!--VIEWER_SCRIPT-->" in template, "template has no script marker"
    page = template.replace("<!--VIEWER_SCRIPT-->", '<script type="module">\n' + bundle + "\n</script>")
    out = os.path.join(ROOT, "match_viewer.html")
    open(out, "w", encoding="utf-8").write(page)
    print(f"wrote {out}  ({len(page) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
