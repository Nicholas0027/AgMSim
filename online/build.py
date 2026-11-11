"""Build the online demo: a static site on which AgMSim runs in the visitor's browser.

    .venv/bin/python online/build.py --repo https://github.com/<user>/AgMSim
    .venv/bin/python -m http.server -d online/site 8770        # try it at http://localhost:8770

``online/site/`` then holds everything a static host such as GitHub Pages needs: the same page
that ``agmsim serve`` shows, two small scripts (``backend.js``, ``worker.js``) that put Pyodide
behind it, and the ``agmsim`` package as a zip. No language model can be reached from a browser
tab, so the online demo offers the built-in decision models and the offline stand-in only.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent


def build(repo: str = "") -> Path:
    site = HERE / "site"
    shutil.rmtree(site, ignore_errors=True)
    site.mkdir()
    page = (ROOT / "agmsim" / "web" / "index.html").read_text(encoding="utf-8")
    marker = "<script>\n"
    assert page.count(marker) == 1, "the page is expected to have exactly one inline script"
    (site / "index.html").write_text(page.replace(marker, '<script src="backend.js"></script>\n' + marker), encoding="utf-8")
    (site / "backend.js").write_text((HERE / "backend.js").read_text(encoding="utf-8").replace("__REPO_URL__", repo),
                                     encoding="utf-8")
    shutil.copy(HERE / "worker.js", site / "worker.js")
    (site / ".nojekyll").write_text("")
    with zipfile.ZipFile(site / "agmsim.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((ROOT / "agmsim").rglob("*")):
            if path.suffix in (".py", ".yaml") and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(ROOT).as_posix())
    return site


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="", help="address of the source repository, linked from the page")
    out = build(parser.parse_args().repo)
    print(f"built {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) // 1024} KB)")
