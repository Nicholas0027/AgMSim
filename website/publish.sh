#!/bin/sh
# Build the project site and publish it to the repository that serves it (GitHub Pages, branch gh-pages).
#     sh website/publish.sh                                    # to https://github.com/Nicholas0027/agenticmobility
#     SITE_REPO=https://github.com/<user>/<repo>.git sh website/publish.sh           # to another repository
# The site is the pages in website/ (Sphinx, Furo theme) plus the simulator that runs in the browser
# (online/build.py), put at simulator/. Its links are relative, so it can be served from any address.
# Needs the site builder in the project's environment, once:
#     uv pip install --python .venv/bin/python -e ".[docs]"      (or: .venv/bin/pip install -e ".[docs]")
set -e
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SITE_REPO=${SITE_REPO:-https://github.com/Nicholas0027/agenticmobility.git}
NAME=$(git config user.name)
EMAIL=$(git config user.email)
HELPER=$(git config --get-all credential.https://github.com.helper | tail -1)

.venv/bin/python website/figures.py
rm -rf website/_build
if ! .venv/bin/python -c "import sphinx, furo, myst_parser" 2>/dev/null; then
    echo "the site builder is not installed in .venv; run once:"
    echo '    uv pip install --python .venv/bin/python -e ".[docs]"     (or: .venv/bin/pip install -e ".[docs]")'
    exit 1
fi
.venv/bin/python -m sphinx -q -b html website website/_build/html
.venv/bin/python online/build.py --repo "$(git remote get-url origin | sed -e 's/\.git$//')"
cp -R online/site website/_build/html/simulator
touch website/_build/html/.nojekyll

TMP=$(mktemp -d)
git -c credential.helper="$HELPER" clone -q "$SITE_REPO" "$TMP" || { echo "cannot reach $SITE_REPO (does the repository exist?)"; exit 1; }
cd "$TMP"
git checkout -q gh-pages 2>/dev/null || git checkout -q --orphan gh-pages
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp -R "$ROOT"/website/_build/html/. .
git add -A
git -c user.name="$NAME" -c user.email="$EMAIL" commit -q -m "Site built from AgMSim $(git -C "$ROOT" rev-parse --short HEAD)" \
    || echo "site unchanged"
git -c credential.helper="$HELPER" push -q origin gh-pages
cd "$ROOT"
rm -rf "$TMP"
echo "published to $SITE_REPO; GitHub Pages updates within a minute or two"
