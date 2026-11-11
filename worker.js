// AgMSim in the browser. Pyodide (Python compiled to WebAssembly) runs the simulator in this worker;
// the page talks to it exactly as it would talk to `agmsim serve` (see agmsim/browser.py).
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.29.5/full/";
importScripts(PYODIDE + "pyodide.js");

let request, advance, pumping = false;
const ready = (async () => {
  const py = await loadPyodide({indexURL: PYODIDE});
  await py.loadPackage("pyyaml");
  const zip = await (await fetch("agmsim.zip")).arrayBuffer();
  py.unpackArchive(zip, "zip");                       // into the working directory, which is on sys.path
  const browser = py.pyimport("agmsim.browser");
  request = browser.request;
  advance = browser.advance;
})();
ready.then(() => postMessage({ready: true}), err => postMessage({failed: String(err)}));

// A run is moved forward in short slices so that the page's requests are answered in between.
async function pump() {
  if (pumping) return;
  pumping = true;
  try {
    while (advance(0.04)) await new Promise(resolve => setTimeout(resolve, 0));
  } finally {
    pumping = false;
  }
}

onmessage = async event => {
  const {id, path, body} = event.data;
  try {
    await ready;
    // (a GET passes no body at all: recent Pyodide turns a JavaScript null into its own object, not None)
    const text = body === undefined ? request(path) : request(path, JSON.stringify(body));
    postMessage({id, text});
    if (body !== undefined) pump();
  } catch (err) {
    postMessage({id, error: String(err)});
  }
};
