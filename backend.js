// Loaded only by the online demo, before the page's own script: the page's API calls are answered
// by a web worker that runs the simulator (worker.js) instead of by a server.
(() => {
  const worker = new Worker("worker.js"), waiting = new Map();
  const status = document.getElementById("status");
  let next = 1;
  status.textContent = "Loading the simulator into your browser (a few seconds, once per visit)…";
  worker.onmessage = event => {
    const m = event.data;
    if (m.ready) { status.textContent = "Ready. Runs happen in your browser; nothing is sent to a server."; return; }
    if (m.failed) { status.textContent = "Could not load the simulator: " + m.failed; status.className = "err"; return; }
    const call = waiting.get(m.id);
    waiting.delete(m.id);
    if (m.error) call.reject(new Error(m.error)); else call.resolve(JSON.parse(m.text));
  };
  window.agmsimBackend = (path, body) => new Promise((resolve, reject) => {
    waiting.set(next, {resolve, reject});
    worker.postMessage({id: next++, path, body});
  });
  const repo = "https://github.com/Nicholas0027/AgMSim";
  if (repo.startsWith("http")) {
    const link = document.createElement("a");
    link.href = repo; link.textContent = "Source and local install on GitHub"; link.style.marginLeft = "auto";
    document.querySelector("header").appendChild(link);
  }
})();
