---
hide-toc: true
---

# Simulator

The simulator runs in this page, in your browser: nothing to install, nothing sent to a server. The first
load fetches a Python runtime and takes a few seconds.

<a class="start-button" href="simulator/index.html">Open in a full page</a>

<div class="sim-embed">
<iframe src="simulator/index.html" title="The AgMSim simulator"></iframe>
</div>

## What to try

1. Press **Run**. The testbed, a simplified Montreal on a weekday morning, plays on the map; the outcome
   measures appear under **Results**.
2. Under **Agents**, set the private cars to `L0` and run again: they no longer leave the Decarie expressway
   when it is blocked by the crash.
3. Click any line under **Decisions** to see why that agent was asked, what it saw, what it could choose and
   what it chose.
4. To let a language model decide, choose `LLM` for a kind of agent and type your own API key. The key stays
   in this browser tab and the calls go straight from your browser to the provider.

What each part of the page shows: [](quickstart/interface.md). The network and the morning it simulates:
[](testbed.md).
