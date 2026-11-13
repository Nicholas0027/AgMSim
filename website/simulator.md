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

1. Under **Agents**, click **built-in for all** and press **Run**. The testbed, a simplified Montreal on a weekday
   morning, plays on the map within seconds; the outcome measures appear under **Results**.
2. Under **Agents**, set the private cars to `L0` and run again: they no longer leave the Decarie expressway
   when it is blocked by the crash.
3. Click any line under **Decisions** to see why that agent was asked, what it saw, what it could choose and
   what it chose.
4. The page opens with a language model (DeepSeek Flash) deciding for travellers, cars, charging stations and the
   orchestrator. To run it that way, type your own API key under **Language model**: the key stays in this browser
   tab and the calls go straight from your browser to the provider. Such a run takes about half an hour.
5. Open the **Guide** (top right) to see every part of the simulation: what it is, what it may do, how its
   built-in model decides, what a language model is told, all its settings, and what it is doing in the run.
6. Under **Events**, set the weather (rain and snow are shortcuts; the sliders take any strength, for any part
   of the morning), or switch the crash off.
7. Under **Agents**, the link **goal and prompt** of a kind decided by a language model shows the prompt it
   will be sent. Change the kind's goal, add instructions, or write the prompt yourself, and run again.

What each part of the page shows: [](quickstart/interface.md). The network and the morning it simulates:
[](testbed.md).
