# Language models

Any kind of agent can be decided for by a language model: set `use_llm: true` for the kind and name a model.

```bash
.venv/bin/agmsim run montreal --set agents.vehicle.use_llm=true --set llm.preset=deepseek
```

The model receives what any other decision model receives: the observation as JSON, with the actions it is
permitted. Its reply is parsed and validated against those actions; an invalid reply is re-asked once and
then replaced by the kind's built-in model. Fallbacks, tokens, latency and estimated cost are reported.

## Backends and presets

- `openai_compatible`: any Chat Completions server, local or hosted.
- `mock`: an offline stand-in for tests and demos. It is **not** a language model; it returns a heuristic's
  choice formatted as model output so that the whole path can be exercised.

`llm.preset` names a ready-made setting; anything you set next to it overrides the preset.

| Preset | Model | Needs |
| - | - | - |
| `qwen-local` | Qwen2.5-7B-Instruct served by mlx-lm on port 8089 | the server below |
| `deepseek` | `deepseek-flash`, thinking switched off | `DEEPSEEK_API_KEY` |
| `deepseek-thinking` | `deepseek-flash` with its reasoning phase on | `DEEPSEEK_API_KEY` |
| `deepseek-pro` | `deepseek-v4-pro`, thinking switched off | `DEEPSEEK_API_KEY` |

```bash
HF_HUB_OFFLINE=1 .venv/bin/mlx_lm.server --model Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8089
```

Another server or model:

```yaml
llm: {backend: openai_compatible, base_url: "http://127.0.0.1:8089/v1", model: "...", api_key_env: MY_KEY}
```

## Keys

API keys are read from the environment, or from a file named `.env` in the project folder. `.env` is ignored
by git and a key is never written to a configuration, trace or log. `llm.max_calls` and `llm.budget_usd` cap
spending.

## What the model is told

Before every decision the model gets a prompt that the simulator writes from what the kind declares about
itself: who the agent is, why it may be asked, what each action means, and the answer format. It is followed
by what the agent sees at that moment, which includes its **goal**: a sentence under `objective` in its state.
Both can be read and changed, per kind.

```bash
.venv/bin/agmsim prompt montreal vehicle      # the goal and the prompt of the private cars
```

```yaml
agents:
  vehicle:
    objective: "Get your owner there without using a bridge unless it saves ten minutes."
    llm_policy_params:
      extra_instructions: "Say in your reason which road you avoided."   # added at the end of the prompt
      # system_prompt: "..."        # your own prompt in place of the simulator's
```

| Setting | What it changes |
| - | - |
| `objective` | what the agents of the kind are told to pursue, in place of the kind's own goal. Travellers have none of their own: they are told who they are and decide as that person would. For on-demand cars the words `profit`, `service` and `balanced` stand for three ready-made goals |
| `llm_policy_params.extra_instructions` | text added at the end of the prompt |
| `llm_policy_params.system_prompt` | your own prompt in place of the simulator's. Keep its part on the answer format: a reply that cannot be read is asked for once more and then left to the built-in model |

The built-in models follow their own rules and read neither (the on-demand cars' built-in model follows the
three ready-made goals). On the page, a kind decided by a language model has a link **goal and prompt**: it
shows the prompt as it will be sent and lets you edit all three. The Guide shows the goal and the prompt of
every kind, and a selected decision shows the text the model was sent for it, word for word. In Python,
`agmsim.prompts.told("montreal")` returns the goal and the prompt of every kind.

## What the model is asked for

The answer format is set per kind with `llm_policy_params: {rationale: ...}`:

- `true` (default): the action, then one sentence of reason;
- `false`: the action only, which roughly halves output tokens and time;
- `first`: the reasoning *before* the action, which changes what models without a reasoning phase choose,
  not only what they say.

`llm_policy_params: {reflect: true}` tells the model to read its memory before it chooses.

## Faster runs

A run waits for the model once per tick in which someone decides. What helps, in order of effect:

- run arms and seeds in parallel (`agmsim matrix --workers N`, `experiments/suite.py`);
- leave out the rationale;
- agents deciding in the same tick are asked simultaneously (`llm.max_concurrency`);
- a cassette (`llm.cassette: file.jsonl`) replays recorded replies, so re-runs make no model calls.

`experiments/bench_llm.py` measures time per decision, throughput and the share of valid replies on real
decision points of the testbed.

## A decision benchmark

A closed-loop run shows what a decision model does to the system, but every model meets different decision
points. `agmsim/benchmark.py` fixes them: observations filled with numbers from a factorial design, each
labelled with the action that serves the person concerned.

| Task | Who decides | Items |
| - | - | - |
| `route_delegation` | vehicle (L2): pay a toll for the owner or not | 192 |
| `suggestion_response` | traveller: accept the vehicle's proposal or not | 171 |
| `transfer_proposal` | vehicle (L3): propose park-and-ride or drive on | 108 |

```python
from agmsim.benchmark import evaluate
report = evaluate("route_delegation", {"agents.vehicle.use_llm": True, "llm.preset": "deepseek"})
report["scores"]["sensitivity"]
```
