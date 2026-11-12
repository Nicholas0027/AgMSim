# Installation

## Nothing to install

The [simulator](../simulator.md) runs in your browser. Use it to
try the testbed, change levels and decision models, and read decisions. To let a language model decide,
choose `LLM` for a kind of agent and type your own API key: the key stays in that browser tab and the calls
go straight from your browser to the provider.

## On your machine

Python 3.10 or newer.

```bash
git clone https://github.com/Nicholas0027/AgMSim.git
cd AgMSim
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

or, without `uv`:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Check it:

```bash
.venv/bin/agmsim run montreal
.venv/bin/python -m pytest -q
```

## A language model (optional)

Nothing below is needed for runs with the built-in decision models.

- **A hosted model.** Put its key in a file named `.env` in the project folder (`DEEPSEEK_API_KEY=...`; copy
  `.env.example`). The file is ignored by git, and a key is never written to a configuration, trace or log.
- **A local model.** Any server that speaks the Chat Completions protocol. The preset `qwen-local` expects
  Qwen2.5-7B-Instruct served by `mlx-lm` on port 8089.

See [](../guide/llm.md).
