# decider local API

`decider_server.py` runs [decider-2b](https://huggingface.co/Mapika/decider-2b) as a local HTTP API. Its request format is the same as TypeSafe's Jev API (`POST /v1/systemone`), so code written for Jev can call it.

decider-2b is a 2B-parameter decision model fine-tuned from Qwen3.5-2B-Base. It does not generate text: it reads a state and typed questions and returns a calibrated probability for every option. The first start downloads about 3.5 GB of model files, pinned to a tested commit, into `~/.cache/huggingface`.

## Setup

decider needs Python 3.11 or newer. Create a virtual environment and install the dependencies:

```bash
cd ~/tools/decider
python3 -m venv .venv
.venv/bin/pip install decider-ai==1.2.1 fastapi uvicorn
```

## Start the server

```bash
.venv/bin/python decider_server.py
```

It listens on `http://127.0.0.1:8000`. The server is ready when uvicorn prints `Application startup complete`.

| Option | Env var | Default | Description |
|---|---|---|---|
| `--host` | `DECIDER_HOST` | `127.0.0.1` | Address to bind. Use `0.0.0.0` to accept connections from your network. |
| `--port` | `DECIDER_PORT` | `8000` | Port to listen on. |
| `--model` | `DECIDER_MODEL` | `Mapika/decider-2b` | Hub repo or local folder of a decider checkpoint, e.g. `Mapika/decider-0.8b` or `Mapika/decider-4b`. |
| `--revision` | `DECIDER_REVISION` | pinned commit | Hub commit, branch or tag. The default pins `Mapika/decider-2b` to a tested commit; other repos default to `main`. |
| `--device` | `DECIDER_DEVICE` | auto | `cuda`, `mps` (Apple GPU) or `cpu`. |
| `--api-key` | `DECIDER_API_KEY` | none | When set, every request except `/health` needs `Authorization: Bearer <key>`. |

With no API key the server has no authentication. Only bind to `0.0.0.0` together with `--api-key`.

Open `http://127.0.0.1:8000/` for the playground (`index.html`), a page for building questions and running them against the model.

Web pages can call the API directly from the browser: the server allows any origin (CORS) and answers Chrome's local-network permission check. Pirate Raid's auto-play in nuxtpolymarket uses this.

## Using the API

### `POST /v1/systemone`

Send a `state` (a string, JSON object or list) and one or more named `questions`. The server answers all of them in one pass.

```bash
curl -s localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "subject": "Duplicate charge on invoice #4411",
      "body": "We were billed twice for March. Refund the duplicate today or we cancel."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": null}
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["not urgent", "soon", "critical deadline"]
      },
      "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel?"
      }
    }
  }'
```

### Question types

| Type | `criteria` | Answer fields |
|---|---|---|
| `choice` | Object mapping each option to a description (or `null`), 2 to 255 options | `choice`, `probabilities`, `confidence`, `certainty` |
| `score` | List of 2 to 10 levels, lowest first; each level's position is its score | `score` (probability-weighted, can fall between levels), `legend`, `probabilities`, `confidence`, `certainty`, `level_fit`, `fit_mass` |
| `noul` | Optional `{"true": "...", "false": "..."}` | `noul`: probability the answer is yes, from 0 to 1 |

Every question needs non-empty `instructions`. `instructions` and every description may be a string or any JSON value.

### Response

The response for the request above, measured on CPU:

```json
{
  "model": "decider-v10",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9972,
      "certainty": 0.982,
      "probabilities": {"billing": 0.9972, "technical": 0.0002, "other": 0.0026}
    },
    "urgency": {
      "type": "score",
      "score": 1.49,
      "confidence": 0.5249,
      "certainty": 0.248,
      "legend": {"0": "not urgent", "1": "soon", "2": "critical deadline"},
      "probabilities": {"0": 0.0387, "1": 0.4364, "2": 0.5249},
      "level_fit": {"0": 0.029, "1": 0.3271, "2": 0.3935},
      "fit_mass": 0.7497
    },
    "churn_risk": {"type": "noul", "noul": 0.9579}
  },
  "usage": {"input_tokens": 200, "output_tokens": 0},
  "latency_ms": 20459.7
}
```

`confidence` is the calibrated probability of the top option; `certainty` is 1 minus the normalised entropy. Each score level is judged on its own: `level_fit` holds the per-level fits and `fit_mass` their sum, which is near 1 when exactly one level fits. `latency_ms` is added by this server and is not part of Jev's response.

Errors come back as `{"detail": "..."}`: status 401 for a bad or missing API key and 422 for an invalid question.

### A state per question (extension)

A question can carry its own `state`, which replaces the shared one for that question. The top-level `state` is then optional.

```json
{
  "questions": {
    "danger": {"type": "noul", "instructions": "Is the ship in serious danger?", "state": "Hull: sinking. Three ships can hit you."},
    "sail_east": {"type": "noul", "instructions": "Is sailing east safe?", "state": "Sailing east: open water for a long way; no enemy ship there."}
  }
}
```

decider scores every question in its own row, so each question only needs the facts it judges. Short states of their own are faster than one long shared state, and each question sees only what matters to it.

### `GET /health`

Returns the loaded model, the device and the decider-ai version. It never needs an API key.

## From Python

Standard library only:

```python
import json
import urllib.request

def ask(state, questions, url="http://127.0.0.1:8000/v1/systemone"):
    req = urllib.request.Request(url, json.dumps({"state": state, "questions": questions}).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)["answers"]

answers = ask("My payment failed twice", {"is_urgent": {"type": "noul", "instructions": "Is this urgent?"}})
print(answers["is_urgent"]["noul"])
```

With the official TypeSafe SDK (`pip install typesafe-sdk`, which needs a newer Python than macOS's built-in 3.9), point it at this server:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8000
export TYPESAFE_API_KEY=local   # the SDK requires a value; use your --api-key if you set one
```

## Limits

- decider-2b is English only.
- On CPU it is slow. On a 4-core machine, two short questions took about 4.7 s, and the example above took about 20 s, because each score level is scored as its own row. The model card reports a few milliseconds per request on a CUDA GPU.
- It is a 2B model without reasoning. Split a judgment that needs several steps into several questions, and state rules as plain questions with described options.
- Calibration is measured on public datasets, not on your traffic. Check `confidence` against your own labels before acting on it automatically.
- Name or describe a catch-all option (`general_support`, or `other` with a description) rather than using a terse bucket name.
- Requests are handled one at a time.
