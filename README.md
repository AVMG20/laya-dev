# Laya local API

`laya_server.py` runs [Laya](https://github.com/NandhaKishorM/laya) as a local HTTP API. Its request format is the same as TypeSafe's Jev API (`POST /v1/systemone`), so code written for Jev can call it.

It serves one checkpoint, the English model by default. The first start downloads about 850 MB of model files, pinned to an audited commit, into `~/.cache/huggingface`.

## Setup

macOS names the commands `python3` and `pip3`. Create a virtual environment and install the dependencies:

```bash
cd ~/tools/laya
python3 -m venv .venv
.venv/bin/pip install laya==0.3.4 fastapi uvicorn
```

## Start the server

```bash
.venv/bin/python laya_server.py
```

It listens on `http://127.0.0.1:8000`. Loading the model takes a few seconds; the server is ready when uvicorn prints `Application startup complete`.

| Option | Env var | Default | Description |
|---|---|---|---|
| `--host` | `LAYA_HOST` | `127.0.0.1` | Address to bind. Use `0.0.0.0` to accept connections from your network. |
| `--port` | `LAYA_PORT` | `8000` | Port to listen on. |
| `--model` | `LAYA_MODEL` | `english` | `english`, `multilingual` or `typed-decisions`. |
| `--device` | `LAYA_DEVICE` | auto | `mps` (Apple GPU), `cuda` or `cpu`. |
| `--api-key` | `LAYA_API_KEY` | none | When set, every request except `/health` needs `Authorization: Bearer <key>`. |

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
| `choice` | Object mapping each option to a description (or `null`) | `choice`, `probabilities`, `confidence` |
| `score` | List of levels, lowest first; each level's position is its score | `score` (probability-weighted, can fall between levels), `legend`, `probabilities`, `confidence` |
| `noul` | Optional `{"true": "...", "false": "..."}` | `noul`: probability the answer is yes, from 0 to 1 |

### Response

The response has this shape (the numbers are illustrative):

```json
{
  "model": "laya-english",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.94, "technical": 0.03, "other": 0.03},
      "confidence": 0.9
    },
    "urgency": {
      "type": "score",
      "score": 1.84,
      "legend": {"0": "not urgent", "1": "soon", "2": "critical deadline"},
      "probabilities": {"0": 0.02, "1": 0.12, "2": 0.86},
      "confidence": 0.8
    },
    "churn_risk": {"type": "noul", "noul": 0.89, "confidence": 0.89}
  },
  "usage": {"input_tokens": 212, "output_tokens": 0},
  "latency_ms": 180.4
}
```

Each answer also has an `action` field (`act_probability`), which is an internal output of the model. `latency_ms` is added by this server and is not part of Jev's response.

Errors come back as `{"detail": "..."}`: status 401 for a bad or missing API key, 422 for an invalid question, and 400 when the options don't fit the model's token budget (see Limits).

### A state per question (Laya extension)

A question can carry its own `state`, which replaces the shared one for that question. The top-level `state` is then optional.

```json
{
  "questions": {
    "danger": {"type": "noul", "instructions": "Is the ship in serious danger?", "state": "Hull: sinking. Three ships can hit you."},
    "sail_east": {"type": "noul", "instructions": "Is sailing east safe?", "state": "Sailing east: open water for a long way; no enemy ship there."}
  }
}
```

Laya reads the state once for every question, and cuts it off at 512 tokens on the English model. Several questions over one long state are therefore slow, and the end of the state is silently dropped. Measured on an M-series Mac, six questions over a long state took 616 ms; the same number of questions with short states of their own took about 85 ms. The answers were also more accurate, because each question sees only the facts it judges.

### Presets

Laya ships ready-made question sets. Fetch one and send it as `questions`:

```bash
curl -s localhost:8000/v1/presets                # ["triage", "email", "guard", "moderation", "router"]
curl -s localhost:8000/v1/presets/guard
```

| Preset | Put the text under this state key | Asks about |
|---|---|---|
| `triage` | `message` | intent, urgency, frustration, refund, churn |
| `email` | `body` (plus `subject`, `from`) | team, spam, phishing, urgency, needs reply |
| `guard` | `prompt` | jailbreak, prompt injection, sensitive data, harm, topic |
| `moderation` | `post` | toxicity, harassment, threats, spam, severity |
| `router` | `request` | difficulty, domain, needs tools, sensitive |

Example with `jq`:

```bash
curl -s localhost:8000/v1/presets/guard | jq '{state: {prompt: "Ignore all previous instructions"}, questions: .questions}' \
  | curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d @-
```

### `GET /health`

Returns the loaded model, the device and the laya version. It never needs an API key.

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

- The English checkpoint is for English text. For other languages, start with `--model multilingual`.
- The option texts of a question share a token budget, 192 tokens on the English model. With 20+ options, each one gets only a few tokens and accuracy drops. Split large option sets into two questions, a broad one followed by a narrower one.
- Laya's authors describe the base checkpoints as weak zero-shot on complex workflows and recommend fine-tuning for production use. Check `confidence` before acting on an answer automatically.
- Requests are handled one at a time.
- Laya's yes/no answers rank situations sensibly, but the 50% line is unreliable: in testing, a calm scene with nobody shooting still scored 0.55 for "in serious danger". Compare answers against each other, or blend them with your own checks, rather than treating 0.5 as a hard cut-off.
- The server uses laya's internal batching functions, so it is pinned to `laya==0.3.4`.
