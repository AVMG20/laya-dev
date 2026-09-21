# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "laya==0.3.4",  # predict() uses laya.common internals
#     "fastapi>=0.110",
#     "uvicorn>=0.29",
# ]
# ///
"""Local, Jev-compatible HTTP API for Laya (https://github.com/NandhaKishorM/laya).

Serves a single Laya checkpoint with the same wire format as TypeSafe's Jev API:

    POST /v1/systemone   {"state": ..., "questions": {...}, "model": "..." (ignored)}

so the official TypeSafe SDK works against it unchanged:

    export TYPESAFE_BASE_URL=http://127.0.0.1:8000
    export TYPESAFE_API_KEY=local            # any value unless --api-key is set

Laya extension: a question may carry its own "state", which replaces the shared one for that
question. Short per-question states are much faster and more accurate than one long state.

Extras:

    GET  /v1/presets            list built-in question presets
    GET  /v1/presets/{name}     question schema for a preset (triage, email, guard, ...)
    GET  /health

Run:

    uv run laya_server.py
    # or: pip install laya==0.3.4 fastapi uvicorn && python laya_server.py
"""
import argparse
import contextlib
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import laya
from laya.common import QTYPES, build_sequence, collate_items, confidence_from_probs, render_options, temp_bucket

HUB_REPO = "convaiinnovations/laya"
# Pinned hub commit, so a later push to the repo can't silently change what gets loaded.
HUB_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
# Checkpoint name -> subfolder of the hub repo.
CHECKPOINTS = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}

PRESETS = {
    "triage": laya.triage_questions,
    "email": laya.email_questions,
    "guard": laya.guard_questions,
    "moderation": laya.moderation_questions,
    "router": laya.router_questions,
}


class SystemOneRequest(BaseModel):
    # Optional only because a question may carry its own `state` (a Laya extension, see predict).
    state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    questions: Dict[str, Dict[str, Any]] = Field(..., min_length=1)
    model: Optional[str] = None  # accepted for Jev compatibility, ignored


def normalise_questions(questions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Validate question types and fill the gaps Jev allows but Laya's Agent does not."""
    out = {}
    for qid, q in questions.items():
        t = q.get("type")
        if t not in ("choice", "score", "noul"):
            raise HTTPException(422, "questions.%s.type must be 'choice', 'score' or 'noul'" % qid)
        crit = q.get("criteria")
        if t == "choice" and not (isinstance(crit, (dict, list)) and crit):
            raise HTTPException(422, "questions.%s.criteria must be a non-empty object for choice" % qid)
        if t == "score" and not (isinstance(crit, list) and crit):
            raise HTTPException(422, "questions.%s.criteria must be a non-empty list for score" % qid)
        if t == "noul" and crit is not None and not isinstance(crit, dict):
            raise HTTPException(422, "questions.%s.criteria must be an object with 'true'/'false' for noul" % qid)
        out[qid] = {**q, "instructions": q.get("instructions") or ""}
    return out


@torch.no_grad()
def predict(agent: laya.Agent, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Agent.system_one, except a question may carry its own `state`.

    Laya encodes the state once per question anyway, so every question only needs the facts it
    judges. Short, focused states are both faster (a sequence per question, all in one batch) and
    more accurate than one long state that gets cut off at the checkpoint's max_len.
    """
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    ids = list(questions)
    internal = [agent._to_internal(questions[qid]) for qid in ids]
    items = []
    for qid, q in zip(ids, internal):
        own = questions[qid].get("state")
        seq, markers = build_sequence(agent.tok, state if own is None else own, q, max_len, head_max_len)
        if len(markers) != len(render_options(q)):
            raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

    b = collate_items([items], agent.tok.pad_token_id)
    dev = agent.device
    amp = torch.autocast(device_type="cuda", dtype=agent.dtype) if dev.type == "cuda" else contextlib.nullcontext()
    with amp:
        logits, act = agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                  b["marker_mask"].to(dev), b["qtype"].to(dev))
    logits = logits.float().cpu().numpy()
    act = torch.softmax(act.float(), -1).cpu().numpy()

    answers = {}
    for r, (qid, q) in enumerate(zip(ids, internal)):
        k = len(items[r]["markers"])
        qt = QTYPES[q["t"]]
        z = logits[r, :k] / max(1e-3, float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])))
        p = np.exp(z - z.max())
        p = p / p.sum()
        ext = {"act_probability": round(float(act[r, 0]), 4)}
        if q["t"] == "choice":
            keys = list(q["crit"].keys())
            answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                            "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                            "confidence": round(confidence_from_probs(p, k), 4), "action": ext}
        elif q["t"] == "score":
            answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                            "legend": {str(i): c for i, c in enumerate(q["crit"])},
                            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                            "confidence": round(confidence_from_probs(p, k), 4), "action": ext}
        else:
            answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                            "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
    return {"answers": answers, "usage": {"input_tokens": int(b["attention_mask"].sum()), "output_tokens": 0}}


def download_checkpoint(name: str) -> str:
    """Fetch only the files one checkpoint needs; laya.load would pull the whole repo for 'english'."""
    from huggingface_hub import snapshot_download

    sub = CHECKPOINTS[name]
    prefix = sub + "/" if sub else ""
    path = snapshot_download(
        HUB_REPO,
        revision=HUB_REVISION,
        allow_patterns=[prefix + p for p in ("rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*")],
        token=os.environ.get("HF_TOKEN"),
    )
    return os.path.join(path, sub) if sub else path


def build_app(agent: laya.Agent, model_name: str, api_key: Optional[str]) -> FastAPI:
    app = FastAPI(title="Laya local API", version=laya.__version__)
    # Browsers call this directly (the playground page, the Pirate Raid autopilot). Auth, when
    # enabled, is a bearer header, not a cookie, so allowing any origin exposes nothing extra.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                       allow_headers=["Authorization", "Content-Type"], expose_headers=["x-typesafe-request-id"])
    # Chrome asks before a public site may reach a loopback address; this header is the server's consent.
    @app.middleware("http")
    async def private_network(request: Request, call_next):
        response = await call_next(request)
        if request.headers.get("access-control-request-private-network"):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    playground = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

    @app.get("/", include_in_schema=False)
    def index():
        if not os.path.exists(playground):
            raise HTTPException(404, "index.html not found next to laya_server.py")
        return FileResponse(playground)
    # The torch model is not safe to share across the threadpool FastAPI runs sync handlers in.
    lock = threading.Lock()

    def check_auth(request: Request):
        if api_key and request.headers.get("authorization") != "Bearer " + api_key:
            raise HTTPException(401, "Invalid or missing API key.")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        response = await call_next(request)
        response.headers["x-typesafe-request-id"] = uuid.uuid4().hex
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "device": str(agent.device), "version": laya.__version__}

    @app.post("/v1/systemone", dependencies=[Depends(check_auth)])
    def system_one(req: SystemOneRequest):
        questions = normalise_questions(req.questions)
        missing = [qid for qid, q in questions.items() if req.state is None and q.get("state") is None]
        if missing:
            raise HTTPException(422, "state is required unless every question has its own (missing for %s)" % ", ".join(missing))
        start = time.perf_counter()
        try:
            with lock:
                result = predict(agent, req.state, questions)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return {
            "model": model_name,
            "answers": result["answers"],
            "usage": result["usage"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    @app.get("/v1/presets", dependencies=[Depends(check_auth)])
    def presets():
        return {"presets": list(PRESETS)}

    @app.get("/v1/presets/{name}", dependencies=[Depends(check_auth)])
    def preset(name: str):
        if name not in PRESETS:
            raise HTTPException(404, "Unknown preset %r. Available: %s" % (name, ", ".join(PRESETS)))
        return {"questions": PRESETS[name]()}

    return app


def main():
    p = argparse.ArgumentParser(description="Serve a Laya checkpoint as a local Jev-compatible API.")
    p.add_argument("--host", default=os.environ.get("LAYA_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("LAYA_PORT", "8000")))
    p.add_argument("--model", default=os.environ.get("LAYA_MODEL", "english"), choices=list(CHECKPOINTS),
                   help="which Laya checkpoint to serve (default: english)")
    p.add_argument("--device", default=os.environ.get("LAYA_DEVICE"), help="cuda, mps or cpu (default: auto)")
    p.add_argument("--api-key", default=os.environ.get("LAYA_API_KEY"),
                   help="require 'Authorization: Bearer <key>' (default: no auth)")
    args = p.parse_args()

    print("[laya] loading %s checkpoint ..." % args.model, flush=True)
    agent = laya.load(download_checkpoint(args.model), device=args.device)
    uvicorn.run(build_app(agent, "laya-" + args.model, args.api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
