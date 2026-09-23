# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "decider-ai==1.2.1",
#     "fastapi>=0.110",
#     "uvicorn>=0.29",
# ]
# ///
"""Local, Jev-compatible HTTP API for decider (https://huggingface.co/Mapika/decider-2b).

Serves a single decider checkpoint with the same wire format as TypeSafe's Jev API:

    POST /v1/systemone   {"state": ..., "questions": {...}, "model": "..." (ignored)}

so the official TypeSafe SDK works against it unchanged:

    export TYPESAFE_BASE_URL=http://127.0.0.1:8000
    export TYPESAFE_API_KEY=local            # any value unless --api-key is set

Extension: a question may carry its own "state", which replaces the shared one for that
question. Short per-question states are faster and more accurate than one long state.

Extras:

    GET  /health

Run:

    uv run decider_server.py
    # or: pip install decider-ai==1.2.1 fastapi uvicorn && python decider_server.py
"""
import argparse
import json
import os
import threading
import time
import uuid
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from decider.infer import Decider

HUB_REPO = "Mapika/decider-2b"
# Pinned hub commit, so a later push to the repo can't silently change what gets loaded.
HUB_REVISION = "9839cc9d908be16c5988c0d041034b5fdf82c7a2"
DECIDER_VERSION = version("decider-ai")


class SystemOneRequest(BaseModel):
    # Optional only because a question may carry its own `state` (an extension, see predict).
    state: Optional[Union[str, Dict[str, Any], List[Any]]] = None
    questions: Dict[str, Dict[str, Any]] = Field(..., min_length=1)
    model: Optional[str] = None  # accepted for Jev compatibility, ignored


def predict(decider: Decider, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Decider.system_one, except a question may carry its own `state`.

    decider scores every question in its own row anyway, so every question only needs the facts it
    judges. Questions that share a state are sent together; the answers come back in request order.
    """
    groups: Dict[str, Any] = {}
    for qid, q in questions.items():
        own = q.get("state")
        st = state if own is None else own
        key = json.dumps(st, sort_keys=True, ensure_ascii=False)
        groups.setdefault(key, (st, {}))[1][qid] = {k: v for k, v in q.items() if k != "state"}

    answers, input_tokens = {}, 0
    for st, qs in groups.values():
        out = decider.system_one(st, qs)
        answers.update(out["answers"])
        input_tokens += out["usage"]["input_tokens"]
    return {"answers": {qid: answers[qid] for qid in questions},
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


def download_checkpoint(repo: str, revision: Optional[str]) -> str:
    """Fetch only the weights, tokenizer and configs; the repo also carries code and a README."""
    if os.path.isdir(repo):
        return repo
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=revision or (HUB_REVISION if repo == HUB_REPO else None),
        allow_patterns=["*.json", "*.safetensors", "*.jinja"],
        token=os.environ.get("HF_TOKEN"),
    )


def build_app(decider: Decider, api_key: Optional[str]) -> FastAPI:
    app = FastAPI(title="decider local API", version=DECIDER_VERSION)
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
            raise HTTPException(404, "index.html not found next to decider_server.py")
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
        return {"status": "ok", "model": decider.name, "device": str(decider.dev), "version": DECIDER_VERSION}

    @app.post("/v1/systemone", dependencies=[Depends(check_auth)])
    def system_one(req: SystemOneRequest):
        missing = [qid for qid, q in req.questions.items() if req.state is None and q.get("state") is None]
        if missing:
            raise HTTPException(422, "state is required unless every question has its own (missing for %s)" % ", ".join(missing))
        start = time.perf_counter()
        try:
            with lock:
                result = predict(decider, req.state, req.questions)
        except (ValueError, AssertionError) as e:
            # decider rejects malformed questions (unknown type, bad criteria, no instructions) this way.
            raise HTTPException(422, str(e))
        return {
            "model": decider.name,
            "answers": result["answers"],
            "usage": result["usage"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }

    return app


def main():
    p = argparse.ArgumentParser(description="Serve a decider checkpoint as a local Jev-compatible API.")
    p.add_argument("--host", default=os.environ.get("DECIDER_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("DECIDER_PORT", "8000")))
    p.add_argument("--model", default=os.environ.get("DECIDER_MODEL", HUB_REPO),
                   help="hub repo or local folder of a decider checkpoint (default: %s)" % HUB_REPO)
    p.add_argument("--revision", default=os.environ.get("DECIDER_REVISION"),
                   help="hub commit, branch or tag (default: a pinned commit for %s, else main)" % HUB_REPO)
    p.add_argument("--device", default=os.environ.get("DECIDER_DEVICE"), help="cuda, mps or cpu (default: auto)")
    p.add_argument("--api-key", default=os.environ.get("DECIDER_API_KEY"),
                   help="require 'Authorization: Bearer <key>' (default: no auth)")
    args = p.parse_args()

    print("[decider] loading %s ..." % args.model, flush=True)
    decider = Decider(download_checkpoint(args.model, args.revision), device=args.device)
    uvicorn.run(build_app(decider, args.api_key), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
