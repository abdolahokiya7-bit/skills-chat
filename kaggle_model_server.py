"""
Kaggle GPU model server for the skills chat.

Run this in a Kaggle Notebook with GPU enabled (Settings -> Accelerator -> GPU T4 x2)
and Internet turned ON (required for model download and the tunnel).

Steps in the notebook:
  1. Add a new cell:   !python /kaggle/working/kaggle_model_server.py
     (or paste this file's content into a cell and run it)
  2. Copy the printed PUBLIC_URL and the printed API_TOKEN into the web server's .env.
  3. Keep the notebook session alive. Sessions stop after about 12 hours;
     run the cell again to get a new URL and token.

Environment variables (optional):
  MODEL_ID   default: Qwen/Qwen2.5-7B-Instruct
  API_TOKEN  default: random token generated on start
  PORT       default: 8000
  PUBLIC_BASE  optional: fixed public URL when running on your own server (skips the tunnel)
"""
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
API_TOKEN = os.environ.get("API_TOKEN") or secrets.token_urlsafe(24)
PORT = int(os.environ.get("PORT", "8000"))


def pip_install():
    pkgs = ["transformers>=4.45", "accelerate", "bitsandbytes", "fastapi", "uvicorn"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def ensure_cloudflared():
    if shutil.which("cloudflared"):
        return "cloudflared"
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    path = "/kaggle/working/cloudflared"
    urllib.request.urlretrieve(url, path)
    os.chmod(path, 0o755)
    return path


def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def build_app(tokenizer, model):
    import torch
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel
    from typing import List, Optional

    app = FastAPI()

    class Message(BaseModel):
        role: str
        content: str

    class GenerateRequest(BaseModel):
        system: Optional[str] = None
        messages: List[Message]
        max_new_tokens: int = 1024
        temperature: float = 0.7

    @app.get("/health")
    def health():
        return {"ok": True, "model": MODEL_ID}

    @app.post("/generate")
    def generate(req: GenerateRequest, authorization: Optional[str] = Header(None)):
        if authorization != f"Bearer {API_TOKEN}":
            raise HTTPException(status_code=401, detail="bad token")

        messages = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages += [m.model_dump() for m in req.messages]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new = min(max(req.max_new_tokens, 16), 2048)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=req.temperature > 0,
                temperature=max(req.temperature, 0.01),
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {"text": text}

    return app


def _drain(pipe):
    """Keep reading cloudflared's output. If nobody reads the pipe, it fills up
    and the tunnel process blocks, which silently kills the public URL."""
    for _ in pipe:
        pass


def start_tunnel(port):
    """Open a quick Cloudflare tunnel and return (public https URL, process)."""
    binary = ensure_cloudflared()
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://localhost:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    pattern = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = time.time() + 90
    for line in proc.stdout:
        m = pattern.search(line)
        if m:
            threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
            return m.group(0), proc
        if time.time() > deadline:
            break
    proc.kill()
    raise RuntimeError("tunnel URL not found; check cloudflared output")


def announce_tunnel(port):
    """Runs in a background thread after the server has started."""
    time.sleep(2)
    for attempt in range(3):
        try:
            public_url, _ = start_tunnel(port)
            print("=" * 60, flush=True)
            print(f"PUBLIC_URL = {public_url}", flush=True)
            print(f"API_TOKEN  = {API_TOKEN}", flush=True)
            print("=" * 60, flush=True)
            return
        except Exception as e:
            print(f"tunnel attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(5)
    print("could not open the tunnel. Run this cell again.", flush=True)


def run_server(app):
    """Run uvicorn in its own thread. Kaggle notebooks already run an asyncio event
    loop in the main thread, and uvicorn.run() cannot start inside a running loop.
    A new thread has no running loop, so this works in notebooks and scripts."""
    import uvicorn

    config = uvicorn.Config(app, host="0.0.0.0", port=PORT, log_level="warning")
    uvicorn.Server(config).run()


def main():
    pip_install()

    print(f"loading {MODEL_ID} in 4-bit ...", flush=True)
    tokenizer, model = load_model()
    app = build_app(tokenizer, model)

    threading.Thread(target=run_server, args=(app,), daemon=True).start()
    fixed = os.environ.get("PUBLIC_BASE")  # set this on your own server; no tunnel needed
    if fixed:
        print(f"PUBLIC_URL = {fixed}\nAPI_TOKEN  = {API_TOKEN}", flush=True)
    else:
        threading.Thread(target=announce_tunnel, args=(PORT,), daemon=True).start()

    # keep the notebook cell alive while the server runs
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
