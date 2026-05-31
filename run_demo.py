"""Launch the HEMT-CLIP Streamlit demo (optionally via an ngrok tunnel).

Replaces the old `notebooks/06_demo.ipynb` — a plain script is simpler for a demo.

It (1) discovers the headline checkpoint, (2) exports the env vars the app reads
(`HEMT_CLIP_CKPT`, `HEMT_CLIP_VARIANT`), (3) starts `app/streamlit_app.py`, and
(4) opens an ngrok HTTPS tunnel so the demo is reachable from any browser (e.g.
during the viva). Ctrl-C tears everything down.

Usage
-----
Colab / remote (public URL via ngrok) — set your token once, then run:
    NGROK_AUTHTOKEN=xxxxx  python run_demo.py

Local (no tunnel, opens on http://localhost:8501):
    python run_demo.py --no-tunnel

Useful flags:
    --variant gated_fusion     # model to serve (default: the headline α-gated HEMT-CLIP)
    --ckpt /path/to/best.pt    # explicit checkpoint (skips auto-discovery)
    --ckpt-dir /path/to/dir    # where to auto-discover checkpoints
    --port 8501                # streamlit port
    --no-tunnel                # skip ngrok; serve locally only

Prerequisites: `pip install -r requirements.txt` (includes streamlit + pyngrok),
and for the public tunnel an ngrok auth token in $NGROK_AUTHTOKEN (free tier is fine).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", default="gated_fusion",
                   help="Model variant to serve (default: gated_fusion = headline HEMT-CLIP).")
    p.add_argument("--ckpt", default=None,
                   help="Explicit checkpoint path. If omitted, auto-discover by --variant.")
    p.add_argument("--ckpt-dir", default=None,
                   help="Directory to auto-discover checkpoints in (default: cfg.checkpointing.dir).")
    p.add_argument("--config", default=str(ROOT / "configs/base.yaml"))
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--no-tunnel", action="store_true",
                   help="Serve locally only; do not open an ngrok tunnel.")
    return p.parse_args()


def discover_ckpt(variant: str, ckpt_dir: Path) -> str:
    """Latest non-`_seed*` best.pt for the variant (same rule evaluate.py uses)."""
    cands = [c for c in ckpt_dir.glob(f"hemt_{variant}_*_best.pt") if "_seed" not in c.name]
    if not cands:
        sys.exit(f"No checkpoint matching hemt_{variant}_*_best.pt in {ckpt_dir}. "
                 f"Pass --ckpt explicitly, or train the variant first.")
    return str(sorted(cands, key=lambda p: p.stat().st_mtime)[-1])


def wait_until_up(port: int, timeout: int = 25) -> bool:
    """Poll the streamlit port until it answers or the timeout elapses."""
    for i in range(timeout):
        time.sleep(1)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2)
            print(f"streamlit is up after {i + 1}s")
            return True
        except Exception:
            continue
    return False


def main() -> int:
    args = parse_args()
    if args.ckpt:
        ckpt = args.ckpt
    else:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ckpt_dir = Path(args.ckpt_dir or cfg["checkpointing"]["dir"])
        ckpt = discover_ckpt(args.variant, ckpt_dir)

    print(f"variant   : {args.variant}")
    print(f"checkpoint: {ckpt}")

    # The app reads these on startup.
    env = {**os.environ,
           "HEMT_CLIP_CKPT": ckpt,
           "HEMT_CLIP_VARIANT": args.variant}

    # Start streamlit headless.
    log_path = ROOT / "streamlit.log"
    proc = subprocess.Popen(
        ["streamlit", "run", str(ROOT / "app/streamlit_app.py"),
         "--server.port", str(args.port),
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT, env=env,
    )
    print(f"streamlit pid={proc.pid}, logs -> {log_path}")

    if not wait_until_up(args.port):
        print(f"streamlit did not come up in time — see {log_path}")
        proc.terminate()
        return 1

    tunnel = None
    try:
        if args.no_tunnel:
            print("\n" + "=" * 60)
            print(f"  HEMT-CLIP demo:  http://localhost:{args.port}")
            print("=" * 60)
        else:
            token = os.environ.get("NGROK_AUTHTOKEN")
            if not token:
                sys.exit("NGROK_AUTHTOKEN not set. Export it (free token from "
                         "https://dashboard.ngrok.com/get-started/your-authtoken) "
                         "or run with --no-tunnel for a local-only demo.")
            from pyngrok import conf, ngrok
            ngrok.set_auth_token(token)
            conf.get_default().monitor_thread = False
            ngrok.kill()  # clear stale tunnels from a previous run
            tunnel = ngrok.connect(args.port, proto="http", bind_tls=True)
            print("\n" + "=" * 60)
            print(f"  HEMT-CLIP demo URL:  {tunnel.public_url}")
            print("=" * 60)

        print("\nDemo is live. Press Ctrl-C to shut down.")
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        if tunnel is not None:
            from pyngrok import ngrok
            ngrok.kill()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        print("Demo stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
