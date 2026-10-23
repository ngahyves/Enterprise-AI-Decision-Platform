# scripts/measure_operational.py
"""Measure RAM, throughput, and latency of the running API.

Requires: uvicorn app.main:app running on port 8000.

Measures:
  - RAM of the uvicorn process
  - Throughput and average latency over N COLD requests (cache disabled)
  - Tokens per query (via LangSmith if LANGCHAIN_API_KEY is set)

Note: cache is explicitly disabled (use_cache=False) to measure cold latency.
      Repeated queries would return in ~6 ms thanks to Redis.
"""

import os
import time

import psutil
import requests

from src.common.config_manager import env_settings

BASE = "http://localhost:8000"
API_KEY = os.getenv("API_KEY") or getattr(env_settings, "api_key", None)
if not API_KEY:
    raise RuntimeError(
        "API_KEY not found. Set it in your .env file (same variable used by app/auth.py)."
    )

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

QUESTION = "What was TD net income 2025?"
N_REQUESTS = 3
TIMEOUT_S = 300


def find_uvicorn_process():
    """Return the uvicorn process (highest RSS python.exe)."""
    candidates = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["name"] and "python" in p.info["name"].lower():
                candidates.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.memory_info().rss)


def measure_ram(proc):
    if proc is None:
        return None
    mem = proc.memory_info()
    return {
        "rss_gb": mem.rss / (1024**3),
        "vms_gb": mem.vms / (1024**3),
        "pid": proc.pid,
    }


def measure_throughput():
    latencies = []
    for i in range(N_REQUESTS):
        t0 = time.time()
        try:
            r = requests.post(
                f"{BASE}/v1/ask",
                json={"question": QUESTION, "use_cache": False},
                headers=HEADERS,
                timeout=TIMEOUT_S,
            )
        except requests.RequestException as e:
            print(f"  [{i + 1}/{N_REQUESTS}] request failed: {e}")
            continue
        elapsed = time.time() - t0
        if r.status_code == 200:
            latencies.append(elapsed)
            print(f"  [{i + 1}/{N_REQUESTS}] {elapsed:.1f}s")
        else:
            print(f"  [{i + 1}/{N_REQUESTS}] HTTP {r.status_code}: {r.text[:120]}")

    if not latencies:
        return None

    avg = sum(latencies) / len(latencies)
    return {
        "n": len(latencies),
        "avg_latency_s": avg,
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "throughput_req_per_s": 1.0 / avg if avg > 0 else 0,
    }


def measure_tokens_langsmith():
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "default"
    if not api_key:
        return None
    try:
        from langsmith import Client

        client = Client()
        runs = list(client.list_runs(project_name=project, limit=20))
        if not runs:
            return None
        in_tokens = sum(getattr(r, "prompt_tokens", 0) or 0 for r in runs)
        out_tokens = sum(getattr(r, "completion_tokens", 0) or 0 for r in runs)
        return {
            "n_runs": len(runs),
            "avg_input_tokens": in_tokens / len(runs),
            "avg_output_tokens": out_tokens / len(runs),
        }
    except Exception as e:
        print(f"  LangSmith error: {e}")
        return None


def measure_cached_latency():
    # Warm-up with use_cache=True to populate Redis
    requests.post(
        f"{BASE}/v1/ask",
        json={"question": QUESTION, "use_cache": True},  # ← True, pas False
        headers=HEADERS,
        timeout=TIMEOUT_S,
    )
    # Now measure the cached response
    t0 = time.time()
    r = requests.post(
        f"{BASE}/v1/ask",
        json={"question": QUESTION, "use_cache": True},
        headers=HEADERS,
        timeout=30,
    )
    elapsed = time.time() - t0
    if r.status_code == 200:
        data = r.json()
        return {
            "cached": data.get("cached", False),
            "latency_ms": data.get("latency_ms", 0),
            "wall_s": elapsed,
        }
    return None


def main():
    print("=" * 60)
    print("OPERATIONAL METRICS")
    print("=" * 60)

    print("\n[1] RAM (uvicorn process)")
    proc = find_uvicorn_process()
    ram = measure_ram(proc)
    if ram:
        print(f"  PID: {ram['pid']}")
        print(f"  RSS: {ram['rss_gb']:.2f} GB")
        print(f"  VMS: {ram['vms_gb']:.2f} GB")
    else:
        print("  Uvicorn process not found")

    print(f"\n[2] Throughput ({N_REQUESTS} COLD requests, cache disabled)")
    thr = measure_throughput()
    if thr:
        print(f"  Avg latency : {thr['avg_latency_s']:.1f}s")
        print(f"  Min / Max   : {thr['min_latency_s']:.1f}s / {thr['max_latency_s']:.1f}s")
        print(f"  Throughput  : {thr['throughput_req_per_s']:.4f} req/s")
    else:
        print("  No successful requests")

    print("\n[3] Tokens per query (LangSmith)")
    tok = measure_tokens_langsmith()
    if tok:
        print(f"  Runs sampled: {tok['n_runs']}")
        print(f"  Avg input   : {tok['avg_input_tokens']:.0f} tokens")
        print(f"  Avg output  : {tok['avg_output_tokens']:.0f} tokens")
    else:
        print("  Not available (LANGCHAIN_API_KEY not set or no runs)")

    print("\n" + "=" * 60)
    print("\n[3] Cached latency (2nd call, use_cache=True)")
    cached = measure_cached_latency()
    if cached:
        print(f"  cached     : {cached['cached']}")
        print(f"  latency_ms : {cached['latency_ms']:.1f} ms")
        print(f"  wall time  : {cached['wall_s'] * 1000:.1f} ms")
    else:
        print("  Failed")


if __name__ == "__main__":
    main()
