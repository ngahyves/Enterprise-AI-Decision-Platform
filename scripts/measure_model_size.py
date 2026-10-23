# scripts/measure_model_sizes.py
"""Measure disk size of all cached HuggingFace models."""

from pathlib import Path

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

if not CACHE_DIR.exists():
    print(f"Cache not found: {CACHE_DIR}")
    raise SystemExit(1)

print(f"{'Model':<60} {'Size':>10}")
print("=" * 72)

total = 0.0
for model_dir in sorted(CACHE_DIR.glob("models--*")):
    size_gb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024**3)
    total += size_gb
    name = model_dir.name.replace("models--", "").replace("--", "/")
    print(f"{name:<60} {size_gb:>8.2f} GB")

print("=" * 72)
print(f"{'TOTAL':<60} {total:>8.2f} GB")
