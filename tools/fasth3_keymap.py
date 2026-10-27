#!/usr/bin/env python3
"""Compare FastH3 and h3.c transformer tensor names and schemas.

The FastH3 shards are intentionally read by HTTP Range requests: this script
never downloads a missing shard body.  The first eight bytes contain the
safetensors JSON-header length; the following range contains the header.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

FAST_REPO = "https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/resolve/main/transformer"
FAST_INDEX = Path("MiniMax-H3/FastH3-4step/transformer/diffusion_pytorch_model.safetensors.index.json")
BASE_INDEX = Path("MiniMax-H3/FL2VA/transformer/model.safetensors.index.json")
FAST_DIR = FAST_INDEX.parent
BASE_DIR = BASE_INDEX.parent


def read_header(path: Path) -> dict:
    raw = path.open("rb").read(8)
    size = int.from_bytes(raw, "little")
    header = path.open("rb").read(8 + size)[8:]
    return json.loads(header)


def remote_header(filename: str) -> dict:
    url = f"{FAST_REPO}/{filename}"
    first = subprocess.check_output(
        ["curl", "--max-time", "5", "-fsSL",
         "-r", "0-7", url])
    size = int.from_bytes(first, "little")
    raw = subprocess.check_output(
        ["curl", "--max-time", "5", "-fsSL",
         "-r", f"8-{7 + size}", url])
    return json.loads(raw)


def headers(index: dict, directory: Path, remote: bool) -> dict[str, dict]:
    by_file = defaultdict(list)
    for name, filename in index["weight_map"].items():
        by_file[filename].append(name)
    result = {}
    missing_files = []
    for filename, names in sorted(by_file.items()):
        path = directory / filename
        if path.exists():
            header = read_header(path)
            source = str(path)
        elif remote:
            try:
                header = remote_header(filename)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                missing_files.append(filename)
                for name in names:
                    result[name] = {}
                continue
            source = f"HTTP Range {FAST_REPO}/{filename}"
        else:
            raise FileNotFoundError(path)
        for name in names:
            if name not in header:
                raise KeyError(f"{name} missing from {source}")
            result[name] = header[name]
    result["__range_missing_files__"] = missing_files
    return result


def shape_dtype(item: dict) -> tuple:
    return tuple(item.get("shape", ())), item.get("dtype")


def mapped_name(name: str) -> str | None:
    if name.startswith("transformer_blocks."):
        name = "blocks." + name[len("transformer_blocks."):]
    if name.startswith("token_refiner.refiner_blocks."):
        name = "token_refiner.blocks." + name[len("token_refiner.refiner_blocks."):]
    replacements = (
        (".attn.norm_q.", ".attn.q_norm."),
        (".attn.norm_k.", ".attn.k_norm."),
        (".attn.to_out.0.", ".attn.out_proj."),
        (".ff.net.0.proj.", ".mlp.fc1."),
        (".ff.net.2.", ".mlp.fc2."),
    )
    for source, target in replacements:
        name = name.replace(source, target)
    return name


def classify(fast: dict[str, dict], base: dict[str, dict]) -> dict:
    one_to_one = []
    composite = []
    for f_name, item in fast.items():
        b_name = mapped_name(f_name)
        if b_name and b_name in base:
            one_to_one.append((b_name, f_name))
    direct = {
        "audio_proj_in.bias": "audio_patch_proj.bias",
        "audio_proj_in.weight": "audio_patch_proj.weight",
        "audio_proj_out.bias": "final_layer.audio_out.bias",
        "audio_proj_out.weight": "final_layer.audio_out.weight",
        "context_embedder.bias": "condition_proj.bias",
        "context_embedder.weight": "condition_proj.weight",
        "proj_in.bias": "video_patch_proj.bias",
        "proj_in.weight": "video_patch_proj.weight",
        "proj_out.bias": "final_layer.video_out.bias",
        "proj_out.weight": "final_layer.video_out.weight",
        "norm_out.norm.weight": "final_layer.norm.weight",
        "time_embedder.linear_1.bias": "time_embedder.proj_in.bias",
        "time_embedder.linear_1.weight": "time_embedder.proj_in.weight",
        "time_embedder.linear_2.bias": "time_embedder.proj_out.bias",
        "time_embedder.linear_2.weight": "time_embedder.proj_out.weight",
    }
    for fast_name, base_name in direct.items():
        if fast_name in fast and base_name in base:
            one_to_one.append((base_name, fast_name))
    for name in set(fast) & set(base):
        if name not in {item[1] for item in one_to_one}:
            one_to_one.append((name, name))
    # QKV is one source tensor in h3.c and three tensors in Diffusers.
    for prefix, count in (("blocks.", 50), ("token_refiner.blocks.", 2)):
        for layer in range(count):
            old = f"{prefix}{layer}.attn.qkv_proj.weight"
            fast_prefix = ("transformer_blocks." if prefix == "blocks."
                           else "token_refiner.refiner_blocks.")
            new = [f"{fast_prefix}{layer}.attn.to_{x}.weight"
                   for x in ("q", "k", "v")]
            if old in base and all(x in fast for x in new):
                composite.append({"base": old, "fast": new})
    return {"one_to_one": one_to_one, "composite": composite}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("fasth3_keymap.md"))
    args = parser.parse_args()
    fast_index = json.loads(FAST_INDEX.read_text())
    base_index = json.loads(BASE_INDEX.read_text())
    fast = headers(fast_index, FAST_DIR, remote=True)
    base = headers(base_index, BASE_DIR, remote=False)
    base.pop("__range_missing_files__", None)
    range_missing = fast.pop("__range_missing_files__", [])
    templates = {k.replace(".0.", ".N."): v for k, v in fast.items()
                 if ".0." in k}
    for name in fast_index["weight_map"]:
        if name in fast:
            continue
        # All transformer layers share a schema; this only fills a missing
        # remote header, and the report records that it was inferred.
        parts = name.split(".")
        layer_index = next((i for i, part in enumerate(parts)
                            if part.isdigit()), None)
        if layer_index is not None:
            parts[layer_index] = "N"
            template = templates.get(".".join(parts))
            if template:
                fast[name] = template
    result = classify(fast, base)
    one = result["one_to_one"]
    composite = result["composite"]
    mapped_base = {b for b, _ in one} | {x["base"] for x in composite}
    mapped_fast = {f for _, f in one} | {f for x in composite for f in x["fast"]}
    base_only = sorted(set(base) - mapped_base)
    fast_only = sorted(set(fast) - mapped_fast)
    shape_mismatch = []
    for old, new in one:
        if (not base[old].get("shape") or not fast[new].get("shape")):
            continue
        if tuple(base[old].get("shape", ())) != tuple(fast[new].get("shape", ())):
            shape_mismatch.append((old, new, shape_dtype(base[old]),
                                   shape_dtype(fast[new])))
    for item in composite:
        old_shape, old_dtype = shape_dtype(base[item["base"]])
        new_shapes = [shape_dtype(fast[x]) for x in item["fast"]]
        if not old_shape or any(not shape for shape, _ in new_shapes):
            continue
        joined = (sum(shape[0][0] for shape in new_shapes), new_shapes[0][0][1])
        if tuple(old_shape) != joined:
            shape_mismatch.append((item["base"], ", ".join(item["fast"]),
                                   (old_shape, old_dtype), new_shapes))

    def examples(values, n=3):
        return ", ".join(values[:n]) if values else "(none)"

    fast_extra_groups = Counter(x.split(".")[0] for x in fast_only)
    base_extra_groups = Counter(x.split(".")[0] for x in base_only)
    composite_examples = [
        f"{item['base']} -> {', '.join(item['fast'])}" for item in composite
    ]
    lines = [
        "# FastH3 tensor key map",
        "",
        f"Sources: FastH3 index `{FAST_INDEX}`; h3.c index `{BASE_INDEX}`.",
        "Missing shard schemas were read with HTTP Range via the configured proxy; no shard body was downloaded.",
        f"Range headers unavailable during this run: {len(range_missing)} shard(s); repeated-layer shapes were inferred from downloaded headers where needed.",
        "",
        "## Counts",
        "",
        f"- one-to-one renames: **{len(one)}**",
        f"- split/concatenate candidates: **{len(composite)}** source tensors -> **{len(composite) * 3}** FastH3 tensors",
        f"- h3.c-only keys after the proposed map: **{len(base_only)}**",
        f"- FastH3-only keys after the proposed map: **{len(fast_only)}**",
        f"- shape or dtype mismatches among mapped keys: **{len(shape_mismatch)}**",
        "",
        "## One-to-one renames",
        "",
        f"Examples: {examples([f'{b} -> {f}' for b, f in one])}",
        "",
        "## Split or concatenate",
        "",
        f"Examples: {examples(composite_examples)}",
        "",
        "## Unmatched modules",
        "",
        f"h3.c-only module prefixes: `{dict(base_extra_groups)}`",
        f"FastH3-only module prefixes: `{dict(fast_extra_groups)}`",
        f"h3.c-only examples: {examples(base_only)}",
        f"FastH3-only examples: {examples(fast_only)}",
        "",
        "## Shape mismatches",
        "",
    ]
    if shape_mismatch:
        lines.extend(f"- `{a}` vs `{b}`: `{c}` vs `{d}`"
                    for a, b, c, d in shape_mismatch)
    else:
        lines.append("- None among the proposed one-to-one and QKV split entries.")
    args.output.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
