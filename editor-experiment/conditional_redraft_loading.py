"""Pinned local-only loading for the conditional-redrafting reference.

There are no implicit network downloads or cloud launches. Weights must already
be staged; full file hashes are checked before constructing a released model.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import types

SOURCES = Path(__file__).resolve().parents[1] / "notes/drafter-branching-research-2026-09-17/sources"
SOURCE_HASHES = {
    "zlab_4b_b16__utils.py": "00b91431932c94b87443d9e2c0cf4d743846bd6c9666c9c736008df7bbd0faf1",
    "zlab_4b_b16__dflash.py": "17ab761ba665a41965b27bdc3a1c6795d1b07b285ec1f0789ad146f2ee9c1cc5",
    "SpeculatorsRedHat__src__speculators__models__dflash__model_definitions.py": "da8a7a0d3ea5e37256c14def4af12003de30a44d138ff45122f9a9fa7bf04039",
    "SpeculatorsRedHat__src__speculators__models__dflash2__model_definitions.py": "b393adc0fa89d3240ea8ccc23493324335f06c73805b3409a7206108b4e86299",
}
TARGET_CONFIG_SHA = "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
DRAFT_CONFIG_SHA = "2de97cdf48f429628074d229d024753f1ea02b575732983d6ed2bc882bbd63c8"
TARGET_REPO = "Qwen/Qwen3-4B"
DRAFT_REPO = "z-lab/Qwen3-4B-DFlash-b16"


def check_file(path: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"Pinned file hash mismatch: {path.name}")


def _package(name):
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(SOURCES)]
        sys.modules[name] = package


def _source_module(name, filename, *, imports=None):
    path = SOURCES / filename
    check_file(path, SOURCE_HASHES[filename])
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Only relocate a pinned component's import to a private namespace. No
    # function/class body is changed, and an installed speculators is untouched.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and imports and node.module in imports:
            node.module = imports[node.module]
    sys.modules[name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise
    return module


def native_dflash_module():
    prefix = "_conditional_native_dflash_b74e3a3"
    _package(prefix)
    _source_module(prefix + ".utils", "zlab_4b_b16__utils.py")
    return _source_module(prefix + ".dflash", "zlab_4b_b16__dflash.py")


def native_dflash2_components():
    """Pinned training-release layer/selector definitions, not a vLLM runtime."""
    prefix = "_conditional_native_speculators_0a1b3e0"
    _package(prefix)
    base_name = prefix + ".dflash_layers"
    _source_module(base_name, "SpeculatorsRedHat__src__speculators__models__dflash__model_definitions.py")
    return _source_module(
        prefix + ".dflash2_layers",
        "SpeculatorsRedHat__src__speculators__models__dflash2__model_definitions.py",
        imports={"speculators.models.dflash.model_definitions": base_name},
    )


def verify_staged_release(folder, repo):
    """Verify full safetensors files against pinned public LFS identities."""
    folder = Path(folder)
    manifest = json.loads((SOURCES / "conditional-checkpoint-headers.json").read_text(encoding="utf-8"))
    files = [f for f in manifest["files"] if f["repo"] == repo]
    if not files:
        raise ValueError("Release is absent from the pinned header manifest")
    for entry in files:
        path = folder / entry["file"]
        if path.stat().st_size != entry["full_file_bytes"]:
            raise ValueError(f"Wrong checkpoint file size: {path.name}")
        check_file(path, entry["full_file_sha256_expected"])
    return files


def from_local_pretrained(model_class, folder, *, device, dtype, attention_backend):
    """Shared native load path; fail on silent partial/mismatched state loading."""
    model, information = model_class.from_pretrained(
        folder, local_files_only=True, dtype=dtype,
        attn_implementation=attention_backend, use_safetensors=True,
        output_loading_info=True,
    )
    errors = {name: information.get(name) for name in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if information.get(name)}
    if errors:
        raise ValueError(f"Native checkpoint did not load completely: {errors}")
    return model.to(device).eval().requires_grad_(False)


def load_target(folder, *, device, dtype, attention_backend="sdpa"):
    from transformers import AutoModelForCausalLM
    folder = Path(folder)
    check_file(folder / "config.json", TARGET_CONFIG_SHA)
    check_file(folder / "model.safetensors.index.json",
               "6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2")
    verify_staged_release(folder, TARGET_REPO)
    # No device_map/Accelerate dependency and no trust_remote_code execution.
    return from_local_pretrained(
        AutoModelForCausalLM, folder, device=device, dtype=dtype,
        attention_backend=attention_backend,
    )


def load_ordinary_drafter(folder, *, device, dtype, attention_backend="sdpa"):
    folder = Path(folder)
    check_file(folder / "config.json", DRAFT_CONFIG_SHA)
    verify_staged_release(folder, DRAFT_REPO)
    native = native_dflash_module()
    return from_local_pretrained(
        native.DFlashDraftModel, folder, device=device, dtype=dtype,
        attention_backend=attention_backend,
    )
