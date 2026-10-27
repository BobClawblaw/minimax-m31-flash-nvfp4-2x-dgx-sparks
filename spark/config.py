"""Settings: recipe.toml's defaults, local.toml's overrides for one pair of Sparks, then environment variables in upper
case (by key name alone: CONTEXT, PARALLEL, NVFP4_LAYERS, ...)."""

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    source_repo: str = ""
    source_revision: str = ""
    served_name: str = ""
    source_dir: str = ""
    model_dir: str = ""
    nvfp4_layers: str = ""
    dense_fp4: bool = True  # convert with --dense-fp4: NVFP4 projections, MXFP8 head
    tensorfold_src: str = ""
    tensorfold_repo: str = ""
    tensorfold_branch: str = ""
    tensorfold_sha: str = ""
    binary_dir: str = ""  # a built zig-out/native to serve instead of the pinned worktree's (development)
    zig: str = "zig"
    xgrammar: str = ""
    build_dir: str = ""
    head: str = ""
    worker: str = ""
    master_addr: str = ""
    nccl_port: int = 29581
    iface: str = ""
    hca: str = ""
    nccl_channels: int = 16
    host: str = "0.0.0.0"
    port: int = 8897
    context: int = 32768
    parallel: int = 3
    drafts: bool = True
    vision: bool = True
    vision_image_tokens: int = 4096
    prompt_cache_gib: float = 3
    max_tokens: int = 32768
    thinking: bool = False
    thinking_budget: int = 0  # reasoning tokens before the answer is forced (0: the engine's default)
    reserve_gib: float = 3
    api_key: str = ""
    extra_args: str = ""

    @staticmethod
    def path(p: str) -> Path:
        return Path(os.path.expanduser(p))

    @property
    def source(self) -> Path:
        return self.path(self.source_dir or "~/models/MiniMax-M3.1-Flash")

    @property
    def model(self) -> Path:
        return self.path(self.model_dir or ("~/models/MiniMax-M3.1-Flash-nvfp4-dense4" if self.dense_fp4
                                            else "~/models/MiniMax-M3.1-Flash-nvfp4-mix"))

    @property
    def build(self) -> Path:
        return self.path(self.build_dir)

    @property
    def worktree(self) -> Path:
        return self.build / "engine" / (self.tensorfold_sha[:12] or "unpinned")

    @property
    def native(self) -> Path:
        """The built server's folder: bin/tensorfold-native and lib/libtfgrammar.so."""
        return self.path(self.binary_dir) if self.binary_dir else self.worktree / "zig-out" / "native"

    @property
    def run_dir(self) -> Path:
        """Where both nodes run the server from (the same path on each) and keep their logs."""
        return self.build / "run"

    def env(self, rank: int) -> dict[str, str]:
        e = {
            "TF_MINIMAX_RANK": str(rank),
            "TF_MINIMAX_WORLD": "2",
            "NCCL_COMM_ID": f"{self.master_addr}:{self.nccl_port}",
            "NCCL_SOCKET_IFNAME": self.iface,
            "NCCL_IB_HCA": self.hca,
            "NCCL_MAX_NCHANNELS": str(self.nccl_channels),
            "NCCL_DEBUG": "WARN",
            "TENSORFOLD_MEMORY_RESERVE_GIB": str(self.reserve_gib),
            "TENSORFOLD_NO_UPDATE_CHECK": "1",
            "HF_HUB_OFFLINE": "1",
        }
        if not self.drafts:
            e["TF_MINIMAX_DRAFTER"] = "0"
        # engine switches for both ranks, from this shell (ENGINE_ENV="TF_ROCE=0 TF_ROCE_MAX_KB=1024")
        for kv in os.environ.get("ENGINE_ENV", "").split():
            k, _, v = kv.partition("=")
            if k.startswith(("TF_", "NCCL_")):
                e[k] = v
        return e

    def args(self, rank: int) -> list[str]:
        a = ["serve", str(self.model), "--name", self.served_name, "--context", str(self.context),
             "--parallel", str(self.parallel), "--max-tokens", str(self.max_tokens),
             "--prompt-cache-gib", str(self.prompt_cache_gib), "--no-update-check",
             "--thinking" if self.thinking else "--no-thinking"]
        if self.thinking_budget:
            a += ["--thinking-budget", str(self.thinking_budget)]
        if self.vision:
            a += ["--vision", "--vision-image-tokens", str(self.vision_image_tokens)]
        if not self.drafts:
            a += ["--no-drafts"]
        if rank == 0:
            a += ["--host", self.host, "--port", str(self.port)]
            if self.api_key:
                a += ["--api-key", self.api_key]
        return a + (self.extra_args.split() if self.extra_args else [])


def _coerce(kind, raw):
    if kind is bool:
        return raw if isinstance(raw, bool) else str(raw).lower() in ("1", "true", "yes", "on")
    return kind(raw)


def load(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    merged: dict = {}
    for name in ("recipe.toml", "local.toml"):
        p = ROOT / name
        if p.exists():
            for table in tomllib.load(open(p, "rb")).values():
                merged.update(table)
    s = Settings()
    for f in fields(Settings):
        kind = type(getattr(s, f.name))
        if f.name in merged:
            setattr(s, f.name, _coerce(kind, merged[f.name]))
        if f.name.upper() in env:
            setattr(s, f.name, _coerce(kind, env[f.name.upper()]))
    return s
