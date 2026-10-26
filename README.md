# MiniMax-M3.1-Flash on two DGX Sparks (TensorFold native CUDA, NVFP4)

Serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a 428B-A23B mixture of experts with
block-sparse attention and a vision tower, across two NVIDIA DGX Spark (GB10, 128 GB) nodes at tensor-parallel 2 with
TensorFold's native CUDA engine: an OpenAI-compatible server with images, tool calls, structured output, thinking on or
off, the release's own DSpark drafter, a prompt cache and two streams.

It reads MiniMax's release in its own formats: the MXFP8 projections (attention, shared expert, dense layers) and the
NVFP4 routed experts as published, decoded by the engine's kernels with no conversion. The release does not fit two
Sparks whole: its experts alone are 216.4 GiB, and the two ranks have about 220 GiB between them after the system's share.
So 15 of the 57 MoE layers keep their NVFP4 experts exactly as released, and the other 42 take a 3-bit affine refit
of the same NVFP4 values (groups of 64), which brings the model to about 201 GiB and leaves each rank room for two
32k-token streams, the prompt cache and the drafter.

The 15 layers are the ones whose experts lose most at 3 bits, ranked on captured activations
([`tools/requant_study.json`](tools/requant_study.json); the engine's `tools/zig/minimax_m3_requant_study.py` rebuilds
it): 20-29, 31, 49, 52, 57 and 59. `recipe.toml`'s `nvfp4_layers` chooses them; a pair with more memory to spare can
keep more.

| part | format | GiB, both ranks |
|---|---|---:|
| routed experts, 15 layers | NVFP4 as released (e2m1, e4m3 scales a 16, global scale) | 57.0 |
| routed experts, 42 layers | 3-bit affine refit of the NVFP4 values (g64, bf16 scales and biases) | 124.0 |
| attention, shared expert, dense layers | MXFP8 as released (e4m3, e8m0 scale a 32) | 11.3 |
| embeddings, head | bf16 as released | 4.6 |
| DSpark drafter, vision tower | MXFP8 and bf16 as released | 3.8 |

The engine's `minimax_m3` family has a CUDA backend for this ([ashhart/TensorFold#1388](https://github.com/ashhart/TensorFold/pull/1388), commit `1d559dd44`), beside the Metal one of
[ashhart/TensorFold#1235](https://github.com/ashhart/TensorFold/pull/1235). Until it merges, `spark build` builds the
pinned commit of its branch (`minimax-m3-cuda` on [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold)).

## Quick start

Needs two DGX Sparks (GB10, 128 GB each) joined by their ConnectX-7 link, CUDA 13, Python 3.11+, git, `hf`,
[Zig 0.17.0](https://ziglang.org/download/) and passwordless ssh from the head node to the worker. About 240 GB of
disk on each node for the converted checkpoint (and 235 GB more on the head for the release while converting).

```bash
cp local.example.toml local.toml   # the worker's ssh login, the link's addresses, zig's path
python3 -m spark build             # the pinned engine and libtfgrammar (structured output, forced tool calls)
python3 -m spark download          # the release at its pinned revision, on the head node
python3 -m spark convert           # the mixed checkpoint (about 75 minutes), checked, then copied to the worker
python3 -m spark up                # rank 1 on the worker, rank 0 here: http://0.0.0.0:8897/v1
python3 -m spark chat "Explain block-sparse attention in three sentences."
```

`python3 -m spark down|status|logs` stop and inspect it.

## Settings

`recipe.toml` holds the defaults, `local.toml` overrides them, and an environment variable of the same name in upper case
overrides both (`CONTEXT=16384 python3 -m spark up`).

| setting | default | |
|---|---|---|
| `nvfp4_layers` | 20-29,31,49,52,57,59 | the MoE layers whose experts stay NVFP4 (convert again after changing it) |
| `context` | 32768 | a stream's window |
| `parallel` | 2 | streams decoded together |
| `drafts` | true | the release's DSpark head drafts each stream's next block; replies are identical with drafts off |
| `vision`, `vision_image_tokens` | true, 4096 | images a request may hold, their tokens in all (an image takes at most 576) |
| `prompt_cache_gib` | 3 | kept prompt states, at 2,048-token chunk starts |
| `thinking` | false | the default for requests; `chat_template_kwargs: {"enable_thinking": true}` turns it on for one |
| `reserve_gib` | 3 | memory each rank leaves the system |

## What works, and how it was checked

Every number here comes from [recipe-harness](https://github.com/BobClawblaw/recipe-harness) runs of the target
`minimax-m31-nvfp4mix-gb10x2`, on the binary in the PR (sha256 `a878eb001eda4020...`) unless a row says otherwise.

| Check (harness quick plan, run `20261026-060449-quick`) | Result |
|---|---|
| soak, concurrent, streaming, multi-turn, stop, max-tokens, copy | 60/60, 12/12, 4/4, 8/8, 3/3, 3/3, 2/2 |
| thinking on/off switch | 10/10 |
| DSpark drafts: drafted reply == plain reply | 12/12 |
| structured output (JSON schema, regex), drafted == plain | 18/18, 12/12; a bad regex gets 400 |
| tool calls | 5/5 |
| seeded sampling repeats, and equal with drafts off | 9/9 |
| prompt cache: a resumed turn == the same turn cold after a restart | yes (4096 of 4826 tokens restored) |
| image URLs | refused with 400 (off by design; images go in as data URLs) |

Quality with thinking off (harness `dsv41-quality-off` suites, compared with the same suite checkout on the M5 Ultra
MLX recipe of this model):

| Suite | This recipe | M5 Ultra (two runs) |
|---|---:|---:|
| tools | 0.950 | 0.933 / 0.950 |
| GSM8K | 0.972 | 0.976 / 0.968 |
| MGSM | 0.838 | 0.821 / 0.846 |
| MMLU | 0.836 | 0.819 / 0.827 |
| IFEval | 0.587 | 0.593 / 0.580 |
| JSON schema | 1.000 | 1.000 |
| MATH-500 | not run (stopped; at two streams it takes more than 4 hours) | 0.900 / 0.890 |

Quality with thinking on was not run for this release.

## Measured

Decode, tokens a second for one request after the first token (harness decode ruler, DSpark drafts on):

| Streams | prose | code | structured | list | first token | run |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 23.6 | 41.1 | 25.0 | 32.6 | 0.47-0.50 s | `20261026-060449-quick` |
| 2 (total) | 29.8 | 49.3 | 32.6 | 53.9 | 0.81-0.86 s | `20261026-105149-speed` |

Drafting averages 2.7 tokens a round at one stream with about half the drafts accepted. Weight reads run at about
84% of the GB10's measured read floor, so most of the remaining decode time is memory bandwidth.

## Limits

- Two streams at a 32k window: the two ranks have about 5 GiB each left after the weights. A shorter window takes more
  streams or more prompt cache.
- Only the 15 listed MoE layers are the release's own experts; the other 42 are a 3-bit refit, a lossy step that the
  quality numbers above include.
- Images come as data URLs, at most 576 tokens an image (672 x 672). Video is not served.
- Only one pair of DGX Sparks was checked.

## License

The recipe is MIT. The weights are under the MiniMax Community License (non-commercial use; see the release's LICENSE);
TensorFold is Apache-2.0.
