# MiniMax-M3.1-Flash on two DGX Sparks, dense FP4 (TensorFold native CUDA)

Serves [MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash), a 428B-A23B mixture of experts with
block-sparse attention and a vision tower, across two NVIDIA DGX Spark (GB10, 128 GB) nodes at tensor-parallel 2 with
TensorFold's native CUDA engine: an OpenAI-compatible server with images, tool calls, structured output, thinking on or
off, the release's own DSpark drafter, a prompt cache and three streams.

This is the speed branch (`fp4-dense`) of the recipe. The `master` branch serves the release's projections and head as
published; this one converts them to narrower formats for 12-32% faster decode, at a measured cost in fidelity
([the dense FP4 conversion](#the-dense-fp4-conversion)). `dense_fp4 = false` in `recipe.toml` converts the `master`
recipe's checkpoint instead.

The release does not fit two Sparks whole: its experts alone are 216.4 GiB, and the two ranks have about 220 GiB between
them after the system's share. So 15 of the 57 MoE layers keep their NVFP4 experts exactly as released, and the other 42
take a 3-bit affine refit of the same NVFP4 values (groups of 64), as on `master`. The 15 layers are the ones whose
experts lose most at 3 bits, ranked on captured activations ([`tools/requant_study.json`](tools/requant_study.json); the
engine's `tools/zig/minimax_m3_requant_study.py` rebuilds it): 20-29, 31, 49, 52, 57 and 59. `recipe.toml`'s
`nvfp4_layers` chooses them.

On top of that the conversion turns the release's MXFP8 projections (attention q, k, v, o and the index projections, the
shared experts, the three dense layers' MLPs) into NVFP4, and its bf16 head into MXFP8. Each NVFP4 block's e4m3 scale is
the one, among its largest value's code, the next above and six below, that leaves the least squared error; projections
the engine stacks into one matmul share one global scale. A token's decode reads about a third fewer bytes. The model
comes to 194 GiB and leaves each rank room for three 32k-token streams, the prompt cache and the drafter.

| part | format | GiB, both ranks |
|---|---|---:|
| routed experts, 15 layers | NVFP4 as released (e2m1, e4m3 scales a 16, global scale) | 57.0 |
| routed experts, 42 layers | 3-bit affine refit of the NVFP4 values (g64, bf16 scales and biases) | 124.0 |
| attention, shared experts, dense layers | NVFP4 from the release's MXFP8 (e2m1, e4m3 scales a 16, global scale) | 5.5 |
| head | MXFP8 from the release's bf16 (e4m3, e8m0 scale a 32) | 1.2 |
| embeddings | bf16 as released | 2.3 |
| DSpark drafter, vision tower | MXFP8 and bf16 as released | 3.4 |

The engine's `minimax_m3` family has a CUDA backend ([ashhart/TensorFold#1388](https://github.com/ashhart/TensorFold/pull/1388)),
beside the Metal one of [ashhart/TensorFold#1235](https://github.com/ashhart/TensorFold/pull/1235); loading the converted
projections and head, and drafting several streams' blocks in one pass, come on top of it ([#1426](https://github.com/ashhart/TensorFold/pull/1426) and
[#1427](https://github.com/ashhart/TensorFold/pull/1427), commit `bcbb2e344`). Until they merge, `spark build` builds the pinned
commit of their branch (`minimax-m3-fp4` on [BobClawblaw/TensorFold](https://github.com/BobClawblaw/TensorFold)).

## Quick start

Needs two DGX Sparks (GB10, 128 GB each) joined by their ConnectX-7 link, CUDA 13, Python 3.11+, git, `hf`,
[Zig 0.17.0](https://ziglang.org/download/) and passwordless ssh from the head node to the worker. About 210 GB of
disk on each node for the converted checkpoint (and 235 GB more on the head for the release while converting).

```bash
cp local.example.toml local.toml   # the worker's ssh login, the link's addresses, zig's path
python3 -m spark build             # the pinned engine and libtfgrammar (structured output, forced tool calls)
python3 -m spark download          # the release at its pinned revision, on the head node
python3 -m spark convert           # the checkpoint (about 90 minutes, most of it the 3-bit refit), checked, copied to the worker
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
| `parallel` | 3 | streams decoded together |
| `drafts` | true | the release's DSpark head drafts each stream's next block; replies are identical with drafts off |
| `vision`, `vision_image_tokens` | true, 4096 | images a request may hold, their tokens in all (an image takes at most 576) |
| `prompt_cache_gib` | 3 | kept prompt states, at 2,048-token chunk starts |
| `thinking` | false | the default for requests; `chat_template_kwargs: {"enable_thinking": true}` turns it on for one |
| `reserve_gib` | 3 | memory each rank leaves the system |
| `dense_fp4` | true | convert the projections to NVFP4 and the head to MXFP8 (convert again after changing it) |

## What works, and how it was checked

Every number here comes from [recipe-harness](https://github.com/BobClawblaw/recipe-harness) runs of the target
`minimax-m31-nvfp4mix-dense4-gb10x2`, unless a row says otherwise. They were taken on a binary (sha256 `09ac4a25...`)
built from the pinned commit's code before its last formatting and comment edits (`zig fmt`, one-line comments); the
pinned commit itself builds `6298e604...`, with the same code.

| Check (harness quick plan, run `20261026-192739-quick`) | Result |
|---|---|
| soak, concurrent, streaming, multi-turn, stop, max-tokens, copy | 60/60, 12/12, 4/4, 8/8, 3/3, 3/3, 2/2 |
| thinking on/off switch | 10/10 |
| DSpark drafts: drafted reply == plain reply | 12/12 |
| structured output (JSON schema, regex), drafted == plain | 18/18, 12/12; a bad regex gets 400 |
| tool calls | 5/5 |
| seeded sampling repeats, and equal with drafts off | 9/9 |
| prompt cache: a resumed turn == the same turn cold after a restart | yes (4096 of 4826 tokens restored) |
| memory at three streams, idle and busy | 11.1-11.5 GiB available on the head, 10.3-10.7 on the worker |
| image URLs | refused with 400 (off by design; images go in as data URLs) |

Quality with thinking off (harness plan `minimax-quality-off`, run `20261026-200035-minimax-quality-off`, three
streams), beside the `master` recipe's release formats on the same pair and the M5 Ultra MLX recipe of this model, with
the same suite checkout:

| Suite (cases) | This recipe | `master` | M5 Ultra (two runs) |
|---|---:|---:|---:|
| tools (60) | 0.950 | 0.950 | 0.933 / 0.950 |
| GSM8K (250) | 0.968 | 0.972 | 0.976 / 0.968 |
| MGSM (240) | 0.817 | 0.838 | 0.821 / 0.846 |
| MMLU (342) | 0.819 | 0.836 | 0.819 / 0.827 |
| IFEval (150) | 0.620 | 0.587 | 0.593 / 0.580 |
| JSON schema (30) | 1.000 | 1.000 | 1.000 |

Each gap to `master` is within one standard error of the suite's size (about 0.025 for MGSM, 0.021 for MMLU, 0.04 for
IFEval). MATH-500 was not run (it alone takes more than four hours here).

Quality with thinking on (harness plan `minimax-thinking`, run `20261026-215020-minimax-thinking`, server started with
`THINKING_BUDGET=28672`, three requests at a time): the harness's thinking problems 10/10 right, every reply with its
reasoning apart, none cut at 29,696 tokens; GSM8K 0.968 (242/250), no reply cut. MATH-500 and GPQA with thinking on were
not run: at up to 28k reasoning tokens a question they take many hours here.

## Measured

Decode, tokens a second after the first token (harness decode ruler, DSpark drafts on, run `20261026-191004-speed3`
of target `minimax-m31-nvfp4mix-dense4-gb10x2`); in brackets the `master` recipe's release formats (target
`minimax-m31-nvfp4mix-gb10x2`, `20261026-105149-speed`):

| Streams | prose | code | structured | list | first token |
|---:|---:|---:|---:|---:|---:|
| 1 | 27.0 (23.0) | 45.2 (40.5) | 32.7 (24.8) | 36.5 (32.0) | 0.47-0.50 s |
| 2 (total) | 36.5 (29.8) | 57.6 (49.3) | 41.6 (32.6) | 65.0 (53.9) | 0.81-0.85 s |
| 3 (total) | 42.5 | 61.5 | 46.4 | 75.0 | 1.14-1.19 s |

Where a token's time goes (the release formats, one row, rank 0, synchronized section timers): routed experts 15.2 ms,
attention projections 15.7, shared experts and dense MLPs 8.3, head 5.0, attention 3.8, norms and adds 3.2, the
all-reduces 2.2, routers 2.0, 55.4 in all. The weight reads run at 85-100% of the GB10's measured read ceiling, which is
why narrower projections and head are where the speed comes from. A verify window of 8 rows takes 113 ms against 55 for
one, most of it the experts the extra rows touch.

## The dense FP4 conversion

The projections and head leave the release's formats, so the logits move. Measured against the release's own weights
(every expert NVFP4, every projection MXFP8, the head bf16) on a 2,406-token chat of prose, code, math, German, Chinese
and JSON, all 60 layers in fp32 reference code (`tools/zig/minimax_m3_kl_study.py` in the engine; corpus and results in
[`evidence/kl/`](evidence/kl/)):

| | KL to the release, nats a token | top-1 agreement |
|---|---:|---:|
| `master`'s checkpoint (the 3-bit refit on 42 layers) | 0.105 | 88.6% |
| this one (the refit, NVFP4 projections, MXFP8 head) | 0.143 | 86.9% |
| the same with the attention output (o) kept MXFP8 | 0.121 | 87.6% |

The converted folder's `mix.json` (every output file's sha256; `spark convert` checks them) is sha256
`b21fdee27b56408db9f77bb2dc22b93224960d419106db758007d00e9cf43711`.

The MXFP8 head alone adds 0.0009. The study's quantizer is written apart from the converter's: it takes one global
scale a tensor where the converter shares one across a stacked group, and it judges a block's candidate scales after
rounding to bf16 where the converter judges them in fp32. On an attention output projection they pick another scale in
13% of the blocks, near-ties, and their total squared error differs by 0.1% (the converter's lower).

## Limits

- Three streams at a 32k window: each rank has 8.3 GiB left after the weights and the 3 GiB reserve, and three streams
  take 7.7 of it (2.56 GiB each). A shorter window takes more streams or more prompt cache.
- Only the 15 listed MoE layers are the release's own experts; the other 42 are a 3-bit refit, and every projection
  and the head are converted too. These are lossy steps; the KL and quality numbers above include them.
- The conversion's NVFP4 projections and MXFP8 head are the same bytes from a CPU or a GPU; the 3-bit refit's are
  only reproduced on a CUDA GPU (a CPU's refit differs), so `mix.json`'s hashes check a CUDA conversion.
- Images come as data URLs, at most 576 tokens an image (672 x 672). Video is not served.
- Only one pair of DGX Sparks was checked.

## License

The recipe is MIT. The weights are under the MiniMax Community License (non-commercial use; see the release's LICENSE);
TensorFold is Apache-2.0.
