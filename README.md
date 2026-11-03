# Can a drafter learn from its own draft?

Shrinav Loka and Frank. Speculative decoding on Qwen3 with the DFlash block drafter.

Speculative decoding lets a small drafter guess a block of tokens that the large target model verifies in one pass. A second drafting pass that re-reads the first draft adds accepted tokens, but costs almost a full extra pass. This project asks whether a much smaller network can recover that gain cheaply.

## What's here

| Folder | Contents |
|---|---|
| [`editor-experiment/`](editor-experiment/README.md) | Our experiment: a 3M-parameter editor trained to imitate a full second drafting pass. Model, training, teacher, cost measurement, analysis. |
| [`results/`](results/SUMMARY.md) | One-page summary of everything tried, the D2SD paper note, and the run write-ups behind the numbers. |
| [`dflash/`](dflash/README.md) | The DFlash drafter we build on, plus Frank's token-only secondary drafter. Install and usage notes there. |
| `docs/`, `.tours/` | Code tours for reading DFlash. |

## Results so far (Qwen3-4B, measured)

| | Extra accepted positions | Extra round cost |
|---|---|---|
| Full second drafting pass | +10.2% | about +30% |
| Best 3M editor | +3.7% | +5% to +6% |

The editor needed about +6% to break even and did not reach it. The second drafter repairs 79 and damages 33 blocks on its training prompts, but 35 and 41 on unseen prompts, so the editor is imitating prompt-specific repairs. A free 24-node tree from the drafter's own scores beats every editor.

## Next

1. Measure real end-to-end speed on Qwen3-8B.
2. Scale editor training data 10x, keeping only redrafts the target verified as better.
3. If that moves the gain, train a D2SD-style second drafter for 8B.

## Setup

```
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[local]'
```

Checkpoints: `Qwen/Qwen3-4B` and `z-lab/Qwen3-4B-DFlash-b16`. The editor experiment needs a 24 GB GPU. Upstream DFlash is MIT licensed; see [`LICENSE`](LICENSE).
