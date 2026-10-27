| Check | Passed | Score | Time | Note |
|---|---:|---:|---:|---|
| drafts | 12/12 | 100.0% | 130s | drafted 30 tok/s vs undrafted 21 tok/s |
| streaming | 4/4 | 100.0% | 31s | identical |
| multiturn | 8/8 | 100.0% | 32s | recalled 8/8 facts |
| stop | 3/3 | 100.0% | 2s | 3/3 stopped before the string |
| thinking | 10/10 | 100.0% | 28s | 0 replies without reasoning_content, 0 with think tags in content |
| copy | 2/2 | 100.0% | 3s | 2/2 byte-identical |
| max-tokens | 3/3 | 100.0% | 3s | 3/3 exact |
| tools | 57/60 | 95.0% | 154s | "How do you say 'where is the station' in": got [] |
| json | 30/30 | 100.0% | 31s | all valid |
| ifeval | 93/150 | 62.0% | 2515s | loose 93/150, strict 81/150, 150 covered prompts |
| gsm8k | 242/250 | 96.8% | 472s | misses at [12, 73, 75, 89, 119, 157, 162, 187] |
| mgsm | 196/240 | 81.7% | 838s | de 26/30, es 28/30, fr 25/30, ja 24/30, zh 27/30, ru 27/30, sw 17/30, bn 22/30 |
| mmlu | 280/342 | 81.9% | 946s | 57 subjects x 6; weakest high_school_physics 2/6, business_ethics 3/6, college_chemistry 3/6 |
