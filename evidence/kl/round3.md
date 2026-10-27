# MiniMax-M3.1-Flash requant KL study (60 layers, 2406 tokens, tail = last 1000 rows)

KL in nats (mean over rows, full vocab, fp32); top-1 = argmax agreement with `exact`.

| variant | KL(exact‖v) all | KL(exact‖v) tail | top-1 % all | top-1 % tail | KL(served‖v) all | KL(served‖v) tail | p99 KL all |
|---|---|---|---|---|---|---|---|
| exact | 0.00000 | 0.00000 | 100.00 | 100.00 | - | - | 0.0000 |
| served | 0.10545 | 0.12043 | 88.57 | 88.90 | - | - | 1.0809 |
| served+head8 | 0.10621 | 0.12155 | 88.20 | 88.40 | 0.00093 | 0.00077 | 1.0923 |
| served+head8+all4 | 0.14318 | 0.14906 | 86.87 | 87.40 | 0.07109 | 0.07622 | 1.7651 |
| served+head8+qkv4+mlp4 | 0.12108 | 0.13141 | 87.57 | 88.00 | 0.04820 | 0.05073 | 1.5373 |

wall time 14.7 min
