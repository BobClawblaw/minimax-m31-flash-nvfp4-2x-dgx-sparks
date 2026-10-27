# MiniMax-M3.1-Flash requant KL study (60 layers, 2406 tokens, tail = last 1000 rows)

KL in nats (mean over rows, full vocab, fp32); top-1 = argmax agreement with `exact`.

| variant | KL(exact‖v) all | KL(exact‖v) tail | top-1 % all | top-1 % tail | KL(served‖v) all | KL(served‖v) tail | p99 KL all |
|---|---|---|---|---|---|---|---|
| exact | 0.00000 | 0.00000 | 100.00 | 100.00 | - | - | 0.0000 |
| served | 0.10545 | 0.12043 | 88.57 | 88.90 | - | - | 1.0809 |
| served+head8 | 0.10621 | 0.12155 | 88.20 | 88.40 | 0.00093 | 0.00077 | 1.0923 |
| served+head4 | 0.11102 | 0.12286 | 88.11 | 88.80 | 0.00728 | 0.00529 | 1.1375 |
| served+qo4 | 0.13891 | 0.14578 | 86.95 | 87.40 | 0.06289 | 0.05988 | 1.6996 |
| served+mlp4 | 0.12104 | 0.13505 | 87.61 | 88.20 | 0.04141 | 0.04171 | 1.2662 |
| served+kvi4 | 0.11230 | 0.13023 | 87.86 | 87.90 | 0.02744 | 0.02902 | 1.2122 |
| served+dense4 | 0.15947 | 0.17163 | 86.95 | 87.90 | 0.09135 | 0.08961 | 1.8556 |
| all3 | 0.14109 | 0.14791 | 85.91 | 87.20 | 0.06531 | 0.05999 | 1.5297 |

wall time 20.4 min
