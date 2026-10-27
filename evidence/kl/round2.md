# MiniMax-M3.1-Flash requant KL study (60 layers, 2406 tokens, tail = last 1000 rows)

KL in nats (mean over rows, full vocab, fp32); top-1 = argmax agreement with `exact`.

| variant | KL(exact‖v) all | KL(exact‖v) tail | top-1 % all | top-1 % tail | KL(served‖v) all | KL(served‖v) tail | p99 KL all |
|---|---|---|---|---|---|---|---|
| exact | 0.00000 | 0.00000 | 100.00 | 100.00 | - | - | 0.0000 |
| served | 0.10545 | 0.12043 | 88.57 | 88.90 | - | - | 1.0809 |
| served+head8 | 0.10621 | 0.12155 | 88.20 | 88.40 | 0.00093 | 0.00077 | 1.0923 |
| served+head8+q4 | 0.10173 | 0.11229 | 87.99 | 88.90 | 0.02460 | 0.02613 | 1.0109 |
| served+head8+o4 | 0.12723 | 0.13649 | 87.82 | 88.40 | 0.04805 | 0.05491 | 1.3817 |
| served+head8+gu4 | 0.10526 | 0.11635 | 88.24 | 87.80 | 0.02496 | 0.02344 | 1.1978 |
| served+head8+down4 | 0.11628 | 0.12920 | 88.03 | 88.20 | 0.02877 | 0.02692 | 1.2735 |
| served+head8+mlp4 | 0.12620 | 0.13917 | 87.78 | 87.80 | 0.03899 | 0.03612 | 1.4847 |
| served+head8+qo4 | 0.12905 | 0.13174 | 87.32 | 88.10 | 0.05121 | 0.05231 | 1.4906 |
| served+head8+qo4+mlp4 | 0.14427 | 0.16050 | 86.87 | 87.70 | 0.06226 | 0.06712 | 1.7515 |
| served+head8+qo4+mlp4(plain) | 0.15669 | 0.17379 | 86.82 | 87.80 | 0.08078 | 0.08259 | 1.9096 |

wall time 16.2 min
