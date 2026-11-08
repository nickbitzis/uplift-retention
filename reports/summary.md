# Study summary

Champion uplift model (best validation Qini): `x_learner`

## Policies at 20% of the test set (discount offer)

| Policy | Targeted | Cost | Incremental retained | Net value | Persuadables | Lost causes | Sleeping dogs |
|---|---:|---:|---:|---:|---:|---:|---:|
| Highest churn risk | 8,026 | $92,620 | 1,197 | $79,021 | 1,621 | 1,870 | 424 |
| Random | 8,026 | $92,492 | 274 | $-53,322 | 687 | 856 | 413 |
| Uplift: T-learner | 8,026 | $101,339 | 1,536 | $123,783 | 1,793 | 1,121 | 257 |
| Uplift: S-learner | 8,026 | $103,044 | 1,607 | $131,540 | 1,884 | 1,016 | 277 |
| Uplift: X-learner | 8,026 | $102,793 | 1,615 | $133,190 | 1,880 | 1,020 | 265 |
| Uplift: class transformation | 8,026 | $102,453 | 1,514 | $118,553 | 1,787 | 961 | 273 |
| Oracle (true persuadables only) | 3,445 | $42,353 | 3,445 | $468,659 | 3,445 | 0 | 0 |

## Qini (test)

| Model | Qini coefficient | Uplift AUC | True gain AUC |
|---|---:|---:|---:|
| Highest churn risk | 320.8 | 1,201.1 | 644.3 |
| Uplift: T-learner | 507.4 | 1,565.3 | 1,031.0 |
| Uplift: S-learner | 539.4 | 1,629.5 | 1,104.8 |
| Uplift: X-learner | 536.9 | 1,626.9 | 1,104.0 |
| Uplift: class transformation | 471.7 | 1,497.4 | 993.5 |
| Random | -10.7 | 547.0 | 4.0 |

## Budget $40,000

| Allocation | Targeted | Cost | Expected net | True net | Discount | Free month | Downgrade |
|---|---:|---:|---:|---:|---:|---:|---:|
| Greedy, three offers, value per dollar | 3,664 | $40,000 | $125,674 | $100,235 | 2,748 | 338 | 578 |
| Greedy, discount only | 3,110 | $39,999 | $113,995 | $96,829 | 3,110 | 0 | 0 |
| Fixed top-k by X-learner, discount | 3,122 | $39,993 | $103,029 | $92,151 | 3,122 | 0 | 0 |
| Fixed top-k by churn risk, discount | 3,453 | $39,990 | $72,550 | $65,685 | 3,453 | 0 | 0 |

LP relaxation bound: $125,675 (greedy reaches 100.0%).
