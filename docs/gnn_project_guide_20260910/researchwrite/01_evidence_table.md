# Evidence table

| Claim | Evidence | Boundary |
|---|---|---|
| Exact matrix teacher is numerically consistent | EXP18 reciprocity 1.32e-14, KCL residual 5.11e-11, minimum real-admittance eigenvalue 3.31e-7 | Synthetic circuit teacher |
| Typed/heterogeneous structure has only a modest present advantage | Residual edge-centric vs homogeneous: graph +4.84%, node +5.58%, edge-current -1.65% | Not a decisive win |
| Absolute hetero model improves passivity rate | 87.33% vs 76.92% homogeneous | Learned direct output still not guaranteed passive |
| Full high-dimensional inverse problem is ill-conditioned | EXP16 Jacobian condition number 1368.85; low response loss but 7.16% scale error | One synthetic setup |
| Fisher/OED helps select an identifiable low-dimensional subspace | EXP17 selected4 3.863% MAPE, lowrank4 3.201%, full8 12.593% | Local-linear synthetic Monte Carlo |
| Peak localization is not solved | Residual peak hit rate about 12--13%; 8+8 only 1.25% | Exact-index metric is strict; absolute 100% is likely trivial |

