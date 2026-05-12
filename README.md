# FKCC Shared-Center Gini

Main entry:

```matlab
run_FKCC_shared_center_gini
```

Expected input data:

- Example dataset `.mat` files are included in `data/`.
- Additional datasets can also be placed in `data/`.
- Each `.mat` file should contain `BPs`, `Y`, and `g`.
- `BPs` is the base partition matrix.
- `Y` is the ground-truth label vector used only for evaluation.
- `g` is the sensitive attribute vector.

Main pipeline:

```text
run_FKCC_shared_center_gini
  -> FKCC/compute_ECA_from_BPs_fast
      -> FKCC/compute_Hc_sparse_from_BPs
      -> FKCC/simxjac_fast
      -> FKCC/computePTS_II_fast
  -> FKCC/kmeanspp_v3_fast2
  -> FKCC/FKCC_shared_gini1_mex(.mexw64)
  -> lib/my_eval_y_fair_mismatch_HAB
```

The package includes the existing Windows MEX binary:

- `FKCC/FKCC_shared_gini1_mex.mexw64`

This upload folder is a compact runtime package. It does not include the C++
source or Eigen headers.

Large datasets are not included. For files larger than GitHub's normal file
limit, use Git LFS or provide an external download link.
