function [Hc_new, clsSimRW, timing] = compute_ECA_from_BPs_fast(BPi, t_walk)
% Memory-aware ECA path for large datasets.
% It avoids the dense Hc allocation before the final Hc_new features.

timing = struct();

t0 = tic;
[Hc_sparse, ~] = compute_Hc_sparse_from_BPs(BPi);
timing.t_sparse_Hc = toc(t0);

t0 = tic;
baseClsSegs = Hc_sparse';
clsSim = simxjac_fast(baseClsSegs);
timing.t_simxjac = toc(t0);

t0 = tic;
clsSimRW = computePTS_II_fast(clsSim, t_walk);
timing.t_pts = toc(t0);

t0 = tic;
Hc_new = Hc_sparse * clsSimRW;
timing.t_hc_new = toc(t0);
end
