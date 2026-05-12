function [Hc_sparse, bcs] = compute_Hc_sparse_from_BPs(BPi)
% Build the ensemble-cluster indicator matrix as sparse without creating
% the dense N-by-M Hc used by compute_Hc.

[N, nBase] = size(BPi);
nClsOrig = max(BPi, [], 1);
offset = [0, cumsum(nClsOrig(1:end-1))];
M = sum(nClsOrig);

bcs = bsxfun(@plus, BPi, offset);
rows = repmat((1:N)', nBase, 1);
cols = bcs(:);
Hc_sparse = sparse(rows, cols, true, N, M);
end
