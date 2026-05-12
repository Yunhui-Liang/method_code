function newS = computePTS_II_fast(S, paraT)
% Lightly optimized computePTS_II. Keeps the same random-walk formulation.

N = size(S, 1);
S(1:N+1:end) = 0;

rowSum = sum(S, 2);
valid = rowSum ~= 0;
P = zeros(N, N);
P(valid, :) = bsxfun(@rdivide, S(valid, :), rowSum(valid));

tmpP = P;
inProdP = P * P';
for ii = 1:(paraT - 1)
    tmpP = tmpP * P;
    inProdP = inProdP + tmpP * tmpP';
end

d = diag(inProdP);
newS = bsxfun(@rdivide, inProdP, sqrt(d * d'));
newS(~isfinite(newS)) = 0;

isolatedIdx = find(sum(P, 2) < 1e-9);
if ~isempty(isolatedIdx)
    newS(isolatedIdx, :) = 0;
    newS(:, isolatedIdx) = 0;
end
end
