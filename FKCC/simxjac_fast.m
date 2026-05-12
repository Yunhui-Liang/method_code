function s = simxjac_fast(a, b)
% Fast extended Jaccard similarity. For this project a is usually sparse
% binary cluster-membership data.

if nargin < 2
    b = a;
end

temp = a * b';
asquared = full(sum(a.^2, 2));
bsquared = full(sum(b.^2, 2));
denom = bsxfun(@plus, asquared, bsquared') - full(temp);
s = full(temp) ./ denom;
s(~isfinite(s)) = 0;
end
