function [L, C] = kmeanspp_v3_fast2(X, k)
% Faster kmeanspp_v3 variant that preserves the original chunked seeding.
% Labels are assigned to columns of X, matching kmeanspp_v3.

[d, n] = size(X);
L = [];
L1 = 0;
x2 = real(sum(X .* X, 1));
chunk_size = 10000;

while length(unique(L)) ~= k
    C = zeros(d, k);
    C(:, 1) = X(:, 1 + round(rand * (n - 1)));
    L = ones(1, n);

    for i = 2:k
        centerNorm = real(sum(C(:, L) .* C(:, L), 1));
        dist2 = x2 - 2 * real(sum(X .* C(:, L), 1)) + centerNorm;
        dist2(dist2 < 0) = 0;

        D = zeros(1, n);
        for i2 = 1:chunk_size:n
            blockIdx = i2:min(i2 + chunk_size - 1, n);
            D(blockIdx) = cumsum(sqrt(dist2(blockIdx)));
        end

        if D(end) == 0
            C(:, i:k) = X(:, ones(1, k - i + 1));
            return;
        end

        C(:, i) = X(:, find(rand < D / D(end), 1));
        [~, L] = max(bsxfun(@minus, 2 * real(C(:, 1:i)' * X), real(sum(C(:, 1:i) .* C(:, 1:i), 1))'), [], 1);
    end

    while any(L ~= L1)
        L1 = L;
        counts = accumarray(L(:), 1, [k, 1])';
        Y = sparse(1:n, L, 1, n, k);
        C = X * Y;
        valid = counts > 0;
        C(:, valid) = bsxfun(@rdivide, C(:, valid), counts(valid));
        [~, L] = max(bsxfun(@minus, 2 * real(C' * X), real(sum(C .* C, 1))'), [], 1);
    end
end
end
