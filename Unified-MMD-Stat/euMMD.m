function result = euMMD(X, Y, param)
% EUMMD - Fast O(n log n) calculation of unified variance for univariate MMD.
%
% This function implements the accelerated algorithms (Algorithm 1-4) as
% described in:
% Zhong, S., et al. (2026). "Unified Unbiased Variance Estimation for MMD: 
% Exact Acceleration under Null and Alternative Hypotheses."
%
% Input:
%   X, Y  : Univariate sample vectors (projections) of size n x 1 and m x 1.
%   param : Kernel bandwidth (sigma) for the Laplace-like (exponential) kernel.
%
% Output:
%   result.MMD2     - Exact unbiased MMD^2.
%   result.sigma2   - Total unified variance (V1 + V2).
%   result.sigma2_1 - First-order variance component (V1).
%   result.sigma2_2 - Second-order variance component (V2).
%   result.rho1     - First-order variance contribution ratio.
    
    n = size(X,1);
    m = size(Y,1);
    sigma = param;
    
    if n < 4 || m < 4
        error('Sample sizes must be at least 4 for unbiased 4th-order variance estimation.');
    end

    % 1. Sorting (Fundamental for O(n log n) efficiency)
    x_s = sort(X);
    y_s = sort(Y);

    % 2. Intra-sample Accumulators (Base Bandwidth)
    [R_x, L_x] = PrefixSuffix(x_s, sigma);
    [R_y, L_y] = PrefixSuffix(y_s, sigma);

    % 3. Inter-sample Accumulators (Base Bandwidth)
    [A_xy, Z_xy, A_yx, Z_yx] = CrossPrefixSuffix(x_s, y_s, sigma);

    % 4. Vectorized Row Sums
    Kxx1 = R_x + L_x;      % n x 1
    Kyy1 = R_y + L_y;      % m x 1
    Kxy1 = A_xy + Z_xy;    % n x 1, corresponds to K_XY * 1_m
    Kyx1 = A_yx + Z_yx;    % m x 1, corresponds to K_XY' * 1_n

    sum_Kxx = sum(Kxx1);
    sum_Kyy = sum(Kyy1);
    sum_Kxy = sum(Kxy1);   % Global cross-sum

    % 5. Exact Unbiased MMD^2 Computation
    MMD2 = sum_Kxx/(n*(n-1)) + sum_Kyy/(m*(m-1)) - 2*sum_Kxy/(n*m);

    % 6. Squared Accumulators (Half Bandwidth)
    sigma_sq = sigma / 2;
    [R_x_sq, L_x_sq] = PrefixSuffix(x_s, sigma_sq);
    [R_y_sq, L_y_sq] = PrefixSuffix(y_s, sigma_sq);
    [A_xy_sq, Z_xy_sq, ~, ~] = CrossPrefixSuffix(x_s, y_s, sigma_sq);

    % ---------------------------------------------------------------------
    % 7. Fast Sub-Estimators Extraction (O(N) traces and projections)
    % ---------------------------------------------------------------------
    % Frobenius norms
    normKxxF2 = sum(R_x_sq + L_x_sq);
    normKyyF2 = sum(R_y_sq + L_y_sq);
    normKxyF2 = sum(A_xy_sq + Z_xy_sq);

    % L2 norms of row sums
    normKxx1_2 = sum(Kxx1.^2);
    normKyy1_2 = sum(Kyy1.^2);
    normKxy1_2 = sum(Kxy1.^2);
    normKyx1_2 = sum(Kyx1.^2);

    % Mixed cross-covariance projections (Element-wise dot product of row sums)
    mix_XX_XY = sum(Kxx1 .* Kxy1); % 1_n^T K_XX K_XY 1_m
    mix_YY_YX = sum(Kyy1 .* Kyx1); % 1_m^T K_YY K_XY^T 1_n

    % Combinatorics
    n2 = n*(n-1); n3 = n2*(n-2); n4 = n3*(n-3);
    m2 = m*(m-1); m3 = m2*(m-2); m4 = m3*(m-3);

    % ---------------------------------------------------------------------
    % 8. First-Order Variance Components (nu_P and nu_Q)
    % ---------------------------------------------------------------------
    nu_P = (1/n3) * (normKxx1_2 - normKxxF2) ...
         - (2 / (m*n2)) * mix_XX_XY ...
         + (1 / (n*m2)) * (normKxy1_2 - normKxyF2) ...
         - (1/n4) * (sum_Kxx^2 - 4*normKxx1_2 + 2*normKxxF2) ...
         + (2 / (m*n3)) * (sum_Kxx * sum_Kxy - 2*mix_XX_XY) ...
         - (1 / (n2*m2)) * (sum_Kxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    nu_Q = (1/m3) * (normKyy1_2 - normKyyF2) ...
         - (2 / (n*m2)) * mix_YY_YX ...
         + (1 / (m*n2)) * (normKyx1_2 - normKxyF2) ...
         - (1/m4) * (sum_Kyy^2 - 4*normKyy1_2 + 2*normKyyF2) ...
         + (2 / (n*m3)) * (sum_Kyy * sum_Kxy - 2*mix_YY_YX) ...
         - (1 / (n2*m2)) * (sum_Kxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    V1 = max(0, (4/n) * nu_P + (4/m) * nu_Q); % Truncated at 0 for null hypothesis stability

    % ---------------------------------------------------------------------
    % 9. Second-Order Variance Components (tau_P, tau_Q, tau_PQ)
    % ---------------------------------------------------------------------
    tau_P = (1/n2) * normKxxF2 ...
          - (2/n3) * (normKxx1_2 - normKxxF2) ...
          + (1/n4) * (sum_Kxx^2 - 4*normKxx1_2 + 2*normKxxF2);

    tau_Q = (1/m2) * normKyyF2 ...
          - (2/m3) * (normKyy1_2 - normKyyF2) ...
          + (1/m4) * (sum_Kyy^2 - 4*normKyy1_2 + 2*normKyyF2);

    tau_PQ = (1/(n*m)) * normKxyF2 ...
           - (1/(n*m2)) * (normKxy1_2 - normKxyF2) ...
           - (1/(m*n2)) * (normKyx1_2 - normKxyF2) ...
           + (1/(n2*m2)) * (sum_Kxy^2 - normKyx1_2 - normKxy1_2 + normKxyF2);

    V2 = max(0, (2/n2) * tau_P + (2/m2) * tau_Q + (4/(n*m)) * tau_PQ);

    % 10. Output Result
    result.MMD2 = MMD2;
    result.sigma2 = V1 + V2;
    result.sigma2_1 = V1;
    result.sigma2_2 = V2;
    result.rho1 = V1 / (V1 + V2);
end

%% Helper: Algorithm 1 (Intra-sample Prefix/Suffix)
function [R, L] = PrefixSuffix(s, sigma)
    l = length(s);
    R = zeros(l, 1); L = zeros(l, 1);
    % Forward Scan
    for i = 2:l
        D = exp(-(s(i) - s(i-1))/sigma);
        R(i) = (R(i-1) + 1) * D;
    end
    % Backward Scan
    for i = l-1:-1:1
        D = exp(-(s(i+1) - s(i))/sigma);
        L(i) = (L(i+1) + 1) * D;
    end
end

%% Helper: Algorithm 2 (Cross-sample Merge & Sweep)
function [A_xy, Z_xy, A_yx, Z_yx] = CrossPrefixSuffix(x_s, y_s, sigma)
    n = length(x_s); m = length(y_s);
    N = n + m;
    
    % Linear Merge (Two-pointer)
    z = zeros(N, 1);
    origin = zeros(N, 1);
    i = 1; j = 1; k = 1;
    
    while i <= n && j <= m
        if x_s(i) <= y_s(j)
            z(k) = x_s(i); origin(k) = 1;
            i = i + 1;
        else
            z(k) = y_s(j); origin(k) = 2;
            j = j + 1;
        end
        k = k + 1;
    end
    while i <= n, z(k)=x_s(i); origin(k)=1; i=i+1; k=k+1; end
    while j <= m, z(k)=y_s(j); origin(k)=2; j=j+1; k=k+1; end

    % Forward and Backward sweeps for cross-contribution
    A_xy_full = zeros(N,1); 
    A_yx_full = zeros(N,1);
    
    for k = 2:N
        dist = z(k) - z(k-1);
        D = exp(-dist/sigma);
        A_xy_full(k) = D * (A_xy_full(k-1) + (origin(k-1) == 1));
        A_yx_full(k) = D * (A_yx_full(k-1) + (origin(k-1) == 2));
    end
    
    Z_xy_full = zeros(N,1); 
    Z_yx_full = zeros(N,1);
    for k = N-1:-1:1
        dist = z(k+1) - z(k);
        D = exp(-dist/sigma);
        Z_xy_full(k) = D * (Z_xy_full(k+1) + (origin(k+1) == 1));
        Z_yx_full(k) = D * (Z_yx_full(k+1) + (origin(k+1) == 2));
    end
    
    % Extract mapped sequences for origin components
    A_xy = A_yx_full(origin == 1); 
    Z_xy = Z_yx_full(origin == 1);
    
    A_yx = A_xy_full(origin == 2); 
    Z_yx = Z_xy_full(origin == 2);
end