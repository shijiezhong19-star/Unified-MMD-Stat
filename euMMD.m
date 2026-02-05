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
%   result.sigma2   - Total unified variance (sigma2_t1 + sigma2_t2).
%   result.rho1     - First-order variance contribution ratio.
    
    n = size(X,1);
    m = size(Y,1);
    sigma = param;

    % 1. Sorting (Fundamental for O(n log n) efficiency)
    x_s = sort(X);
    y_s = sort(Y);

    % 2. Intra-sample Accumulators (Algorithm 1)
    % Compute row sums for Kxx and Kyy in linear time
    [R_x, L_x] = PrefixSuffix(x_s, sigma);
    [R_y, L_y] = PrefixSuffix(y_s, sigma);

    % 3. Inter-sample Accumulators (Algorithm 2)
    % Compute row sums for Kxy and Kyx in O(n+m) time
    [A_xy, Z_xy, A_yx, Z_yx] = CrossPrefixSuffix(x_s, y_s, sigma);

    % 4. Exact Unbiased MMD^2 Computation
    % Utilizing row sum identity: sum(K_ij) = R_i + L_i
    sum_Kxx = sum(R_x + L_x);
    sum_Kyy = sum(R_y + L_y);
    sum_Kxy = sum(A_xy + Z_xy);

    A = sum_Kxx / (n * (n - 1));
    B = sum_Kyy / (m * (m - 1));
    C = sum_Kxy / (n * m);
    MMD2 = A + B - 2*C;

    % 5. First-order Variance (sigma2_t1)
    % U and V represent the influence functions from Hoeffding decomposition
    U = (R_x + L_x) / (n - 1) - (A_xy + Z_xy) / m;
    V = (R_y + L_y) / (m - 1) - (A_yx + Z_yx) / n;
    
    sigma2_t1 = 4*(n-2)/(n*(n-1)) * var(U) + 4*(m-2)/(m*(m-1)) * var(V);

    % 6. Second-order Variance (Algorithm 3 & 4)
    % Quadratic kernel statistics: exp(-|x-y|/sigma)^2 = exp(-|x-y|/(sigma/2))
    sigma_sq = sigma / 2;
    
    [R_x_sq, L_x_sq] = PrefixSuffix(x_s, sigma_sq);
    [R_y_sq, L_y_sq] = PrefixSuffix(y_s, sigma_sq);
    [A_xy_sq, Z_xy_sq, A_yx_sq, Z_yx_sq] = CrossPrefixSuffix(x_s, y_s, sigma_sq);

    % Extract Frobenius norms and squared row-sum norms efficiently
    stats = get_second_order_stats(R_x, L_x, R_y, L_y, ...
                                   A_xy, Z_xy, A_yx, Z_yx, ...
                                   R_x_sq, L_x_sq, R_y_sq, L_y_sq, ...
                                   A_xy_sq, Z_xy_sq, A_yx_sq, Z_yx_sq);

    % Exact combinatorial assembly for unbiasedness
    n2=n*(n-1); n3=n2*(n-2); n4=n3*(n-3);
    m2=m*(m-1); m3=m2*(m-2); m4=m3*(m-3);

    EgA2 = (1/n2)*stats.normKxxF2 - (2/n3)*(stats.normKxx1_2 - stats.normKxxF2) ...
         + (1/n4)*(sum_Kxx^2 - 4*stats.normKxx1_2 + 2*stats.normKxxF2);
         
    EgB2 = (1/m2)*stats.normKyyF2 - (2/m3)*(stats.normKyy1_2 - stats.normKyyF2) ...
         + (1/m4)*(sum_Kyy^2 - 4*stats.normKyy1_2 + 2*stats.normKyyF2);
         
    EgC2 = (1/(n*m))*stats.normKxyF2 - (1/(n*m2))*(stats.normKyx1_2 - stats.normKxyF2) ...
         - (1/(m*n2))*(stats.normKxy1_2 - stats.normKxyF2) ...
         + (1/(n2*m2))*(sum_Kxy^2 - stats.normKxy1_2 - stats.normKyx1_2 + stats.normKxyF2);

    sigma2_t2 = 2/(n*(n-1))*EgA2 + 2/(m*(m-1))*EgB2 + 4/(n*m)*EgC2;

    % 7. Output Result
    result.MMD2 = MMD2;
    result.sigma2 = sigma2_t1 + sigma2_t2;
    result.sigma2_1 = sigma2_t1;
    result.sigma2_2 = sigma2_t2;
    result.rho1 = sigma2_t1/(sigma2_t1+ sigma2_t2);
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
    
    A_xy = A_yx_full(origin == 1); 
    Z_xy = Z_yx_full(origin == 1);
    
    A_yx = A_xy_full(origin == 2); 
    Z_yx = Z_xy_full(origin == 2);
end

%% Helper: Algorithm 3 (Statistics Extraction)
function stats = get_second_order_stats(Rx, Lx, Ry, Ly, Axy, Zxy, Ayx, Zyx, ...
                                        Rx_sq, Lx_sq, Ry_sq, Ly_sq, ...
                                        Axysq, Zxysq, Ayxsq, Zyxsq)
    
    % Compute norms via row-sum reduction identities
    stats.normKxx1_2 = sum((Rx + Lx).^2);
    stats.normKyy1_2 = sum((Ry + Ly).^2);
    stats.normKxxF2 = sum(Rx_sq + Lx_sq); 
    stats.normKyyF2 = sum(Ry_sq + Ly_sq);
    stats.normKxy1_2 = sum((Axy + Zxy).^2);
    stats.normKyx1_2 = sum((Ayx + Zyx).^2);
    stats.normKxyF2 = (sum(Axysq + Zxysq) + sum(Ayxsq + Zyxsq)) / 2;
end