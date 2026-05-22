# Unified-MMD-Stat
Official implementation of the paper:

**Unified Unbiased Variance Estimation for Maximum Mean Discrepancy: Robust Finite-Sample Performance with Imbalanced Data and Exact Acceleration under Null and Alternative Hypotheses**  

[![arXiv](https://img.shields.io/badge/arXiv-2601.13874-b31b1b.svg)](https://arxiv.org/abs/2601.13874)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 1. Overview
This toolbox provides a unified framework for unbiased estimation of **Maximum Mean Discrepancy (MMD)** variance. It addresses key challenges in kernel-based hypothesis testing:

*   **Unbiasedness**: Reliable performance under both the Null Hypothesis ($\mathcal{H}_0$) and Alternative Hypothesis ($\mathcal{H}_1$).
*   **Imbalanced Data**: Robust performance even when sample sizes are unequal ($n \neq m$).
*   **Exact Acceleration**: Reduces complexity from $O(n^2)$ to **$O(n \log n)$** for univariate Laplacian kernels.

## 2. Core Algorithms
*   **`MMD_propose.m`**: The proposed unbiased variance estimator.
*   **`euMMD.m`**: The exact acceleration engine. It utilizes linear prefix sums and sorting to achieve quasi-linear time complexity without any numerical approximation.

## 3. Numerical Experiments
Reproduce Section 4 of the paper:

### 3.1. Variance analysis under heterogeneous settings (Section 4.1)
*   **`sec1_caseA_null_balanced.m`**: The Null hypothesis, balanced samples (Table 1).
*   **`sec1_caseB_alt_balanced.m`**: The Alternative hypothesis, balanced samples (Table 2).
*   **`sec1_caseC_null_unbalanced.m`**: The Null hypothesis, unbalanced samples (Table 3).
*   **`sec1_caseD_alt_unbalanced.m`**: The Alternative hypothesis, unbalanced samples (Table 4).

### 3.2. Properties of the estimator (Section 4.2)
*   **`sec2_property1.m`**: Verifies the exact algebraic equivalence of the statistics (Figure 2).
*   **`sec2_property2.m`**: Tracks the empirical variance decay profiles (Figure 3).

### 3.3. Accuracy and efficiency of Laplacian-based accelerated euMMD (Section 4.3)
*   **`sec3_compare_accuracy.m`**: Compares MMD and euMMD (Table 5).
*   **`sec3_step1_benchmark_time_efficiency.m`**: Measures execution time scaling.
*   **`sec3_step2_benchmark_space_eummd.m`**: Measures peak memory for the $O(n \log n)$ `euMMD`.
*   **`sec3_step3_benchmark_space_matrix.m`**: Measures peak memory for the $O(n^2)$ matrix-form.
*   **`sec3_step4_plot_efficiency.m`**: Generates efficiency plots (Figure 4).

## 4. Baselines for Comparison
This toolbox includes implementations of state-of-the-art MMD variance estimators:

*   **`MMD_gretton.m`**: [Gretton et al. (2012)](https://www.jmlr.org/papers/v13/gretton12a.html)
*   **`MMD_sutherland.m`**: [Sutherland & Deka (2019)](https://arxiv.org/abs/1906.02104)
*   **`MMD_wei.m`**: [Wei et al. (2025)](https://arxiv.org/abs/2512.13997)

## Citation

If you find this code or paper useful, please cite:

```bibtex
@article{zhong2026unified,
  title={Unified Unbiased Variance Estimation for Maximum Mean Discrepancy: Robust Finite-Sample Performance with Imbalanced Data and Exact Acceleration under Null and Alternative Hypotheses},
  author={Zhong, Shijie and Yang, Yikun and Gong, Da and Fu, Jiangfeng},
  journal={arXiv preprint arXiv:2601.13874},
  year={2026}
}
