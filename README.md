# Unified-MMD-Stat
Official implementation of the paper:

**Unified Unbiased Variance Estimation for Maximum Mean Discrepancy: Robust Finite-Sample Performance with Imbalanced Data and Exact Acceleration under Null and Alternative Hypotheses**  

[![arXiv](https://img.shields.io/badge/arXiv-2601.13874-b31b1b.svg)](https://arxiv.org/abs/2601.13874)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview
This repository provides a unified framework for **Maximum Mean Discrepancy (MMD)** variance estimation. It addresses key challenges in kernel-based hypothesis testing:

**Unified Characterization**: Works under both null ($\mathcal{H}_0$) and alternative ($\mathcal{H}_1$) hypotheses.

**Imbalanced Data**: Robust performance even when sample sizes are unequal ($n \neq m$).

**Exact Acceleration**: Reduces complexity from $O(n^2)$ to **$O(n \log n)$** for univariate Laplacian kernels.

## Core Algorithms
**`MMD_propose.m`**: The proposed **unbiased variance estimator**. It implements the full Hoeffding decomposition for finite-sample reliability.

**`euMMD.m`**: The **exact acceleration engine**. It utilizes linear prefix sums and sorting to achieve quasi-linear time complexity without any numerical approximation.

## Numerical Experiments
Reproduce Section 5 of the paper:

### 1. Statistical Accuracy (Section 5.1)

**`CaseA_Null_Accuracy.m`**: Null hypothesis, balanced samples (Table 1).

**`CaseB_Null_Unbalanced.m`**: Null hypothesis, unbalanced samples (Table 2).

**`CaseC_Alt_Balanced.m`**: Alternative hypothesis, balanced samples (Table 3).

**`CaseD_Alt_Unbalanced.m`**: Alternative hypothesis, unbalanced samples (Table 4-6).

**`CaseE_Transition_Shift.m`**: Continuous transition from $\mathcal{H}_0$ to $\mathcal{H}_1$ (Figure 2).

### 2. Efficiency & Scalability (Section 5.2)

Profile the computational gains of the $O(n \log n)$ algorithm:

**`Compare_Accuracy_Efficiency.m`**:  Compares MMD and euMMD (Table 7).

**`benchmark_time_efficiency.m`**: Measures execution time scaling.

**`benchmark_space_matrix.m`**: Measures peak memory for $O(n^2)$ matrix-form.

**`benchmark_space_eummd.m`**: Measures peak memory for the proposed $O(n \log n)$ `euMMD`.

## Baselines for Comparison
This toolbox includes implementations of state-of-the-art MMD variance estimators:

**`MMD_gretton.m`**: [Gretton et al. (2012)](https://www.jmlr.org/papers/v13/gretton12a.html)

**`MMD_sutherland.m`**: [Sutherland & Deka (2019)](https://arxiv.org/abs/1906.02104)

**`MMD_wei.m`**: [Wei et al. (2025)](https://arxiv.org/abs/2512.13997)

## Citation

If you find this code or paper useful, please cite:

@article{zhong2026unified,
  title={Unified Unbiased Variance Estimation for Maximum Mean Discrepancy: Robust Finite-Sample 
  Performance with Imbalanced Data and Exact Acceleration under Null and Alternative Hypotheses},
  author={Zhong, Shijie and Yang, Yikun and Gong, Da and Fu, Jiangfeng},
  journal={arXiv preprint arXiv:2601.13874},
  year={2026}
}
