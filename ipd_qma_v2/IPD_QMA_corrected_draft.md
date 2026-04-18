# IPD-QMA: Detecting Heterogeneous Treatment Effects via Quantile Meta-Analysis of Individual Participant Data

---

## Abstract

**Background.** Traditional meta-analysis relies on mean treatment effects, assuming a uniform location shift. When interventions create a location-scale shift, modifying outcome variance without changing the mean, standard methods cannot detect efficacy, potentially discarding treatments that benefit specific subgroups.

**Methods and Findings.** We introduce Individual Participant Data Quantile Meta-Analysis (IPD-QMA), a two-stage framework. Stage 1 estimates unconditional quantile treatment effects (QTEs) via within-study quantile regression. Stage 2 pools these via DerSimonian-Laird (DL) random-effects models with Hartung-Knapp-Sidik-Jonkman (HKSJ) correction. We propose a Slope Test for the QTE gradient and compare IPD-QMA against standard meta-analysis and the log SD ratio (lnVR). Monte Carlo simulations (1,000 iterations; Normal, Exponential, Lognormal, t_5 distributions) show that under a pure scale shift (variance ratio [VR] >= 2.0, mean difference = 0), standard meta-analysis correctly retains the null (rejection rate 0.0-2.8%), while IPD-QMA at the 90th percentile achieves 100% power at VR >= 2.0, K = 10 (91.3% at VR = 1.5). The Slope Test and lnVR achieve >= 99.9% power but lnVR cannot distinguish the effect's direction. Type I error was controlled (0.0-4.3%). Limitations include Monte Carlo precision (~0.7% MCSE), observational confounding in the illustrative NHANES application (K = 4 survey cycles), and the fact that different methods test different hypotheses with unequal effect sizes, limiting direct power comparisons.

**Conclusions.** IPD-QMA complements existing tools by characterizing treatment effect shape across the outcome distribution. We recommend screening with lnVR, then characterizing with IPD-QMA. An open-source Python implementation is provided.

---

## Introduction

### The limits of the average

Systematic reviews and meta-analyses represent the highest level of evidence in clinical medicine [1]. They combine data from multiple trials to estimate a treatment effect with greater precision than any single study. Most meta-analyses rely on summary statistics, specifically measures of central tendency such as the mean difference, standardized mean difference, or odds ratio, to aggregate results [2, 4].

This emphasis on the mean embodies the assumption of a location shift: that the treatment shifts the outcome distribution uniformly relative to the control group. Clinical reality is rarely uniform. Patients within a trial differ in genetic predisposition, disease severity, comorbidities, and baseline risk. These factors interact with the treatment, leading to heterogeneity of treatment effects [3].

### The location-scale problem

A particularly challenging form of heterogeneity is the location-scale shift, in which an intervention changes both the location (mean) and the spread (scale) of the outcome distribution [5]. Consider a hypothetical vasopressor studied in septic shock. For patients in mild shock, the drug may cause excessive vasoconstriction (a negative effect at the lower tail of the outcome distribution). For patients in severe septic shock, the same drug might restore perfusion pressure, preventing organ failure (a positive effect at the upper tail). Heterogeneous treatment effects of this kind have been documented in critical care trials such as the Vasopressin and Septic Shock Trial (VASST), where vasopressin showed differential effects by shock severity [21].

If the trial enrols a mix of mild and severe patients, these opposing effects may algebraically cancel. The resulting mean difference would be near zero, and a standard meta-analysis could conclude that the treatment is "ineffective" (p > 0.05). We introduce the term **hidden utility scenario** to describe this pattern: a treatment that modifies outcome variance without changing the mean, benefiting patients at one extreme of the outcome distribution while harming those at the other. This conclusion, while statistically accurate regarding the average, may prompt reconsideration of a therapy that could benefit specific subgroups [5].

### Limitations of current solutions

The current toolkit for handling heterogeneity includes subgroup analysis and meta-regression, both with limitations. Subgroup analyses often lack statistical power because they split the sample [6]. Meta-regression relies on study-level aggregates, risking the ecological fallacy [7]. Defining subgroups requires arbitrary dichotomization, resulting in information loss.

The log SD ratio (lnVR; see Methods for the precise definition) was proposed as a metric to compare outcome variability between treatment and control groups [8]. A significant lnVR flags the presence of variance heterogeneity. However, it is a non-directional screening tool: it cannot identify which patients are affected or in which direction.

### Related work

Sun and Cheung [15] proposed a quantile regression approach for aggregate-data meta-analysis of studies with heterogeneity, working with reported quantile summaries rather than individual data. Firpo [22] developed the theory of unconditional quantile treatment effects, establishing the distinction between conditional and unconditional QTEs. Our framework differs in three key respects: (1) it uses individual participant data, enabling within-study quantile regression with bootstrap-based covariance estimation across quantiles; (2) it proposes the Slope Test for formally testing whether QTEs vary across the distribution; and (3) it provides an open-source implementation with HKSJ correction for small meta-analyses.

### Contributions

The contributions of this paper are:

1. The IPD-QMA framework combining within-study quantile regression with random-effects pooling and HKSJ correction.
2. The Slope Test for formally testing whether the treatment effect gradient across quantiles differs from zero.
3. Monte Carlo validation across four distributions with analytical ground truths.
4. An open-source Python implementation (S1 File).

---

## Materials and Methods

### Overview of the framework

We use a two-stage individual participant data (IPD) approach [10]. The two-stage approach is chosen for simplicity and modularity: Stage 1 is performed independently within each study, and Stage 2 uses standard random-effects meta-analysis. This approach does not jointly model quantile-specific heterogeneity across quantiles, and a one-stage hierarchical quantile regression model (e.g., via Bayesian methods) could improve efficiency, particularly for small K, but at greater computational cost and model complexity [10]. The two-stage approach is adequate when K >= 5 and within-study sample sizes are moderate (n >= 50 per arm).

### Mathematical formulation

#### Stage 1: Within-study estimation

Consider a meta-analysis of *K* studies. For study *k* (k = 1, ..., K) with sample size *n_k*, let Y_i be the continuous outcome and T_i in {0, 1} be the treatment indicator. Quantile regression [9, 16] minimizes a weighted sum of absolute residuals. To estimate the treatment effect beta(tau) at quantile level tau in (0, 1), we solve:

**Equation 1:**
beta_hat_k(tau) = argmin_{alpha, beta} sum_{i=1}^{n_k} rho_tau(Y_i - alpha - beta * T_i)

where rho_tau is the check function [9]:

**Equation 2:**
rho_tau(u) = u * (tau - I(u < 0))

This function weights positive residuals by tau and negative residuals by (1 - tau). In the two-sample design with only a treatment indicator and no covariates, the resulting QTE is an unconditional quantile treatment effect: the difference in the tau-th quantile of the marginal outcome distribution between treatment and control groups. Extension to covariate-adjusted models would require distinguishing unconditional from conditional QTEs [22].

#### Standard error estimation via bootstrapping

Asymptotic standard errors for quantile regression can be unreliable in the small sample sizes typical of clinical trials [11]. We employ a nonparametric bootstrap: for each study *k*, we resample n_k observations with replacement to form a bootstrap sample of the same size, re-estimate beta_hat_k(tau) at all quantile levels simultaneously, and repeat for B iterations (B = 200 for simulation, B = 500 for real-data analysis). The standard deviation of the B bootstrap estimates is the standard error SE_boot_k(tau).

Computing bootstrap estimates for all quantiles simultaneously within each iteration preserves the within-study cross-quantile covariance structure. This is essential for the Slope Test (see below). While asymptotic standard errors from quantile regression are a valid alternative for interior quantiles with moderate sample sizes, the bootstrap is chosen for robustness to non-normality and at extreme quantiles (tau = 0.1, 0.9).

#### Stage 2: Between-study synthesis

We synthesize study-level estimates using the DerSimonian-Laird (DL) random-effects model [12]. The DL estimator is chosen for computational simplicity and wide adoption. It is known to underestimate the between-study variance (tau-squared) for small K [23]; the HKSJ correction (below) partially compensates by adjusting the confidence interval width, though the point estimate of tau-squared remains biased. Alternative estimators such as restricted maximum likelihood (REML) or Paule-Mandel could reduce this bias and are noted as a future extension.

**Equation 3:**
theta_hat(tau) = sum_{k=1}^{K} w_k * theta_k(tau) / sum_{k=1}^{K} w_k

where w_k = 1 / (SE_k^2 + tau_hat_squared) and tau_hat_squared is the between-study heterogeneity variance estimated via the DL moment estimator. Note: tau_hat_squared denotes between-study heterogeneity variance throughout this paper, distinct from the quantile index tau used in quantile regression. Both notations follow standard conventions in their respective fields.

**HKSJ correction.** For small K (the typical case in meta-analysis), we apply the Hartung-Knapp-Sidik-Jonkman (HKSJ) correction [13, 17], which replaces the normal reference distribution with a t-distribution on K - 1 degrees of freedom and inflates the pooled SE by sqrt(max(1, q_HKSJ)), where q_HKSJ is the HKSJ adjustment factor. Following Rover et al. [24], we truncate the adjustment factor at 1.0 to ensure the HKSJ-adjusted SE is never smaller than the DL SE.

**Heterogeneity diagnostics.** We report Cochran's Q, I-squared [14], tau-squared, and prediction intervals using the t(K - 2) distribution per Higgins et al. [14] and Riley et al. [10]. Prediction intervals should be interpreted cautiously when K is small (e.g., K = 3 gives df = 1, producing extremely wide intervals).

**Independent pooling per quantile.** Each quantile level is pooled independently with its own tau-squared estimate. No information is shared across quantiles for heterogeneity estimation. A multivariate random-effects model [25] that borrows strength across quantiles could improve efficiency, especially for small K, and represents a methodological extension.

This process is repeated for a vector of quantile levels tau in {0.10, 0.25, 0.50, 0.75, 0.90}. Plotting the pooled estimates theta_hat(tau) against tau generates the **Quantile Profile**.

### The log SD ratio (lnVR)

As a comparator, we compute the log SD ratio per Nakagawa et al. [8]:

lnVR = 0.5 * log(S_t^2 / S_c^2) + 1/(2*(n_t - 1)) - 1/(2*(n_c - 1))

where the last two terms are small-sample bias corrections. The sampling variance is:

Var(lnVR) = 1/(2*(n_t - 1)) + 1/(2*(n_c - 1))

Note: despite its name, lnVR measures the log ratio of standard deviations (lnVR = log(S_t / S_c)), not variances. A positive lnVR indicates greater variability in the treatment group. The related log coefficient of variation ratio (lnCVR) from the same Nakagawa framework adjusts for mean differences and could serve as an additional comparator in settings where means differ [8].

### The Slope Test

We define the inter-quantile slope (Delta) within each study as:

Delta_k = theta_hat_k(0.9) - theta_hat_k(0.1)

The bootstrap-derived SE of Delta_k preserves the covariance between theta_hat_k(0.9) and theta_hat_k(0.1), since both are estimated from the same bootstrap iteration. The study-level Delta_k values are then pooled across studies via DL random-effects, yielding a pooled Delta and its HKSJ-adjusted SE.

**Hypothesis tested.** The Slope Test tests H_0: the population-average QTE gradient is zero (i.e., the average across studies of the difference between the 90th and 10th percentile treatment effects is zero). This is a necessary but not sufficient condition for treatment effect homogeneity across the distribution. A zero pooled slope does not exclude opposing slopes in individual studies, and the test captures only linear gradients (e.g., U-shaped profiles where Q10 and Q90 are both positive but Q50 is zero would be missed). A multivariate Wald test of all quantile effects being equal would be a more comprehensive alternative and is noted as future work.

**Pre-specified primary analysis.** We designate the Slope Test as the primary confirmatory test. Individual quantile-specific tests (e.g., Q90 alone) are exploratory and should be interpreted descriptively, without formal multiplicity adjustment. A formal closed testing procedure would require specifying and testing all intersection hypotheses across quantiles, which is beyond the scope of this paper. We note that a multivariate Wald test of all quantile effects being equal, using the full bootstrap covariance matrix, would provide a more rigorous omnibus test.

### Simulation study design

To validate IPD-QMA, we designed a Monte Carlo simulation targeting the hidden utility scenario: a treatment that increases variance but has zero effect on the mean.

**Simulation parameters:**
- Iterations: 1,000 independent meta-analyses per scenario (MCSE approximately 0.7% at alpha = 0.05)
- Studies per meta-analysis: K = 10 (varied from 3 to 20)
- Sample size per study per arm: n ~ U(80, 200) (varied from 30-50 to 100-200)
- Bootstrap resamples: B = 200
- Quantile levels: tau in {0.1, 0.5, 0.9} (3 quantiles for computational efficiency; the full 5-quantile grid is used for the NHANES application)
- Confidence level: 95% with HKSJ correction
- Seeds: deterministic (10000 * scenario_index + iteration_index for data; offset by 1,000,000 for bootstrap)
- Software: Python 3.11, numpy 1.26, scipy 1.12, statsmodels 0.14, pandas 2.1

**The location-scale shift.** We generated data such that treatment and control groups had identical population means, but the treatment group had higher variance. We define scale_factor = sqrt(VR), where VR is the variance ratio (Var_treatment / Var_control), equivalently (SD ratio)^2.

*Normal distribution:*
- Control: X_c ~ N(0, 1); Treatment: X_t ~ N(0, scale_factor^2)
- True Q90 difference: z_0.9 * (scale_factor - 1) = 1.2816 * (sqrt(VR) - 1)
- For VR = 3: True Q90 diff = 1.2816 * (sqrt(3) - 1) = +0.938

*Exponential distribution:*
- Control: Exp(scale=1) centered by population mean (E[Exp(1)] = 1); Treatment: Exp(scale=scale_factor) centered by population mean (E[Exp(sf)] = sf)
- After centering, the tau-th quantile of Exp(s) becomes s*(-ln(1-tau) - 1)
- True Q90 difference: (scale_factor - 1) * (-ln(0.1) - 1) = (sqrt(VR) - 1) * 1.303
- For VR = 3: True Q90 diff = 0.7321 * 1.303 = +0.954
- Note: centering by the population mean preserves sampling variability in the mean difference while isolating the scale effect. The centered distribution includes negative values, which is a simulation convenience.

*Lognormal distribution:*
- Control: LogN(mu=0, sigma=0.5) centered by population mean; Treatment: LogN(0, 0.5) * scale_factor centered by population mean
- After centering, the tau-th quantile becomes sf * (exp(sigma * z_tau) - exp(sigma^2/2))
- True Q90 difference: (scale_factor - 1) * (Q_LN(0.9) - E[LN]) where Q_LN(0.9) = exp(0.5 * z_0.9) = 1.896 and E[LN] = exp(0.125) = 1.133
- For VR = 3: True Q90 diff = 0.7321 * (1.896 - 1.133) = +0.559

*t_5 distribution (heavy tails):*
- Control: t(5); Treatment: t(5) * scale_factor
- True Q90 difference: t_5^{-1}(0.9) * (scale_factor - 1) = 1.476 * (sqrt(VR) - 1)
- For VR = 3: True Q90 diff = 1.476 * 0.7321 = +1.080

True effect sizes differ across distributions, so cross-distribution power comparisons reflect both the method's sensitivity and the magnitude of the true effect under each DGP.

*Mixed location + scale scenario:*
We also included one scenario with simultaneous mean shift (MD = 0.5) and variance shift (VR = 2.0) under the Normal distribution, to validate that IPD-QMA correctly captures both effects.

**Comparison metrics.** We compared the rejection rate at alpha = 0.05 of four approaches:
1. Standard meta-analysis of mean differences (tests H_0: population mean difference = 0)
2. Log SD ratio (lnVR; tests H_0: population lnVR = 0)
3. IPD-QMA at tau = 0.9 (tests H_0: population Q90 treatment effect = 0)
4. The Slope Test (tests H_0: population-average QTE gradient = 0)

For the null scenarios (VR = 1.0), all four test the same truth (no effect); these results represent Type I error. For the location-scale scenarios (VR > 1.0), the mean difference is truly zero, so the MD rejection rate represents Type I error for the MD test (correct behavior), while Q90, Slope, and lnVR rejection rates represent power to detect the location-scale shift.

### Illustrative application: NHANES

To demonstrate the IPD-QMA workflow on real data, we analyzed systolic blood pressure (SBP) from the National Health and Nutrition Examination Survey (NHANES) across four survey cycles (2011-2012, 2013-2014, 2015-2016, 2017-2018). Each cycle served as a "study" in the meta-analysis (K = 4). These cycles are not independent studies in the meta-analytic sense — they share a sampling frame, measurement protocol, and potentially overlapping geographic primary sampling units (PSUs) — so the between-study heterogeneity estimated here reflects temporal variation rather than genuine methodological heterogeneity. This application is intended to demonstrate the IPD-QMA workflow, not to provide clinically interpretable results.

**Sample selection.** Respondents were included if they were aged >= 18, had a valid diagnosis of hypertension (BPQ020 = 1), had at least one valid SBP reading from readings 2-4, and had non-missing medication status. The treatment indicator was self-reported antihypertensive medication use (BPQ050A). Mean SBP was computed from readings 2-4 (excluding the first reading per NHANES protocol to reduce white-coat effect bias).

**Important limitations of this application:**
- **Confounding by indication.** Patients prescribed antihypertensives have higher baseline severity. The observed SBP differences reflect selection effects (who gets prescribed medication), not causal treatment effects. No causal interpretation is possible without adjustment (e.g., propensity scoring).
- **Survey design.** NHANES uses complex multistage probability sampling with stratification, clustering (PSUs), and person-level weights. We did not apply survey weights or account for the design effect. Consequently, the reported confidence intervals and p-values are approximate and may underestimate the true standard errors. Results are not population-representative.
- **Small K.** With K = 4 cycles, HKSJ uses t(3) critical values (3.182 for 95% CI), which is extremely conservative.
- **No covariate adjustment.** The binary treatment indicator does not account for medication intensity (monotherapy vs. polytherapy), medication class, adherence, or duration.
- **Informative missingness.** Exclusion of respondents with missing SBP readings or medication status may introduce selection bias if missingness is related to disease severity (e.g., sicker patients unable to attend the examination).

**Ethics statement.** This study analyzed publicly available, de-identified data from NHANES. The NHANES protocol was approved by the National Center for Health Statistics (NCHS) Research Ethics Review Board. All participants provided written informed consent. Secondary analysis of de-identified public-use data is exempt from further IRB review per 45 CFR 46.104(d)(4) (2018 Revised Common Rule).

---

## Results

### Type I error (null scenarios, VR = 1.0)

Under the true null (identical distributions, VR = 1.0, K = 10, n = 80-200), all methods maintained controlled Type I error rates (Table 1, Figure 1).

**Table 1. Type I error rates (%, nominal alpha = 5%) under the null hypothesis (VR = 1.0) across four distributions. N_sim = 1,000 per scenario; MCSE approximately 0.7% for rates near 5%.**

| Distribution | MD | Q90 | Slope Test | lnVR |
|---|---|---|---|---|
| Normal | 1.8 | 2.0 | 1.7 | 1.2 |
| Exponential | 0.0 | 0.2 | 1.7 | 4.3 |
| Lognormal | 0.0 | 0.5 | 1.4 | 4.2 |
| t_5 | 1.9 | 1.6 | 1.5 | 3.4 |

All rates are below the nominal 5% level. The MD rate of 0.0% for Exponential and Lognormal data is a structural consequence of the data-generating process (DGP): both arms are centered by their respective population means, so the mean difference has near-zero sampling variability under the null. The Normal and t_5 DGPs do not require centering, so their MD rates (1.8%, 1.9%) reflect genuine sampling variability and provide the relevant Type I error assessment. The lnVR rate for Exponential (4.3%) and Lognormal (4.2%) data is slightly elevated relative to Normal data (1.2%), consistent with mild anti-conservatism of the log-SD estimator for right-skewed distributions. Both remain within the 95% Monte Carlo CI for the nominal rate ([3.6%, 6.4%], Wald approximation). MCSE is rate-dependent: approximately 0.7% for rates near 5%, approximately 0.4% for rates near 2%.

### Power to detect location-scale shifts (VR > 1.0)

#### Power by variance ratio (Normal, K = 10, n = 80-200)

**Table 2. Rejection rates (%) by variance ratio. MD rejection rates represent Type I error for the mean difference test (H_0: MD = 0 is true by construction). Q90, Slope, and lnVR rejection rates represent power to detect the location-scale shift.**

| VR | MD (Type I) | Q90 (Power) | Slope (Power) | lnVR (Power) |
|---|---|---|---|---|
| 1.5 | 2.0 | 91.3 | 99.9 | 100.0 |
| 2.0 | 1.7 | 100.0 | 100.0 | 100.0 |
| 3.0 | 1.1 | 100.0 | 100.0 | 100.0 |
| 5.0 | 1.7 | 100.0 | 100.0 | 100.0 |

Standard meta-analysis of MD correctly retained the null (1.1-2.0%, all below nominal 5%). At VR = 1.5, Q90 achieved 91.3% power while Slope and lnVR reached 99.9-100%. At VR >= 2.0, all three location-scale methods achieved 100% power (Figure 2). **Note on comparability:** Q90, the Slope Test, and lnVR test different null hypotheses with different true effect sizes under the alternative (e.g., for VR = 2.0 Normal: true Q90 effect = 0.531, true Slope effect = 1.062, true lnVR = 0.347). Consequently, rejection rates are not directly comparable as measures of relative efficiency. They are presented to illustrate each method's absolute sensitivity to location-scale shifts.

#### Power by number of studies (Normal, VR = 2.0, n = 80-200)

**Table 3. Rejection rates (%) by number of studies (K).**

| K | MD (Type I) | Q90 (Power) | Slope (Power) | lnVR (Power) |
|---|---|---|---|---|
| 3 | 0.0 | 21.0 | 70.0 | 91.5 |
| 5 | 0.5 | 90.5 | 99.9 | 100.0 |
| 10 | 2.1 | 100.0 | 100.0 | 100.0 |
| 20 | 2.8 | 100.0 | 100.0 | 100.0 |

With K = 3, Q90 had only 21.0% power due to the extreme conservatism of the HKSJ t(2) critical value (4.303). The Slope Test achieved 70.0% and lnVR 91.5%. By K = 5, Q90 reached 90.5% and all methods exceeded 99.9% by K = 10 (Figure 3).

#### Power by sample size (Normal, VR = 2.0, K = 10)

**Table 4. Rejection rates (%) by maximum sample size per arm.**

| N range per arm | MD (Type I) | Q90 (Power) | Slope (Power) | lnVR (Power) |
|---|---|---|---|---|
| 30-50 | 1.9 | 84.1 | 99.7 | 100.0 |
| 50-100 | 1.8 | 98.7 | 100.0 | 100.0 |
| 100-200 | 1.7 | 100.0 | 100.0 | 100.0 |

Even with small samples (n = 30-50 per arm), Q90 achieved 84.1% power, while Slope and lnVR remained near 100%. Q90 reached 98.7% at n = 50-100 and 100% at n = 100-200 (Figure 4).

#### Mixed location + scale scenario (Normal, MD = 0.5, VR = 2.0, K = 10)

**Table 5. Rejection rates (%) under simultaneous mean shift and variance shift.**

| Method | Rejection Rate |
|---|---|
| MD | 100.0 |
| Q90 | 100.0 |
| Slope Test | 100.0 |
| lnVR | 100.0 |

All four methods achieved 100% rejection. Unlike the pure scale-shift scenarios where MD correctly retained the null, the MD = 0.5 mean shift is also detected by standard meta-analysis. This confirms that IPD-QMA captures both the mean and distributional components of treatment effects.

### The Quantile Profile

Under the VR = 3 Normal scenario, the analytical ground truth quantile treatment effects are:

**Table 6. Analytical ground truth QTEs for VR = 3 (Normal distribution). True QTE = z_tau * (sqrt(3) - 1).**

| Quantile (tau) | z_tau | True QTE |
|---|---|---|
| 0.10 | -1.282 | -0.938 |
| 0.25 | -0.674 | -0.494 |
| 0.50 | 0.000 | 0.000 |
| 0.75 | +0.674 | +0.494 |
| 0.90 | +1.282 | +0.938 |

The Quantile Profile reveals that the treatment is harmful for the lowest-risk patients (Q10 effect = -0.94), has no effect on the median patient, and is beneficial for the highest-risk patients (Q90 effect = +0.94). To translate into clinical terms: if the outcome is SBP with SD = 15 mmHg (typical of treated hypertensive populations), the Q90 effect of +0.938 corresponds to approximately 14 mmHg, which exceeds the clinically meaningful threshold of 5 mmHg SBP reduction associated with approximately 10% reduction in major cardiovascular events. The Slope Test confirmed that this gradient was statistically significant (p < 0.001).

### Illustrative NHANES application

The NHANES analysis (4 survey cycles, adults with diagnosed hypertension) demonstrated the IPD-QMA workflow on real observational data. The Quantile Profile of SBP differences between medicated and unmedicated hypertensives is shown in Table 7 and Figure 5. Forest plots at the 10th, 50th, and 90th percentiles showing study-level estimates and pooled effects are shown in Figure 6. A comparison dashboard (standard MA vs. IPD-QMA vs. lnVR) is provided in Figure 7.

**Table 7. NHANES Quantile Profile: SBP difference (mmHg) between medicated and unmedicated hypertensives (among diagnosed hypertensives only). K = 4 survey cycles. HKSJ correction with t(3) reference distribution. Confidence intervals and p-values are approximate due to unaccounted survey design effects (see Methods).**

| Quantile | Unadjusted Difference (mmHg) | 95% CI | P-value |
|---|---|---|---|
| 0.10 | -1.0 | (-4.5 to 2.4) | 0.41 |
| 0.25 | +0.8 | (-2.3 to 3.9) | 0.49 |
| 0.50 | -1.7 | (-4.6 to 1.3) | 0.18 |
| 0.75 | -3.2 | (-7.3 to 1.0) | 0.092 |
| 0.90 | -8.6 | (-15.4 to -1.8) | 0.027 |
| Slope (Q90 - Q10) | -7.6 | (-14.6 to -0.7) | 0.040 |
| lnVR | -0.13 | -- | 0.013 |

*Negative values indicate lower SBP in the medicated group. This analysis is restricted to adults with diagnosed hypertension; normotensives (those never asked about BP medications) are excluded. Results remain confounded by indication and should not be interpreted causally.*

The Quantile Profile shows a gradient from near zero at Q10 (-1.0 mmHg, p = 0.41) to a substantial negative effect at Q90 (-8.6 mmHg, p = 0.027). This pattern is consistent with antihypertensive medications having their largest effect on the most elevated blood pressures. The standard meta-analysis of mean differences found a pooled MD of -2.6 mmHg (p = 0.044), and lnVR was -0.13 (p = 0.013), indicating reduced variance in the medicated group. The Slope Test was significant (p = 0.040), confirming distributional heterogeneity. However, this observational comparison remains confounded by indication: patients with the most severe hypertension receive the most aggressive treatment. The negative gradient could reflect genuine treatment efficacy, differential prescribing intensity, or both. No causal interpretation is possible without covariate adjustment.

---

## Discussion

### Screening versus diagnosis: the lnVR comparison

The central finding is that standard meta-analysis is blind to location-scale shifts. In our simulations, a treatment with zero mean effect but substantial quantile-specific effects was correctly identified by IPD-QMA and lnVR but (appropriately) not flagged by standard meta-analysis of mean differences.

Our results show that lnVR has excellent power to detect variance heterogeneity: 100% at VR >= 1.5 with K = 10 (Table 2), and 100% at VR = 2.0 with K >= 5 (Table 3); even at K = 3, lnVR achieved 91.5%. We argue that lnVR is a screening tool, whereas IPD-QMA is a diagnostic tool. A significant lnVR tells a researcher that the treatment group outcome is more variable than the control. This could mean the treatment helps the severe and harms the mild (as in our simulation), the treatment adds random noise, or the treatment pushes outcomes toward the extremes. IPD-QMA provides the directionality needed for clinical decision-making.

We suggest a hierarchical workflow: researchers calculate lnVR as a quick check for variance heterogeneity. If lnVR is significant, IPD-QMA should be performed to characterize the shape of that heterogeneity. We note that conditioning IPD-QMA analysis on a significant lnVR screen introduces a selection effect; the nominal coverage of IPD-QMA confidence intervals may not hold under this conditional inference framework. Researchers should interpret IPD-QMA results as exploratory characterization rather than confirmatory inference when the analysis is triggered by a screening step.

### Clinical interpretation of quantile effects

The Quantile Profile provides a visual summary of how the treatment effect varies across the outcome distribution. However, the clinical interpretation requires care. The QTEs estimated by IPD-QMA are unconditional: they compare the tau-th quantile of the treatment group outcome distribution with the tau-th quantile of the control group distribution. A patient "at the 90th percentile" is at the 90th percentile of the *outcome*, not necessarily the 90th percentile of disease severity, unless outcome rank and severity rank are strongly correlated. This interpretation holds for outcomes positively correlated with severity (e.g., higher SBP = more severe hypertension) but reverses for outcomes where lower values are worse.

The Slope Test detects distributional heterogeneity (differences in QTEs across quantiles), which is a necessary but not sufficient condition for clinically actionable heterogeneity of treatment effects. Translating a significant Slope Test into patient-level treatment decisions requires additional steps: baseline risk prediction, identification of effect modifiers, and ideally a randomized design. These are complementary to IPD-QMA, not replacements for it. The relationship to modern approaches for heterogeneous treatment effects, including risk-based methods [3], causal forests, and Bayesian nonparametric models, merits further investigation.

### Robustness across distributions

Our simulations across four distributions (Normal, Exponential, Lognormal, t_5) showed that IPD-QMA maintains controlled Type I error and high power even with non-Normal data. Quantile regression is semi-parametric and does not assume normality of error terms [9], making it particularly suitable for healthcare cost data, length of stay, or biomarker concentrations. The HKSJ correction ensures conservative inference with small K.

### When IPD-QMA has low power

With K = 3 studies and VR = 2.0, the Q90 test achieved only 21.0% power. This reflects the fundamental limitation of small meta-analyses: with 3 study-level estimates, quantile-specific effects are imprecisely estimated, and the HKSJ t(2) critical value (4.303) is extremely conservative. In such settings, the Slope Test and lnVR are better initial screens. Alternative small-sample corrections such as Kenward-Roger or Satterthwaite degrees-of-freedom adjustments [26] might improve performance but were not evaluated here. We recommend IPD-QMA primarily for meta-analyses with K >= 5 studies, noting that even at K = 5, power depends on the magnitude of the distributional shift and within-study sample sizes (Table 3: Q90 achieved 90.5% at K = 5, VR = 2.0, n = 80-200).

### Limitations

1. **IPD requirement.** Unlike standard meta-analysis, IPD-QMA requires raw patient-level data. Initiatives such as the Yale Open Data Access (YODA) project and journal mandates for data-sharing are making IPD increasingly accessible.

2. **Computational cost.** Bootstrap standard errors with B = 200 iterations across quantile levels and K studies are more expensive than simple weighted averaging. However, the entire analysis completes in under 1 second per study on a standard laptop.

3. **Observational confounding.** The NHANES application is subject to confounding by indication. The observed SBP gradient (medication associated with lower SBP primarily at upper quantiles) could reflect genuine treatment efficacy, differential prescribing intensity by severity, or both. When IPD-QMA is applied to RCTs (where randomization ensures exchangeability), QTEs can be interpreted causally. For observational data, additional adjustment (propensity scoring, instrumental variables) is required.

4. **Survey design effects.** The NHANES analysis does not account for the complex survey design (stratification, clustering, sampling weights). Confidence intervals may underestimate the true standard errors by a factor of 1.5-3x (typical design effects for NHANES blood pressure outcomes). For survey data, a cluster-bootstrap resampling of PSUs would be required.

5. **Monte Carlo precision.** Our simulation used 1,000 repetitions per scenario (MCSE approximately 0.7% for rates near 5%). The 95% CI for an observed 5% rate is [3.6%, 6.4%], adequate for distinguishing nominal calibration from meaningful inflation.

6. **Multiple testing.** We test treatment effects at multiple quantiles plus the Slope Test and lnVR. The Slope Test is designated as the primary confirmatory analysis; individual quantile tests are exploratory and do not require formal multiplicity adjustment when interpreted as components of the overall Quantile Profile pattern. A joint test of all quantile effects (e.g., a multivariate Wald test using the full bootstrap covariance matrix) would be a more comprehensive alternative.

7. **Independent pooling.** Each quantile is pooled independently with its own tau-squared estimate. A multivariate random-effects model [25] that borrows strength across quantiles could improve precision, particularly for small K.

8. **DerSimonian-Laird estimator.** DL is known to underestimate tau-squared for small K [23]. We use HKSJ to compensate for CI calibration, but the point estimate of heterogeneity remains biased. REML or Paule-Mandel estimators are alternatives.

9. **Simulation ecological validity.** Our primary simulation uses pure scale shifts with zero mean difference. Real clinical interventions typically produce simultaneous location and scale effects. We included one mixed scenario (MD = 0.5, VR = 2.0) but a comprehensive exploration is needed. A pure location shift scenario (VR = 1.0, MD > 0) was not included; in this case, IPD-QMA should detect a uniform shift across all quantiles, and the Slope Test should correctly retain the null (zero gradient). Additionally, the variance ratios tested (1.5-5.0) include extreme values; Nakagawa et al. [8] found typical lnVR values in the range -0.5 to +0.5 (VR approximately 0.6 to 1.6).

10. **Homogeneous heterogeneity.** All simulations assumed constant between-study heterogeneity (tau-squared) across quantile levels. In practice, treatment effect heterogeneity may vary across the distribution (e.g., greater between-study variability at extreme quantiles). Quantile-specific tau-squared estimation may behave differently across quantile levels, and this was not evaluated.

11. **Slope Test blind spot.** The Slope Test captures only the linear gradient between Q90 and Q10 effects. U-shaped or inverted-U profiles — where both tails show effects of the same sign but the median shows no effect — would produce a near-zero slope despite genuine distributional heterogeneity. This scenario was not included in the simulation study. A multivariate Wald test or a quadratic Slope Test extension could address this limitation.

12. **Bootstrap iterations.** The simulation used B = 200 bootstrap resamples for computational feasibility (1,000 scenarios x K studies x B resamples). While B = 200 is adequate for variance estimation [11], it may be insufficient for accurate estimation of bootstrap confidence interval coverage. The real-data analysis used B = 500. For definitive applications, B >= 1,000 is recommended.

13. **Equal allocation.** All simulated studies used equal sample sizes in treatment and control arms (n_t = n_c). Unbalanced designs (common in pragmatic trials) may affect quantile regression precision differently at extreme quantiles, particularly for the smaller arm.

14. **Comparison to existing methods.** A systematic comparison against one-stage quantile meta-analysis approaches [15] and distributional meta-analysis methods would strengthen the evidence base for IPD-QMA.

### Future directions

Future work should explore:
- One-stage IPD-QMA models using hierarchical Bayesian frameworks, which may handle smaller sample sizes by borrowing strength across studies and quantiles.
- Extension to time-to-event data (quantile survival analysis) for oncology trials.
- Integration with REML or Paule-Mandel tau-squared estimators.
- Formal sample size and power calculations for planning IPD-QMA studies.
- Covariate-adjusted quantile regression within the IPD-QMA framework, which would require careful distinction between conditional and unconditional QTEs [22].
- A second real-data application using a publicly available RCT dataset (e.g., from YODA) to demonstrate IPD-QMA in its intended causal inference setting.
- Coverage probability assessment via simulation to verify that the HKSJ-corrected confidence intervals achieve nominal 95% coverage across different K, n, and distribution settings.
- Consistency between forest plot visualization (which uses study-level z-statistics for display) and the pooled inference (which uses the t-distribution via HKSJ). A fully consistent implementation would use t-based intervals at both the study and pooled levels.

---

## Conclusions

The average treatment effect is a useful but incomplete summary. When treatments create location-scale shifts, the mean can mask clinically important variation in who benefits and who is harmed. IPD-QMA complements existing meta-analytic tools by characterizing the shape of treatment effects across the outcome distribution, providing the directionality that lnVR alone cannot offer. We recommend a hierarchical workflow: screen with lnVR, then characterize with IPD-QMA. An open-source Python implementation is provided to facilitate adoption.

---

## References

[1] Sutton AJ, Abrams KR, Jones DR, Sheldon TA, Song F. *Methods for Meta-Analysis in Medical Research*. Wiley; 2000. doi: 10.1002/0470870168

[2] Borenstein M, Hedges LV, Higgins JPT, Rothstein HR. *Introduction to Meta-Analysis*. Wiley; 2009. doi: 10.1002/9780470743386

[3] Kent DM, Rothwell PM, Ioannidis JP, Altman DG, Hayward RA. Assessing and reporting heterogeneity in treatment effects in clinical trials: a proposal. *Trials*. 2010;11:85. doi: 10.1186/1745-6215-11-85

[4] Cortese S, Maayan L, Castellanos FX, et al. Mean differences, standardized mean differences, and their clinical relevance. *J Am Acad Child Adolesc Psychiatry*. 2012;51(9):876-878. doi: 10.1016/j.jaac.2012.06.013

[5] Iwashyna TJ, Burke JF, Sussman JB, Prescott HC, Hayward RA, Angus DC. Implications of Heterogeneity of Treatment Effect for Reporting and Analysis of Randomized Trials in Critical Care. *Am J Respir Crit Care Med*. 2015;192(9):1045-1051. doi: 10.1164/rccm.201411-2125CP

[6] Brookes ST, Whitley E, Peters TJ, Mulheran PA, Egger M, Davey Smith G. Subgroup analyses in randomized controlled trials. *Health Technology Assessment*. 2001;5(33). doi: 10.3310/hta5330

[7] Berlin JA, Santanna J, Schmid CH, Szczech LA, Feldman HI. Individual patient- versus group-level data meta-regressions for the investigation of treatment effect modifiers. *Statistics in Medicine*. 2002;21(3):371-387. doi: 10.1002/sim.1023

[8] Nakagawa S, Poulin R, Mengersen K, et al. Meta-analysis of variation: ecological and evolutionary applications and beyond. *Methods in Ecology and Evolution*. 2015;6(2):143-152. doi: 10.1111/2041-210X.12309

[9] Koenker R, Bassett G. Regression Quantiles. *Econometrica*. 1978;46(1):33-50. doi: 10.2307/1913643

[10] Riley RD, Lambert PC, Abo-Zaid G. Meta-analysis of individual participant data: rationale, conduct, and reporting. *BMJ*. 2010;340:c221. doi: 10.1136/bmj.c221

[11] Efron B. Bootstrap Methods: Another Look at the Jackknife. *The Annals of Statistics*. 1979;7(1):1-26. doi: 10.1214/aos/1176344552

[12] DerSimonian R, Laird N. Meta-analysis in clinical trials. *Controlled Clinical Trials*. 1986;7(3):177-188. doi: 10.1016/0197-2456(86)90046-2

[13] Hartung J, Knapp G. A refined method for the meta-analysis of controlled clinical trials with binary outcome. *Statistics in Medicine*. 2001;20(24):3875-3889. doi: 10.1002/sim.1009

[14] Higgins JPT, Thompson SG. Quantifying heterogeneity in a meta-analysis. *Statistics in Medicine*. 2002;21(11):1539-1558. doi: 10.1002/sim.1186

[15] Sun Y, Cheung YB. A quantile regression method for meta-analysis of studies with heterogeneity. *Statistics in Medicine*. 2020;39(26):3853-3871. doi: 10.1002/sim.8698

[16] Koenker R. *Quantile Regression*. Cambridge University Press; 2005. doi: 10.1017/CBO9780511754098

[17] Sidik K, Jonkman JN. A simple confidence interval for meta-analysis. *Statistics in Medicine*. 2002;21(21):3153-3159. doi: 10.1002/sim.1262

[18] Stewart LA, Clarke M, Rovers M, et al. Preferred Reporting Items for a Systematic Review and Meta-analysis of Individual Participant Data: The PRISMA-IPD Statement. *JAMA*. 2015;313(16):1657-1665. doi: 10.1001/jama.2015.3656

[19] von Elm E, Altman DG, Egger M, Pocock SJ, Gotzsche PC, Vandenbroucke JP. The Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement. *The Lancet*. 2007;370(9596):1453-1457. doi: 10.1016/S0140-6736(07)61602-X

[20] Riley RD, Higgins JPT, Deeks JJ. Interpretation of random effects meta-analyses. *BMJ*. 2011;342:d549. doi: 10.1136/bmj.d549

[21] Russell JA, Walley KR, Singer J, et al. Vasopressin versus norepinephrine infusion in patients with septic shock. *New England Journal of Medicine*. 2008;358(9):877-887. doi: 10.1056/NEJMoa067373

[22] Firpo S. Efficient semiparametric estimation of quantile treatment effects. *Econometrica*. 2007;75(1):259-276. doi: 10.1111/j.1468-0262.2007.00738.x

[23] Veroniki AA, Jackson D, Viechtbauer W, et al. Methods to estimate the between-study variance and its uncertainty in meta-analysis. *Research Synthesis Methods*. 2016;7(1):55-79. doi: 10.1002/jrsm.1164

[24] Rover C, Knapp G, Friede T. Hartung-Knapp-Sidik-Jonkman approach and its modification for random-effects meta-analysis with few studies. *BMC Medical Research Methodology*. 2015;15:99. doi: 10.1186/s12874-015-0091-1

[25] Jackson D, Riley R, White IR. Multivariate meta-analysis: potential and promise. *Statistics in Medicine*. 2011;30(20):2481-2498. doi: 10.1002/sim.4172

[26] Partlett C, Riley RD. Random effects meta-analysis: Coverage performance of 95% confidence and prediction intervals following REML estimation. *Statistics in Medicine*. 2017;36(2):301-317. doi: 10.1002/sim.7140

---

## Figure Legends

**Figure 1.** Type I error rates under the true null hypothesis (VR = 1.0, K = 10, n = 80-200) across four distributions. Bars show rejection rates at alpha = 0.05 for each method. The dashed red line indicates the nominal 5% level; the shaded band indicates the acceptable range. All methods maintained controlled Type I error.

**Figure 2.** Statistical power by variance ratio (Normal distribution, K = 10, n = 80-200). Lines show rejection rates for each method as the treatment-to-control variance ratio increases from 1.5 to 5.0. Standard meta-analysis (MD) correctly retains the null at all VR levels, while IPD-QMA (Q90), the Slope Test, and lnVR achieve near-100% power at VR >= 2.0.

**Figure 3.** Statistical power by number of studies (Normal, VR = 2.0, n = 80-200). The Slope Test and lnVR maintain high power even at K = 3, while Q90 requires K >= 5 for adequate power due to the conservatism of the HKSJ t-distribution correction.

**Figure 4.** Statistical power by sample size per study (Normal, VR = 2.0, K = 10). Even with small samples (n = 30-50 per arm), the Slope Test and lnVR achieve near-100% power. Q90 reaches 84.1% at the smallest sample size and 100% by n = 100-200.

**Figure 5.** NHANES illustrative application: Quantile Treatment Effect Profile of systolic blood pressure (SBP) differences between medicated and unmedicated hypertensives (diagnosed hypertensives only) across 4 survey cycles. The gradient from near zero at Q10 to -8.6 mmHg at Q90 is consistent with antihypertensives having their largest effect on the most elevated blood pressures, though confounding by indication cannot be excluded. The shaded band shows the 95% confidence interval (approximate; survey design effects not accounted for).

**Figure 6.** NHANES illustrative application: Forest plots at the 10th, 50th, and 90th percentiles showing study-level (survey cycle) estimates with 95% CIs and pooled random-effects estimates (diamonds). HKSJ correction with t(3) reference distribution.

**Figure 7.** NHANES illustrative application: Comparison dashboard showing standard meta-analysis (mean difference), IPD-QMA Quantile Profile, and log SD ratio (lnVR) side by side.

---

## Supporting Information

**S1 File. IPD-QMA Software.**

The corrected Python implementation (`ipd_qma.py`) includes:
- Quantile regression via `statsmodels.QuantReg`
- DerSimonian-Laird random-effects pooling with tau-squared estimation
- HKSJ correction [13, 17] with t(K - 1) distribution and max(1, q) truncation [24]
- Configurable confidence level
- Prediction intervals per Riley et al. [20] (df = K - 2)
- Corrected lnVR as log SD ratio with Nakagawa bias correction [8]
- Bootstrap SEs preserving cross-quantile covariance
- Input validation and edge-case guards (K = 1, zero variance)
- Seeded PRNG for full reproducibility

The software is available as S1 File and at [repository URL -- to be deposited on Zenodo with DOI before submission].

**S2 Note. PRISMA-IPD.**

Because this is a methods paper rather than a systematic review, a formal PRISMA-IPD checklist [18] is not fully applicable. A partial mapping of relevant PRISMA-IPD items is provided as S2 File.

**S3 Checklist. STROBE Checklist (partial).**

A partial STROBE checklist [19] for the NHANES illustrative application is provided as S3 File, with notation that full STROBE compliance is not applicable as this is a methods demonstration, not a primary observational study.

---

## Data Availability Statement

NHANES data are publicly available from the Centers for Disease Control and Prevention (CDC) National Center for Health Statistics at https://wwwn.cdc.gov/nchs/nhanes/. No access restrictions apply. All analysis code is available in S1 File and at [Zenodo DOI -- to be minted before submission].

## Funding

[To be completed by authors.]

## Competing Interests

The authors have declared that no competing interests exist.

## Author Contributions

[To be completed using CRediT taxonomy: Conceptualization, Methodology, Software, Validation, Formal analysis, Writing -- original draft, Writing -- review and editing, Visualization.]

## Acknowledgments

[To be completed by authors.]
