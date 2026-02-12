Actuarial Standards Compliance
==============================

This document provides disclosures required by `Actuarial Standard of Practice (ASOP) No. 41: Actuarial Communications <https://www.actuarialstandardsboard.org/asops/actuarial-communications/>`_ (adopted December 2010, effective May 1, 2011) and related standards. It is intended as a consolidated reference for actuaries, risk managers, and other users of this framework.

.. contents:: Table of Contents
   :depth: 2
   :local:

Applicable Professional Standards
----------------------------------

This framework and its documentation are subject to the following Actuarial Standards of Practice (ASOPs) issued by the Actuarial Standards Board:

- **ASOP No. 41** -- *Actuarial Communications* (December 2010). Governs the form, content, and disclosure requirements for actuarial communications.
- **ASOP No. 56** -- *Modeling* (December 2019). Governs the design, development, selection, modification, and use of models by actuaries.
- **ASOP No. 23** -- *Data Quality* (December 2016). Governs the selection, review, and use of data in actuarial work.
- **ASOP No. 25** -- *Credibility Procedures* (December 2013). Governs the use of credibility procedures in actuarial analyses. Referenced in the insurance pricing module.
- **ASOP No. 43** -- *Property/Casualty Unpaid Claim Estimates* (June 2011). Governs unpaid claim estimates; relevant to the framework's claim development and reserving models.
- **ASOP No. 13** -- *Trending Procedures in Property/Casualty Insurance* (December 2013). Governs loss cost trending; referenced in the loss distribution and pricing modules.

Users relying on this framework for actuarial purposes should be familiar with these standards and should ensure their own compliance when using framework outputs in professional communications.

.. note::

   A second exposure draft for a proposed revision of ASOP 41 was approved by the ASB in October 2024. The current enforceable version (December 2010) is the compliance target for this document. Should the proposed revision be adopted, this document will be updated accordingly.


Responsible Actuary
--------------------

*Per ASOP 41 SS3.1.4 and SS4.1.1*

The responsible actuary for this framework is:

| **Alex Filiakov, ACAS**
| alexfiliakov@gmail.com
| `https://github.com/AlexFiliakov <https://github.com/AlexFiliakov>`_

Mr. Filiakov has over 10 years of experience in the actuarial field, including 2.5 years specifying and developing similar simulation models at a life insurance carrier. He is an Associate of the Casualty Actuarial Society (ACAS).

.. important::

   The reviewing actuary, Alex Filiakov, does not currently take responsibility for the accuracy of the methodology or results produced by this framework. Review and validation are ongoing. See :ref:`qualification-statement` and :ref:`applicability-of-outputs` for further detail.


Intended Users and Scope
-------------------------

*Per ASOP 41 SS3.7 and SS4.1.3(a)-(b)*

Intended Users
~~~~~~~~~~~~~~~

The intended users of this framework and its outputs are:

- **Qualified actuaries** who are comfortable validating Python-based simulation models and who can independently assess the reasonableness of model outputs with sufficient domain expertise.
- **Actuarial researchers** investigating ergodic economics and its application to insurance decision-making.
- **Risk management professionals** with quantitative backgrounds who use the framework under the supervision or review of a qualified actuary.

Users who lack actuarial training or quantitative modeling experience are **not** intended users of the framework's raw outputs. Non-technical stakeholders (e.g., CFOs, board members) may receive and rely upon summaries and reports prepared by a qualified actuary who has independently validated the outputs.

.. warning::

   This is primarily an **early-stage research tool**. Persons relying on this tool for financial decisions must be qualified actuaries who are comfortable validating Python models and who can verify the outputs of this model with sufficient expertise.

Reliance Limitations
~~~~~~~~~~~~~~~~~~~~~

Per ASOP 41 SS3.7, the following reliance limitations apply:

- Outputs of this framework **should not be relied upon** by any party other than the intended users described above without independent actuarial review.
- No party should rely on framework outputs for regulatory filings, rate opinions, reserve opinions, or statutory reporting without conducting a separate, independent actuarial analysis.
- The responsible actuary does not assume responsibility for reliance by parties beyond the intended users.

Scope and Purpose
~~~~~~~~~~~~~~~~~~

This framework is designed for the following purpose:

- **Research and education** in ergodic economics applied to insurance decision-making.
- **Exploratory analysis** of how time-average growth optimization produces different insurance purchasing recommendations than traditional ensemble-average (expected value) approaches.
- **Illustrative modeling** of insurance program structures for widget manufacturing companies.

This framework is **not** designed for and **should not** be used for:

- Making financial business decisions without independent actuarial review and validation.
- Regulatory capital calculations or statutory reporting.
- Insurance rate filings or rate adequacy opinions.
- Reserve opinions or unpaid claim estimates for financial reporting.
- Statements of Actuarial Opinion for any regulatory or contractual purpose.
- Replacing professional actuarial judgment in any context.

Additional code review, validation, and enhancement must be conducted before this tool can be used for insurance planning, business strategy, or insurance pricing in a production context.

.. note::

   This tool does not constitute an actuarial opinion or rate filing. Its outputs are intended for research purposes only and should be treated as illustrative, not prescriptive.


.. _qualification-statement:

Qualification Statement
------------------------

*Per ASOP 41 SS4.1.3(c)*

The output of this framework **does not constitute a Statement of Actuarial Opinion** as defined by the American Academy of Actuaries' Qualification Standards for Actuaries Issuing Statements of Actuarial Opinion.

The framework has not been reviewed for compliance with the Qualification Standards by a qualified actuary holding the appropriate credentials for issuing Statements of Actuarial Opinion. The responsible actuary (Alex Filiakov, ACAS) is conducting ongoing review but has not completed the level of validation required to issue such a statement.

Any actuary using this framework's outputs in a professional communication that constitutes a Statement of Actuarial Opinion must independently satisfy the Qualification Standards applicable to their specific practice area and assignment.


Material Assumptions
---------------------

*Per ASOP 41 SS3.2 and SS3.4.4*

This section provides a consolidated inventory of the material assumptions embedded in the framework. The peer review test from ASOP 41 SS3.2 requires that methods, procedures, assumptions, and data be identified with sufficient clarity that another actuary qualified in the same practice area could make an objective appraisal of the reasonableness of the work.

Assumptions are categorized below by type and responsibility.

Framework-Embedded Assumptions (Developer Responsibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following assumptions are embedded in the framework's code and methodology. They were selected by the developer primarily through research conducted using Large Language Models and academic literature, and are intended to be illustrative rather than calibrated to any specific real-world entity.

**Mathematical and Stochastic Process Assumptions:**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Assumption
     - Description
     - Sensitivity
   * - Multiplicative wealth dynamics
     - Wealth evolves as :math:`W_{t+1} = W_t \cdot R_t` where returns are multiplicative, not additive. This is the foundational assumption of the ergodic framework.
     - **High**. The entire framework's conclusions depend on this assumption. If wealth dynamics are substantially additive, ensemble and time averages converge and the ergodic advantage disappears.
   * - Geometric Brownian Motion (GBM) for revenue
     - Revenue follows :math:`dR = \kappa(\theta - R)dt + \sigma R \, dW_t` with optional mean reversion. Growth is log-normally distributed.
     - **Moderate**. GBM is a standard assumption in financial modeling but may understate tail risk in revenue processes. Mean reversion partially addresses this.
   * - Compound Poisson loss frequency
     - Claim count follows :math:`N \sim \text{Poisson}(\lambda)` with parameter :math:`\lambda` representing expected annual claim frequency. Claims are independent and identically distributed.
     - **Moderate to High**. Poisson assumes no overdispersion and no loss clustering. Real-world losses often exhibit correlation (e.g., supply chain disruptions producing multiple related claims). See :ref:`deviation-poisson`.
   * - Log-normal loss severity
     - Individual claim amounts follow :math:`X_i \sim \text{LogNormal}(\mu, \sigma^2)`, producing right-skewed, positive-valued loss amounts.
     - **Moderate**. Log-normal is a reasonable severity model for many lines of business but may understate heavy-tailed risk compared to Pareto or GPD models for catastrophic losses.
   * - Independence of losses across time
     - Loss events in different periods are statistically independent. No serial correlation or contagion effects.
     - **Moderate**. Violation would increase retained loss volatility and potentially shift optimal deductibles lower.
   * - Independence of losses and revenue
     - Loss frequency and severity are independent of business performance and revenue levels (except through exposure scaling).
     - **Moderate**. In practice, some losses correlate with business activity levels. Violation could amplify downside risk during economic downturns.
   * - Volatility drag formula
     - Time-average growth is reduced by :math:`\sigma^2/2` relative to ensemble average, following from Jensen's inequality applied to the logarithm.
     - **Low** (mathematically exact for continuous GBM). Higher for discrete approximations.

**Financial and Business Model Assumptions:**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Assumption
     - Description
     - Sensitivity
   * - Fixed operating margins
     - Operating margins remain constant over the simulation horizon. No operating leverage effects.
     - **Moderate**. Real margins fluctuate with business cycles, competitive dynamics, and input costs. Fixed margins understate earnings volatility.
   * - Fixed asset turnover ratios
     - Revenue = Assets x Asset Turnover, with a constant turnover ratio.
     - **Low to Moderate**. Simplification that removes a source of variability.
   * - Flat corporate tax rate (25%)
     - A single flat tax rate applies to all operating income. No progressive brackets, AMT, or international tax considerations.
     - **Low**. Directionally correct for U.S. corporate taxation. Understates tax complexity.
   * - Simplified NOL carryforward
     - Net operating losses carry forward per ASC 740 but with simplified mechanics.
     - **Low**. Conservative simplification.
   * - No investment income
     - The firm earns no return on invested assets. Cash and liquid assets are non-earning.
     - **Low to Moderate**. Understates the opportunity cost of holding capital as self-insurance, potentially making self-insurance appear more attractive than it is.
   * - No external capital raises
     - The firm cannot raise equity or debt. Growth is funded solely through retained earnings.
     - **Moderate**. Removes a real-world safety valve for capital-impaired firms. May overstate ruin probability.
   * - 70% retention ratio
     - 70% of net income is retained; 30% paid as dividends. Fixed throughout simulation.
     - **Low**. Adjustable parameter; illustrative default.
   * - No regulatory capital requirements
     - No minimum capital ratios, rating agency thresholds, or debt covenants constrain operations.
     - **High** for regulated entities. Regulatory capital can be a binding constraint that overrides growth optimization. See :ref:`deviation-regulatory-capital`.
   * - Annual time resolution
     - Financial dynamics are modeled on an annual basis. No intra-year cash flow timing.
     - **Low to Moderate**. Acceptable for strategic-level analysis; insufficient for cash management or liquidity analysis.
   * - Working capital set to 0%
     - No working capital requirements reduce available cash. All cash is available for operations and growth.
     - **Low**. Simplification that slightly overstates available capital.
   * - PP&E ratio set to 0%
     - No property, plant, and equipment. No depreciation expense. Capital-light business model.
     - **Low to Moderate**. Manufacturing companies typically have significant PP&E. Adjustable parameter.

**Insurance-Specific Assumptions:**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Assumption
     - Description
     - Sensitivity
   * - Fixed premium rates
     - Insurance premiums are fixed at inception using the underlying loss distribution and scale with revenue. No experience rating adjustments.
     - **Moderate to High**. Real premiums adjust annually based on loss experience, market conditions, and insurer capital. Dynamic pricing could attenuate or amplify the ergodic advantage.
   * - Fixed target loss ratio pricing
     - Premium = Expected Losses / Target Loss Ratio. No risk loading, expense loading, or market cycle dynamics in the base model.
     - **Moderate**. Understates real-world premium levels, potentially understating the cost of insurance.
   * - No insurer credit risk
     - Insurance recoveries above the deductible are certain. No consideration of insurer insolvency or coverage disputes.
     - **Low** under normal conditions. **High** during insurance market stress or with poorly rated carriers.
   * - Multi-year claim development schedule
     - Claims pay out over a fixed development schedule (10% Year 1, cumulative to 100% Year 10). Based on standard casualty development patterns.
     - **Low to Moderate**. Schedule is adjustable. Faster payment reduces collateral costs; slower payment increases them.
   * - Letter of Credit collateral at 1.5% annual cost
     - Deductible portions require immediate collateral via LoC at 1.5% annual interest on outstanding collateral.
     - **Low to Moderate**. Real LoC costs vary by creditworthiness. Adjustable parameter.
   * - Layer independence
     - Insurance layers respond sequentially and independently to losses. No cross-layer interactions.
     - **Low**. Standard assumption in insurance program design.

User-Selected Assumptions (User Responsibility)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following assumptions are configured by the user at runtime. **Users bear full responsibility for the selection and reasonableness of these parameters.** The responsible actuary does not assume responsibility for user-selected assumption values.

- **Initial assets and capitalization** -- Starting balance sheet values.
- **Loss frequency and severity parameters** -- :math:`\lambda`, :math:`\mu`, :math:`\sigma` for the compound Poisson/log-normal model.
- **Operating margins and asset turnover ratios** -- Business-specific financial parameters.
- **Insurance program structure** -- Deductibles, attachment points, limits, and premium rates.
- **Time horizon and simulation count** -- Duration and sample size for Monte Carlo analysis.
- **Industry selection** -- Determines default parameter profiles (manufacturing, service, retail).

Users should apply data quality standards consistent with ASOP 23 when selecting input parameters. Where parameters are estimated from historical data, users should consider credibility (ASOP 25) and trending (ASOP 13) procedures.


Limitations and Constraints
----------------------------

*Per ASOP 41 SS4.1.3(e)*

This section documents limitations on the use and applicability of framework outputs. Users should not rely upon framework outputs for purposes beyond the stated scope.

General Limitations
~~~~~~~~~~~~~~~~~~~~

- This framework is provided **as-is** with absolutely no guarantee of validity or correctness of methodology or results.
- All users and contributors are expected to perform their own due diligence when relying on this tool for any purpose.
- Framework outputs should be treated as **directional guidance** rather than prescriptive recommendations. Optimal strategies for any specific company require custom calibration incorporating company-specific data, constraints, and circumstances.
- Results are illustrative and depend heavily on modeled loss distributions, business parameters, and simplifying assumptions that do not reflect any individual company's circumstances.
- Use these tools to **inform, not replace**, professional actuarial judgment.

Conditions Under Which Results May Be Unreliable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Framework outputs should be used with particular caution (or may be unreliable) when:

- **Extreme parameter values** are used -- very high or very low loss frequencies, severity levels, or operating margins that fall outside the range validated by the test suite.
- **Heavy-tailed distributions** are required -- the default log-normal severity model may understate tail risk for lines of business with Pareto-like or generalized Pareto severity (e.g., excess liability, catastrophe exposure).
- **Very small companies** (below $1M in assets) are modeled -- where discrete loss events represent a disproportionate share of total assets and the continuous approximation breaks down.
- **Very large companies** ($50M+ capitalization) are modeled -- where the ergodic advantage diminishes as the loss-to-asset ratio decreases and operational volatility has proportionally smaller impact.
- **Correlated loss environments** -- where losses cluster (supply chain disruptions, product liability waves, regulatory actions). The Poisson independence assumption may materially understate risk.
- **Regulated industries** -- where regulatory capital requirements, rating agency capital adequacy standards, or debt covenant restrictions impose binding constraints on retention decisions. Industries such as banking, insurance, and utilities have capital requirements that may override growth optimization.
- **Complex industries** -- pharmaceuticals and life sciences, oil and gas, financial services, and other industries with specialized risk profiles, complex asset portfolios, or unique regulatory regimes that are not captured by the widget manufacturing model.
- **Investment-intensive businesses** -- where investment income on assets materially affects the economics of self-insurance versus risk transfer.
- **Short time horizons** (1-3 years) -- where ergodic convergence may not be achieved and the time-average framework offers limited advantage over expected value analysis.
- **Dynamic business environments** -- where the stationarity assumptions (fixed margins, constant loss parameters) are significantly violated.

Specific Technical Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following specific limitations are documented in the research paper (*Ergodicity and Basic Insurance Simulation*, Filiakov 2026) and apply to all framework outputs:

1. **No policy limits or aggregates in base model.** Real insurance programs have finite policy limits and aggregate limits. The base model's unlimited coverage assumption overstates the guaranteed cost advantage for very large losses.

2. **No dynamic premium modeling.** Premiums are fixed at inception. Real premiums adjust based on loss experience, market conditions, and insurer capital constraints. Dynamic pricing could attenuate the growth advantage by introducing premium volatility.

3. **Frequency tail correlations not modeled.** The Poisson assumption implies that high volumes of attritional losses are statistically independent. In reality, many operational risks exhibit clustering. Such correlation would increase retained loss volatility, potentially shifting optimal deductibles lower.

4. **No regulatory capital requirements.** The model does not incorporate regulatory capital, rating agency capital adequacy standards, or debt covenant restrictions. For publicly traded companies, regulated entities, or leveraged businesses, these constraints may override growth optimization.

5. **Credit and liquidity constraints simplified.** The Letter of Credit mechanism assumes costless collateral posting. Real companies face higher collateral costs, limited credit facility capacity, cash flow constraints, and credit rating impacts from reserve increases.

6. **Loss control and risk engineering services not modeled.** Insurers provide loss control, risk engineering, and claims handling expertise that may reduce expected losses beyond the pure financial transfer.

7. **Stakeholder preferences not modeled.** Board, shareholder, or management preferences for earnings stability, dividend consistency, or other non-growth objectives are not captured.

8. **Reserve development uncertainty not modeled.** The model assumes ultimate claim amounts are known at inception. Real reserves are estimates subject to adverse or favorable development.

Not for Regulatory or Filing Purposes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Framework outputs **must not** be used for:

- Regulatory filings of any kind.
- Rate filings or rate adequacy demonstrations.
- Reserve opinions or statements of reserve adequacy.
- Statutory financial statements or audited financial reports.
- Statements of Actuarial Opinion required by regulators or contracts.
- Any purpose requiring compliance with the NAIC Annual Statement Instructions.

Any use of framework outputs for the above purposes requires a separate, independent actuarial analysis by a qualified actuary who satisfies the applicable Qualification Standards.


Uncertainty and Risk Cautions
------------------------------

*Per ASOP 41 SS3.4.1 and SS4.1.3(d)*

This framework's outputs are subject to significant uncertainty from multiple sources. Users should not treat simulation results as precise predictions.

Sources of Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~

**Parameter Uncertainty**
   Input parameters (loss frequency, severity, operating margins) are estimates, not known quantities. Small changes in these parameters can produce materially different outputs. The claim that "optimal insurance premiums can exceed expected losses by 200-500%" is sensitive to the assumed loss distribution parameters and should always be interpreted with appropriate uncertainty qualifications. Users should conduct sensitivity analysis across plausible parameter ranges.

**Model Uncertainty**
   The framework embeds structural choices (multiplicative dynamics, GBM revenue, Poisson frequency, log-normal severity) that may not match the true data-generating process for any specific company. Alternative model specifications could produce qualitatively different recommendations.

**Process Uncertainty (Stochastic Variation)**
   Even with correct parameters and model structure, individual simulation paths vary widely. The Monte Carlo engine produces distributional estimates, not point predictions. Convergence diagnostics should be monitored to ensure that results are statistically stable.

**Data Limitations**
   The framework relies on user-supplied parameters that may be estimated from limited data, use proxy data from dissimilar populations, or embed unknown biases. The quality of outputs cannot exceed the quality of inputs.

**Estimation Uncertainty in Key Claims**
   Headline claims in project documentation (e.g., "30-50% better long-term growth rates," "60-90% improved survival probability," "200-500% of expected losses") represent outcomes under specific illustrative parameter sets. These ranges should not be interpreted as general predictions. Actual outcomes for any specific company depend on company-specific parameters, the accuracy of assumptions, and the realization of stochastic processes.

Risk Cautions
~~~~~~~~~~~~~~

- **Results are illustrative, not predictive.** Numerical outputs depend on input assumptions and do not constitute forecasts of future performance.
- **Tail risk may be understated.** The log-normal severity model has lighter tails than Pareto or GPD alternatives. For exposures with catastrophic potential, results may understate the probability and magnitude of extreme losses.
- **Survival probabilities are model-dependent.** The reported ruin probabilities assume the model structure is correct. Actual survival rates depend on factors not captured in the model.
- **Sensitivity to assumptions.** Users should always run sensitivity analysis (varying key parameters individually and jointly) before drawing conclusions from any single model run.
- **The "insurance as growth enabler" conclusion requires ongoing validation.** While the ergodic framework provides a theoretically sound basis for this conclusion, its practical magnitude depends on empirical calibration that is ongoing.

.. important::

   This tool is intended for research purposes only. It is provided as-is with absolutely no guarantee of validity or correctness of methodology or results. All users and contributors are expected to do their own due diligence when relying on this tool for any research or analysis.


Reliance Statements
--------------------

*Per ASOP 41 SS3.4.3 and SS4.1.3(g)*

Under ASOP 41 SS3.4.3, the actuary who issues a communication assumes responsibility for it, except to the extent the actuary disclaims responsibility by stating reliance on other sources. The following reliance disclosures apply.

Reliance on User-Supplied Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework accepts user-supplied input parameters including financial data, loss history, operating margins, and insurance program specifications. **The responsible actuary does not assume responsibility for the accuracy, completeness, or appropriateness of user-supplied data.** Users are responsible for:

- Ensuring that input parameters are based on reliable data sources.
- Applying data quality standards consistent with ASOP 23.
- Performing reasonableness checks on input parameters before relying on outputs.
- Ensuring that selected parameters are appropriate for the intended use.

No automated reasonableness checks are applied to user inputs beyond basic validation of data types and ranges. The framework does not verify that user-selected parameters are consistent with observed experience, industry benchmarks, or actuarial standards.

Reliance on External Academic Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework's theoretical foundation relies on published academic research, including:

- Peters, O. (2019). "The ergodicity problem in economics." *Nature Physics*, 15, 1216-1221.
- Peters, O. and Gell-Mann, M. (2016). "Evaluating gambles using dynamics." *Chaos*, 26(2), 023103.
- Peters, O. (2011). "The time resolution of the St Petersburg paradox." *Phil. Trans. R. Soc. A*, 369(1956), 4913-4931.
- Kelly, J. L., Jr. (1956). "A new interpretation of information rate." *Bell System Technical Journal*, 35(4), 917-926.

These works provide the theoretical basis for the ergodic framework. The responsible actuary relies on the mathematical correctness of the published proofs and derivations in these works but has not independently verified them. The application of these theoretical results to insurance optimization is the contribution of this framework and remains subject to the limitations described in this document.

Standard actuarial techniques used in the framework (compound Poisson processes, log-normal severity models, chain ladder development, Kelly criterion) are drawn from established actuarial and statistical literature. References to specific actuarial standards (ASOP 13, 25, 43) in the codebase document the methodological basis for specific calculations.

Reliance on Large Language Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Development of this framework involved extensive reliance on Large Language Models (LLMs) for:

- Actuarial research and literature review.
- Financial and accounting research (GAAP compliance, ASC standards).
- Mathematical derivations and verification.
- Insurance industry research and parameter estimation.
- General business and economic research.
- Code writing, code review, and testing.

The primary models used were **Claude Opus 4** (Anthropic) and **Gemini Pro** (Google). LLM outputs were used as research aids and code generation tools, not as authoritative sources. The responsible actuary has reviewed LLM-generated content but has not independently verified all LLM-sourced research claims against primary sources. Users should be aware that LLM-assisted research may contain inaccuracies.

Reliance on Open-Source Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework relies on third-party open-source Python libraries for numerical computation, statistical analysis, and optimization. Key dependencies include NumPy, SciPy, Pandas, and Numba. The responsible actuary does not assume responsibility for the correctness of these libraries but has selected well-established, widely-used packages with active maintenance and community review.


Responsibility for Assumptions
-------------------------------

*Per ASOP 41 SS3.4.4 and SS4.3*

Each material assumption in this framework is the responsibility of one of the following parties:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Category
     - Examples
     - Responsible Party
   * - Framework-embedded assumptions
     - Poisson frequency, log-normal severity, GBM revenue, multiplicative dynamics, volatility drag, Kelly criterion application
     - Alex Filiakov (developer). These assumptions were selected for illustrative purposes through research conducted using Large Language Models and academic literature review.
   * - User-selected assumptions
     - Initial assets, loss parameters, operating margins, insurance program structure, time horizon, simulation count
     - The user who configures and runs the analysis. The responsible actuary does not assume responsibility for user-selected values.
   * - Simplifying assumptions
     - Poisson instead of ODP, no regulatory capital, deterministic margins, simplified tax treatment, no credit risk, fixed claim payment schedule
     - Alex Filiakov (developer). These simplifications were made to produce research outputs in a reasonable timeframe and at reasonable cost. Each is documented in the :ref:`deviation-disclosures` section.
   * - Default parameter values
     - Industry-specific defaults for manufacturing, service, and retail profiles
     - Alex Filiakov (developer). Defaults are illustrative and were estimated from public data and LLM-assisted research. Users are responsible for validating defaults against their own experience.

Where this communication is silent on the responsibility for a specific assumption, the responsible actuary (Alex Filiakov) is presumed responsible per ASOP 41 SS3.4.4.


.. _deviation-disclosures:

Deviation Disclosures
----------------------

*Per ASOP 41 SS4.4*

The following material deviations from standard actuarial practice are disclosed. For each deviation, the nature, rationale, expected effect, and applicable ASOP are stated.

**General Statement:** These deviations are made to simplify implementation in order to produce research outputs in a reasonable timeframe and at reasonable cost. This framework is not ready for enterprise use, where the below-stated material deviations would be rectified.

.. _deviation-poisson:

1. Poisson Instead of Over-Dispersed Poisson (ODP) Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** The framework uses a standard Poisson distribution for claim frequency rather than the Over-Dispersed Poisson (ODP) model that is preferred in actuarial practice for claim estimation.
- **ASOP(s) Affected:** ASOP 43 (Unpaid Claim Estimates), ASOP 25 (Credibility Procedures).
- **Rationale:** The Poisson model is simpler to implement and parameterize. ODP is preferred because claim estimation introduces uncertainty that increases variance beyond the Poisson assumption. For a research tool, the Poisson model provides a reasonable starting point.
- **Effect on Results:** The Poisson model understates the variance of claim counts, which may:

  - Understate the probability of unusually high or low claim years.
  - Lead to narrower confidence intervals than warranted.
  - Produce modestly optimistic retained loss estimates.
  - The directional effect is to slightly overstate the attractiveness of higher retentions.

.. _deviation-regulatory-capital:

2. No Regulatory Capital Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** The framework does not model regulatory capital requirements, rating agency capital adequacy standards, or debt covenant restrictions.
- **ASOP(s) Affected:** ASOP 56 (Modeling).
- **Rationale:** Regulatory capital requirements are entity-specific and vary widely by industry, jurisdiction, and organizational structure. Including them would require entity-specific customization beyond the scope of a general research tool.
- **Effect on Results:** For regulated entities, omitting capital constraints may produce optimal retentions that are infeasible in practice. The shadow price of regulatory capital can be substantial, potentially reversing the guaranteed cost advantage. Results for unregulated entities are not materially affected.

3. Deterministic Operating Margins
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** Operating margins are fixed constants rather than stochastic variables. No margin volatility, cyclicality, or correlation with loss experience is modeled.
- **ASOP(s) Affected:** ASOP 56 (Modeling).
- **Rationale:** Stochastic margin modeling adds significant complexity and parameterization burden. Fixed margins isolate the effect of insurance on growth dynamics, which is the research question of interest.
- **Effect on Results:** Understates total business volatility. The ergodic advantage may be larger (if margin volatility adds to the volatility drag that insurance mitigates) or smaller (if margin shocks are the dominant risk). Directional effect is ambiguous.

4. Simplified Tax Treatment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** A flat 25% corporate tax rate is applied. No progressive brackets, alternative minimum tax, international tax considerations, or detailed timing of tax payments.
- **ASOP(s) Affected:** General actuarial practice.
- **Rationale:** Tax code complexity is orthogonal to the ergodic insurance research question. The flat rate captures the first-order effect of taxation on retained earnings.
- **Effect on Results:** Minor. Tax simplification is unlikely to materially affect the relative ranking of insurance strategies, though it may modestly affect absolute growth rate estimates.

5. No Credit Risk Modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** Insurance recoveries above the deductible are certain. Insurer insolvency, coverage disputes, and claims handling delays are not modeled.
- **ASOP(s) Affected:** ASOP 56 (Modeling).
- **Rationale:** Insurer credit risk is a second-order effect for well-rated carriers. Modeling it requires insurer-specific financial data beyond the scope of a policyholder-focused research tool.
- **Effect on Results:** Overstates the certainty of insurance recoveries. In periods of insurance market stress or with poorly rated carriers, actual recoveries may fall short. This would reduce the ergodic advantage of insurance.

6. No Loss Development Beyond Fixed Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** Claims develop on a fixed payment schedule (cumulative from 10% in Year 1 to 100% by Year 10). No adverse or favorable development of ultimate loss estimates.
- **ASOP(s) Affected:** ASOP 43 (Unpaid Claim Estimates).
- **Rationale:** Reserve development uncertainty adds complexity without directly illuminating the ergodic insurance question. The fixed schedule represents a central estimate.
- **Effect on Results:** Understates the volatility of retained losses over time. Adverse development would increase the value of insurance (more volatility to mitigate); favorable development would decrease it.

7. Fixed Claim Payment Schedule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** All claims follow the same payment pattern regardless of claim size, type, or complexity.
- **ASOP(s) Affected:** ASOP 43 (Unpaid Claim Estimates).
- **Rationale:** Differentiated payment patterns add parameterization complexity. A single representative pattern is a reasonable starting assumption for a general model.
- **Effect on Results:** Minor for strategic-level analysis. More significant for cash flow and collateral cost analysis where payment timing matters.

8. No Correlation Between Loss Frequency and Business Growth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** Loss frequency is independent of revenue growth. A growing business does not experience proportionally more frequent losses beyond simple exposure scaling.
- **ASOP(s) Affected:** ASOP 56 (Modeling).
- **Rationale:** Isolates the insurance effect from business growth dynamics. Correlation modeling requires empirical data that varies widely by industry.
- **Effect on Results:** May understate risk for rapidly growing businesses where growth introduces new exposures, inexperienced staff, or strained quality controls. May overstate risk for mature businesses where growth does not proportionally increase loss frequency.

9. Ergodic (Time-Average) Framework Rather Than Traditional Ensemble Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Nature:** The framework optimizes time-average growth rates rather than expected values (ensemble averages). This is a deliberate methodological departure from traditional actuarial and financial analysis.
- **ASOP(s) Affected:** Novel methodology. Deviates from the traditional expected value framework used in most actuarial practice.
- **Rationale:** The deviation from standard methodology to use the ergodic approach is done intentionally to explore the time-average impact on insurance selection. This is the central research contribution of the framework. The theoretical basis is established in the academic literature (Peters 2019, Peters & Gell-Mann 2016).
- **Effect on Results:** Produces qualitatively different recommendations than ensemble-average optimization. Specifically, the ergodic framework recommends lower retentions and higher coverage limits than traditional analysis, and justifies premium loadings that would appear excessive under expected value analysis. The magnitude of the difference is an empirical question that depends on model parameters.


Conflict of Interest Disclosure
--------------------------------

*Per ASOP 41 SS3.4.2 and SS4.1.3(f)*

The responsible actuary, Alex Filiakov, is employed by an insurance broker that advocates on behalf of corporate clients in insurance placements and negotiations.

This employment relationship creates a potential conflict of interest: the framework's conclusions -- that insurance provides growth advantages and that higher coverage levels are often optimal -- are directionally consistent with the business interests of insurance brokers, who earn commissions or fees on insurance placements.

To mitigate this potential conflict:

- The framework's methodology is fully open-source and available for independent review at `https://github.com/AlexFiliakov/Ergodic-Insurance-Limits <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits>`_.
- The theoretical foundation (ergodic economics) is drawn from peer-reviewed academic literature independent of the insurance brokerage industry.
- The framework was developed as independent research, not as a product of the responsible actuary's employer.
- All assumptions, methods, and limitations are disclosed in this document for independent evaluation.

Users should consider this potential conflict when evaluating the framework's conclusions and are encouraged to conduct independent validation.


Information Dates
------------------

*Per ASOP 41 SS3.4.5 and SS4.1.3(h)*

- **Framework methodology current as of:** February 2026 (version reflected in CHANGELOG.md).
- **Illustrative parameters based on:** Public financial data and industry benchmarks available through LLM training data (cutoff varies by model) and web research conducted through January 2026.
- **Illustrative published results available at:** `https://applications.mostlyoptimal.com/ <https://applications.mostlyoptimal.com/>`_. Each published result is individually dated.
- **This compliance document current as of:** February 2026.
- **ASOP compliance target:** ASOP 41 as adopted December 2010, effective May 1, 2011.

The framework is presently undergoing validation and calibration through selective code review and research use. Users should verify that the version of the framework they are using reflects the most current methodology by checking the project's CHANGELOG.md and GitHub releases.

Any outputs generated by the framework should include the date of generation and the framework version used. Users are responsible for determining whether the methodology and parameters remain appropriate as of the date they rely on the outputs.


.. _applicability-of-outputs:

Applicability of Outputs as Actuarial Communications
-----------------------------------------------------

*Per ASOP 41 SS1.2*

ASOP 41 applies to actuarial communications that include actuarial opinions, recommendations, findings, or advice. Not all outputs of this framework constitute actuarial communications:

- **The Python framework itself** is not an actuarial communication. It is a software tool.
- **Documentation describing actuarial methods and conclusions** (including this document, the theory documentation, and the research paper) constitutes an actuarial communication when issued by an actuary with respect to actuarial services.
- **Reports generated by the framework** for intended users contain actuarial findings (optimal retentions, premium loading recommendations, risk metrics) and therefore constitute actuarial communications when the user relies on them for actuarial purposes.

**For framework-generated outputs:**

- Outputs are intended for research purposes only. The framework is provided as-is with absolutely no guarantees or support.
- Review is ongoing, and the reviewing actuary, Alex Filiakov, does not currently take responsibility for the accuracy of the methodology or the specific numerical outputs of the framework.
- Users relying on outputs for actuarial purposes should conduct their own due diligence and ensure their own ASOP 41 compliance for any communication they issue based on framework outputs.
- Any actuary who incorporates framework outputs into their own actuarial communication assumes responsibility for that communication per ASOP 41 SS3.1.3 and should independently validate the assumptions, methods, and results.


ASOP 56 Model Documentation Disclosures
-----------------------------------------

*Per ASOP 56 (Modeling)*

The following disclosures are required by ASOP 56 for model-based actuarial communications and supplement the ASOP 41 disclosures above.

Intended Purpose of the Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is intended to aid with **time-average insurance decision research**. Specifically, it provides a simulation framework for studying how ergodic (time-average) optimization produces different insurance purchasing recommendations than traditional ensemble-average (expected value) approaches. The model is not intended for production use, regulatory compliance, or financial reporting.

Material Assumption Inconsistencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following known inconsistencies between model assumptions and real-world conditions are documented:

- The model assumes stationary loss parameters while real-world loss distributions evolve over time due to inflation, exposure changes, and loss trend.
- The model assumes independent losses while real-world losses often exhibit serial correlation and clustering.
- The model assumes a capital-light business (no PP&E) while the widget manufacturing scenario implies capital-intensive operations.
- The model assumes no investment income while real businesses earn returns on invested assets.
- The model assumes instantaneous insurance recovery while real claims involve delays, disputes, and administrative costs.

Material Limitations and Known Weaknesses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the limitations documented above, the following model-specific weaknesses are acknowledged:

- **Single-entity focus.** The model simulates a single company. It does not capture portfolio effects, diversification benefits, or insurance market dynamics.
- **No behavioral modeling.** Management decisions (e.g., cutting costs, raising capital, changing strategy) in response to adverse experience are not modeled. The company follows a fixed operating model regardless of financial performance.
- **Limited loss distribution flexibility.** While the framework supports multiple distributions, the default and most-tested configuration uses Poisson/log-normal. More complex severity models (mixed exponential, Pareto, GPD) are available but less thoroughly validated.
- **Simplified balance sheet.** The double-entry ledger provides GAAP-aligned financial statements but omits many real-world balance sheet items (intangible assets, goodwill, complex debt structures, off-balance-sheet items).
- **No market cycle dynamics in base model.** While market cycle-aware pricing is available as a feature, the base analysis uses fixed premium rates. Real insurance markets exhibit hard/soft cycles that materially affect the cost and availability of coverage.

Reliance on External Models and Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following external models and methods are incorporated into the framework:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Model/Method
     - Source
     - Use in Framework
   * - Ergodic economics
     - Peters (2019), Peters & Gell-Mann (2016)
     - Core theoretical framework for time-average optimization
   * - Kelly criterion
     - Kelly (1956)
     - Foundation for optimal retention sizing
   * - Geometric Brownian Motion
     - Standard financial mathematics (Black-Scholes framework)
     - Revenue dynamics modeling
   * - Compound Poisson process
     - Standard actuarial science (Klugman, Panjer & Willmot)
     - Loss frequency-severity modeling
   * - Chain ladder development
     - Standard reserving technique (ASOP 43)
     - Claim development patterns
   * - Euler-Maruyama discretization
     - Standard numerical methods for SDEs
     - Simulation time-stepping
   * - Hamilton-Jacobi-Bellman PDE
     - Standard stochastic optimal control theory
     - HJB solver for optimal insurance selection
   * - Gelman-Rubin diagnostic
     - Gelman & Rubin (1992)
     - MCMC convergence assessment
   * - Bootstrap confidence intervals
     - Efron (1979)
     - Uncertainty quantification for simulation outputs

Model Validation Results
~~~~~~~~~~~~~~~~~~~~~~~~~

Model validation is conducted through the project's automated test suite, which is run as part of continuous integration (CI) on every code change. The test suite includes:

- **Unit tests** for all core mathematical functions and financial calculations.
- **Integration tests** for end-to-end simulation workflows.
- **Regression tests** to ensure consistency across code changes.
- **Convergence tests** to verify that Monte Carlo estimates stabilize.
- **Conservation tests** to verify double-entry accounting balance and financial statement consistency.

Current test suite status is available in the project's CI pipeline on GitHub.

Additionally, illustrative validation results are published at `https://applications.mostlyoptimal.com/ <https://applications.mostlyoptimal.com/>`_ as ongoing model validation.

Formal model validation by an independent party has not been completed. Users who require validated model outputs should conduct their own validation appropriate to their use case.

Model Governance and Controls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is provided as open-source software under the MIT license. Governance and controls consist of:

- **Version control** via Git, with all changes tracked in the repository history.
- **Code review** conducted by Alex Filiakov on all contributions.
- **Automated testing** via CI/CD pipelines on every commit and pull request.
- **Semantic versioning** with automated releases on the ``main`` branch.
- **Change documentation** in CHANGELOG.md per the Keep a Changelog format.

No formal model risk management framework (e.g., SR 11-7/OCC 2011-12 style model governance) is in place. The model is provided as-is with only ongoing reviews by Alex Filiakov as the sole governance mechanism.


Document History
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Date
     - Description
   * - February 2026
     - Initial ASOP 41 compliance document created, addressing all findings from the compliance audit (Issue #589).
