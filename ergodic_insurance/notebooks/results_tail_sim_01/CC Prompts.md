I have about 1000 different simulation configurations of these results, summarized for each run as:

```
{
  'Ded': 100000, # Deductible, does not vary for this experiment except for uninsured counterpart scenarios
  'LR': 0.3, # Loss Ratio, varies for this experiment between 0.3 and 0.7
  'Pol_Lim': 50000000, # Policy Limit, varies for this experiment from $50M up to $500M, as well as uninsured scenario
  'X_Th_%le': 0.0001, # This varies the extreme loss tail, the percentile of losses corresponding to the threshold of extreme loss, varies for this experiment
  'X_Shape': 1, # This is the shape of the extreme loss tail, varies for this experiment
  'X_Scale': 1, # This is the scale of the extreme loss tail, this is calibrated to match the 1/10th  of the threshold percentile corresponding to the catastrophic loss. In effect, this keeps the catastrophic shape the same except for extreme loss tail.
  'Sims': 100000, # Number of simulations, does not vary by run
  'Yrs': 25, # Number of simulated years (this is an ergodicity study), does not vary by run
  'growth_rate': np.float64(-0.00010129731595399839), # The scenario output of mean growth rate at the end of 25 years over all the scenarios
  'growth_rate_ci': {'0.5': np.float64(-1.495e-05), # The scenario output of growth rate quantiles at the end of 25 years over all the scenarios (quantiles .5, .6, .7, .8, .9, .95, .99)
   '0.6': np.float64(0.0002035),
   '0.7': np.float64(0.0004352),
   '0.8': np.float64(0.0006972),
   '0.9': np.float64(0.0010345),
   '0.95': np.float64(0.001300),
   '0.99': np.float64(0.001736)},
  'risk_of_ruin': {'5': 0.00016, # Risk of Ruin, or chance of insolvency, by a certain year (5, 10, 15, 20, 25)
   '10': 0.00034,
   '15': 0.00046,
   '20': 0.00056,
   '25': 0.00076}
}
```

I'd like to conduct a study of these results to analyze the effects of the loss tail on the corresponding insurance limits that should be purchased. I can conduct additional runs, or bootstrap tail losses excess of thresholds as needed. I can also report additional growth rate statistics.
Please help me design this study to get started. Recommend 3 tiers of analysis, starting from basic analysis, proceeding to more complicated analysis, and finally to the most complex study. For each, recommend any additional data needed.
I just want to study the effect of limits while keeping a reasonable deductible, so limit your recommendations to studies of insurance limits. I can also create layered insurance towers where policy loss ratio varies by attachment point.
For each study tier, recommend the charts needed to support that study. The charts should be poignant, but compelling and interesting. The charts need to be static, or dynamic configurations that I can take screenshots of once tailored.
Ask me clarifying questions, then design three studies as described above.
