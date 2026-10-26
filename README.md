1️⃣ Business Context

    Unplanned machine failures are a major cost driver in manufacturing operations. They lead to:
    
    Production downtime
    
    Higher maintenance costs
    
    Delays in fulfilling customer orders
    
    Reduced overall operational efficiency
    
    Objective: Predict machine failures before they occur, enabling proactive maintenance and minimizing operational costs.

  2️⃣ Model Overview

    We developed a Random Forest Classifier to predict machine failures using operational sensor data:
    
    Air temperature [K]
    
    Process temperature [K]
    
    Rotational speed [rpm]
    
    Torque [Nm]
    
    Tool wear [min]
    
    Failure history flags: TWF, HDF, PWF, OSF, RNF
    
    Key Improvement: We incorporated the categorical feature Type (L, M, H) representing machine type.
    
    Why Type Matters:
    
    Different machine types have different risk profiles.
    
    Ignoring Type treats all machines equally, which reduces predictive accuracy.
    
    Including Type allows the model to tailor predictions for each machine category.
    
    Implementation (Simplified):
    
    ML models require numeric data, so Type was converted using One-Hot Encoding.

    Resulting features: Type_L, Type_M, Type_H (binary indicators).
    
    This ensures no artificial order is imposed (avoiding L < M < H bias).

  3️⃣ Model Results

    After including Type, the model achieved:
    
    High predictive accuracy on test data
    
    Strong classification performance, reliably detecting machines likely to fail
    
    Reduced false negatives, meaning fewer missed failures
    
    Business Interpretation:
    
    The model can proactively flag at-risk machines, allowing maintenance teams to intervene early.
    
    Adding Type improved predictive power, enabling better allocation of maintenance resources.

  4️⃣ Benefits of Implementation

    Implementing this model provides significant operational and financial advantages:
    
    Reduce Unplanned Downtime: Predict failures before they occur.
    
    Optimize Maintenance Costs: Focus resources on machines most likely to fail.
    
    Increase Machine Utilization: Fewer breakdowns improve overall throughput.
    
    High-Impact, Low-Effort Improvement: Adding Type delivers measurable gains without complex changes.
    
    Scalable & Future-Proof: The model can be updated as more machine data becomes available.

    5️⃣ Next Steps

    Deploy the model in production and integrate it with the maintenance scheduling system.
    
    Monitor performance metrics in real time (accuracy, false positives/negatives).
    
    Compare predicted vs actual failures to continuously refine the model.
    
    Optionally, analyze feature importance to understand which variables most influence failures.
    
    6️⃣ Projected Business Impact
    
    Metric	Projected Improvement
    Unplanned downtime	↓ 15–20%
    Maintenance costs	↓ 10–15%
    Operational efficiency	↑ 5–10%
    Revenue impact	+$X million (from fewer lost production hours)
7️⃣ Summary

“Incorporating machine Type into our predictive maintenance model significantly enhances predictive accuracy and business value. 
The model now differentiates risk across machine types, enabling targeted maintenance, reducing downtime, and lowering costs. 
With minimal implementation effort, this model provides measurable operational and financial benefits.
We recommend deploying it immediately as the backbone of a proactive maintenance strategy.”

---

## LLM Ranking: Structured Data Processing

`LLM_Structured_Data_Ranking.ipynb` benchmarks and ranks LLMs on structured-data tasks built from the same AI4I 2020 machine data. Every item has a ground truth computed from the data, so scoring is fully automatic.

| Module | Purpose |
|---|---|
| 1. Configuration | Seed, items per task, task weights, paths |
| 2. Data | AI4I 2020 loader (local CSV → KaggleHub → seeded synthetic fallback) |
| 3. Task suite | Log → JSON extraction, table QA, CSV → JSON transformation, text-to-SQL, rule-based failure classification |
| 4. Model registry & adapters | Anthropic SDK plus any OpenAI-compatible endpoint (OpenAI, Gemini, OpenRouter, Groq, Together, Mistral, DeepSeek, xAI, Ollama), plus calibration baselines |
| 5. Execution engine | Parallel calls, on-disk response cache, latency/token/cost tracking |
| 6. Scoring | JSON recovery, JSON-Schema validation, field/set metrics, sandboxed SQL execution |
| 7. Ranking | Weighted composite, bootstrap 95% CIs, rank probabilities, Bradley–Terry ratings, tiers |
| 8. Reporting | Leaderboard, task heatmap, cost-vs-quality Pareto chart, CSV/Markdown export |

Set provider keys before running, for example `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` with `OPENROUTER_MODELS="vendor/model-a,vendor/model-b"`. Module 4 has the full list. With no keys set, only the calibration baselines run, which checks the pipeline end to end.
