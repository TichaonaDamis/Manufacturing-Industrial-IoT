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

## End-to-End ML Pipeline: 30 Algorithms on AI4I 2020

`Manufacturing_ML_Pipeline_30_Algorithms.ipynb` downloads the free [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) from UCI and applies every algorithm in a full ML workflow:

1. **Data acquisition and EDA**: class balance, sensor distributions, correlations.
2. **Data quality**: removes leakage (the failure-mode flags) and IDs, reviews outliers, and handles **missing values**. Simulated sensor drop-outs are filled by mean, median, KNN or iterative imputation, and the strategy is chosen by CV.
3. **Feature engineering**: temperature gap, power, strain, and type-normalised strain, with an ablation test.
4. **Dimensionality reduction and clustering**: PCA, t-SNE, k-Means vs K-Means++, Hierarchical, DBSCAN, each tuned.
5. **Feature selection**: Genetic Algorithm.
6. **Training and hyper-parameter tuning**: randomised search with 5-fold stratified CV for 19 candidate models. These include Linear/Logistic Regression, Decision Tree, Random Forest, SVM, k-NN, Naive Bayes, Gradient Boosting, AdaBoost, XGBoost, ANN, CNN, RNN, LSTM, Transformer, Autoencoder and Isolation Forest.
7. **Validation and selection**: out-of-fold metrics, cost-optimal thresholds, and corrected paired t-tests. A selection rule fixed in advance picks the model, which is then evaluated once on a held-out test set and saved.
8. **Maintenance policy optimisation**: MDP value iteration, Q-Learning, SARSA, DQN, Policy Gradient and Actor-Critic in a tool-replacement simulator built from the data.
9. **Summary**: all 30 algorithms ranked within their stage.

Set `FAST_MODE=1` for a quick run.
