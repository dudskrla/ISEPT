# [ICAIF 2025] ISEPT: Image-Based Selection and Execution Framework for Pair Trading

> Official implementation of **“ISEPT: Image-Based Selection and Execution Framework for Pair Trading (ICAIF ’25)”**  
> Paper: https://doi.org/10.1145/3768292.3770346

Pair trading aims to earn **market-neutral** profits by hedging two assets and exploiting temporary mispricings. However, many prior approaches **separate pair selection and execution**, which prevents trading outcomes from improving the next round of selection and can lead to overfitting. Moreover, using only raw/aggregated price series often fails to capture the **visual patterns** traders commonly rely on.

We propose **ISEPT**, an end-to-end framework that directly uses **candlestick chart images** and links **pair selection ↔ trading** through a **Sharpe-ratio feedback loop**. A **Convolutional Autoencoder (CAE)** encodes monthly candlestick images into stock-level latent vectors, and an **MLP** predicts the **next-month Sharpe ratio** for each candidate pair. At month-end, realized trading results are fed back as new training data so the model continuously adapts to market regime changes.

![Figure1](https://github.com/user-attachments/assets/b6052ce5-a936-4b67-b3a4-429fd5364231)

---

## Key Highlights

- **Image-based representation**: converts OHLC data into candlestick images to capture within-asset + cross-asset patterns. 
- **Unified selection & execution loop**: uses realized Sharpe ratios to retrain and refine pair rankings each month. 
- **Long-horizon evaluation**: tested on S&P 500 constituents with out-of-sample evaluation spanning ~20 years (2004–2024).

---

## Method Overview (ISEPT)

### (1) Candlestick Image Generation
- Slide a **21-trading-day window** (≈ 1 month) across the prior 12 months and render each candlestick images of size **64×64**.
- Apply **log scaling** to reduce price-scale distortions across tickers. 

### (2) Pair Selection via CAE → MLP
- **CAE** encodes candlestick images into latent vectors.
- For each stock, average embeddings over **T = 12 months**, then concatenate the two stocks’ vectors to form a pair representation.
- **MLP** predicts the pair’s **next-month Sharpe ratio**, ranking all candidate pairs and selecting the **Top-100** for trading.  

### (3) Sharpe-Ratio Feedback Loop (Adaptive Learning)
- At each month-end, compute realized performance and feed **top/bottom pairs’ Sharpe ratios** back into the MLP training set.
- This closes the loop: **selection → trading → realized Sharpe → retraining → improved selection**. 

---

## Experiments

### Dataset
- Daily **OHLC** for **S&P 500 constituents**, preprocessed from **Jan 1990–Dec 2024**.
- Out-of-sample evaluation period: **Jan 2004–Jun 2024**. 

### Baselines
- **GATEV (distance/SSD)** and **VIDYAMURTHY (correlation + Engle–Granger cointegration)** as classical pair-selection/trading baselines. 

### Main Results (2004–2024)
ISEPT-based strategies substantially improve ROI and risk-adjusted performance vs. classical methods (see Table 1 in the paper). 

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kim2025isept,
  title={ISEPT: Image-Based Selection and Execution Framework for Pair Trading},
  author={Kim, Nayoung and Lee, Jangwook and Kang, Yuncheol},
  booktitle={Proceedings of the 6th ACM International Conference on AI in Finance},
  pages={413--421},
  year={2025}
}
