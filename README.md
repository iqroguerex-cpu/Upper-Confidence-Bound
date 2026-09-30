# 🎯 REX-UCB — Multi-Armed Bandit Optimizer

An interactive reinforcement learning web app built with **Streamlit** implementing the **Upper Confidence Bound (UCB1)** algorithm for real-time Click-Through Rate (CTR) optimization.

---

## 🚀 Features

- **Dynamic Data Support**: Upload any custom binary bandit CSV or use the built-in 10-arm simulation.
- **Tunable Exploration**: Control the exploration parameter ($c$) via an interactive slider.
- **Live Analytics**: Real-time metrics, selection histograms, and cumulative reward tracking.

---

## 🧮 How It Works

At each round $t$, the algorithm selects arm $i$ that maximizes:

$$\text{UCB}_i(t) = \bar{X}_i + \sqrt{\frac{c \cdot \ln(t)}{N_i}}$$

- **$\bar{X}_i$**: Average observed reward (exploitation)
- **$\sqrt{\frac{c \cdot \ln(t)}{N_i}}$**: Confidence bonus (exploration)

---

## 🛠️ Quick Start

```bash
# 1. Clone repository
git clone [https://github.com/iqroguerex-cpu/rex-ucb.git](https://github.com/iqroguerex-cpu/rex-ucb.git)
cd rex-ucb

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
streamlit run app.py

```
## Project Structure

```text
rex-ucb/
├── app.py                      # Streamlit application
├── Ads_CTR_Optimisation.csv    # Sample dataset
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## 👨‍💻 Author

Chinmay V Chatradamath
