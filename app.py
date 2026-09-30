import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="UCB Multi-Armed Bandit", page_icon="📈", layout="wide"
)

st.title("🎯 Upper Confidence Bound (UCB) Optimiser")
st.markdown(
    """
This app demonstrates the **Upper Confidence Bound (UCB1)** reinforcement learning algorithm 
for multi-armed bandits (e.g., Click-Through Rate optimization).
"""
)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Ads CSV (Optional)", type=["csv"]
)

# Exploration hyperparameter multiplier (Default 1.5 corresponds to 3/2 in standard UCB)
exploration_scale = st.sidebar.slider(
    "Exploration Factor (c)",
    min_value=0.5,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Higher values favor exploring less-tested ads; lower values favor exploiting current best ads.",
)

# ----------------- DATA LOADING -----------------
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.sidebar.success("Loaded uploaded dataset!")
else:
    st.sidebar.info("Using synthetic dataset with 10 ads and 10,000 rounds.")
    # Reproducible default simulation matching standard 10-arm CTR problems
    np.random.seed(42)
    sample_rates = [
        0.08,
        0.12,
        0.04,
        0.11,
        0.27,
        0.02,
        0.11,
        0.21,
        0.07,
        0.05,
    ]  # Ad 5 is best (~27%)
    rounds = 10000
    sim_data = {
        f"Ad {i+1}": np.random.binomial(n=1, p=p, size=rounds)
        for i, p in enumerate(sample_rates)
    }
    dataset = pd.DataFrame(sim_data)

# ----------------- RUN UCB ALGORITHM -----------------
N = dataset.shape[0]
d = dataset.shape[1]
ad_names = list(dataset.columns)

ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
cumulative_rewards = []

for n in range(0, N):
    ad = 0
    max_upper_bound = -1.0

    for i in range(0, d):
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(
                exploration_scale * math.log(n + 1) / numbers_of_selections[i]
            )
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400

        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = int(dataset.values[n, ad])
    sums_of_rewards[ad] += reward
    total_reward += reward
    cumulative_rewards.append(total_reward)

best_arm_idx = max(range(d), key=lambda i: numbers_of_selections[i])

# ----------------- METRICS DASHBOARD -----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rounds", f"{N:,}")
col2.metric("Arms / Ads", d)
col3.metric("Total Clicks (Rewards)", f"{total_reward:,}")
col4.metric(
    "Overall Conversion Rate", f"{(total_reward / N) * 100:.2f}%"
)

st.success(
    f"🏆 **Optimal Option Selected:** `{ad_names[best_arm_idx]}` (Selected {numbers_of_selections[best_arm_idx]:,} times)"
)

# ----------------- VISUALIZATIONS -----------------
tab1, tab2, tab3 = st.tabs(
    ["Selections Histogram", "Cumulative Performance", "Data Summary"]
)

with tab1:
    st.subheader("Distribution of Ad Selections")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(
        ad_names, numbers_of_selections, color="#2b5c8f", edgecolor="black"
    )
    bars[best_arm_idx].set_color("#28a745")  # Highlight winner in green
    ax.set_ylabel("Number of Times Selected")
    ax.set_xlabel("Ad")
    ax.set_title("Action Allocation Across Rounds")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    st.pyplot(fig)

with tab2:
    st.subheader("Cumulative Clicks Over Time")
    perf_df = pd.DataFrame(
        {"Round": range(1, N + 1), "Cumulative Clicks": cumulative_rewards}
    )
    st.line_chart(perf_df.set_index("Round"))

with tab3:
    st.subheader("Performance Breakdown per Arm")
    breakdown_df = pd.DataFrame(
        {
            "Ad": ad_names,
            "Times Selected": numbers_of_selections,
            "Total Clicks": sums_of_rewards,
            "Observed CTR (%)": [
                (
                    (sums_of_rewards[i] / numbers_of_selections[i]) * 100
                    if numbers_of_selections[i] > 0
                    else 0.0
                )
                for i in range(d)
            ],
        }
    )
    st.dataframe(
        breakdown_df.style.format({"Observed CTR (%)": "{:.2f}%"}),
        use_container_width=True,
    )
