# app.py
# Streamlit app for:
#   - SE  (Sequential Equalizing — progressive filling, infinite storage)
#   - DA  (Decentralized: S(t)/n + optimal individual storage)
#   - DASE / SEIR = DA + SE on residual supply
#
# Requirements:
#   pip install streamlit numpy pandas pulp openpyxl
#
# Run:
#   streamlit run app.py

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import pulp


# ============================
# Parsing helpers
# ============================

def parse_matrix_from_text(text: str) -> np.ndarray:
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    data = []
    for r in rows:
        r = r.replace(",", " ")
        data.append([float(x) for x in r.split()])
    return np.array(data, dtype=float)


def parse_vector_from_text(text: str) -> np.ndarray:
    arr = parse_matrix_from_text(text)
    if arr.shape[0] == 1:
        return arr.reshape(-1)
    if arr.shape[1] == 1:
        return arr.reshape(-1)
    raise ValueError("Supply must be a single row or a single column.")


def validate_inputs(demands: np.ndarray, supply: np.ndarray):
    if demands.ndim != 2:
        raise ValueError("Demands must be a 2D matrix (agents × time).")
    n, T = demands.shape

    if supply.ndim != 1 or len(supply) != T:
        raise ValueError("Supply must be a vector of length T.")

    if np.any(demands < 0) or np.any(supply < 0):
        raise ValueError("Demands and supply must be non-negative.")

    row_sums = demands.sum(axis=1)
    if not np.allclose(row_sums, row_sums[0]):
        raise ValueError("All demand rows must sum to the same value (normalized).")

    return n, T


# ============================
# Core algorithms
# ============================

def sequential_equalizing(demands: np.ndarray, supply: np.ndarray):
    """
    Sequential Equalizing (SE)
    EXACT progressive-filling algorithm from the paper.
    """
    n, T = demands.shape
    w = np.zeros((n, T))
    N = set(range(n))
    N_star = set()

    while N_star != N:
        # cumulative supply
        hatS = np.cumsum(supply)

        # cumulative demands
        hatD = np.cumsum(np.sum(demands, axis=0))

        Q = []
        for t in range(T):
            if hatD[t] > 0:
                Q.append(hatS[t] / hatD[t])
            else:
                Q.append(np.inf)

        t_star = int(np.argmin(Q))
        alpha = Q[t_star]

        for i in (N - N_star):
            # allocate from t*+1 onward
            for t in range(t_star + 1, T):
                w[i, t] += alpha * demands[i, t]

            # fix agent if they have demand up to t*
            if np.any(demands[i, :t_star + 1] > 0):
                for t in range(t_star + 1):
                    w[i, t] = alpha * demands[i, t]
                N_star.add(i)

    return w


def da_single_agent_lp(d_i: np.ndarray, supply: np.ndarray, n: int):
    """
    DA for a single agent:
    gets S(t)/n each time, chooses storage to maximize alpha.
    """
    T = len(d_i)
    S_i = supply / n

    prob = pulp.LpProblem("DA_agent", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("y", range(T), lowBound=0)
    alpha = pulp.LpVariable("alpha", lowBound=0)

    prob += alpha

    prob += w[0] + y[0] <= S_i[0]
    for t in range(1, T):
        prob += w[t] + y[t] <= S_i[t] + y[t - 1]

    for t in range(T):
        if d_i[t] > 0:
            prob += alpha * d_i[t] <= w[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return np.zeros(T)

    return np.array([pulp.value(w[t]) for t in range(T)])


def da_allocation(demands: np.ndarray, supply: np.ndarray):
    n, T = demands.shape
    w = np.zeros((n, T))
    for i in range(n):
        w[i] = da_single_agent_lp(demands[i], supply, n)
    return w


def dase(demands: np.ndarray, supply: np.ndarray):
    """
    DASE / SEIR:
      1) DA
      2) residual supply
      3) push negatives backward
      4) SE on residual
    """
    w_da = da_allocation(demands, supply)
    S_res = supply - np.sum(w_da, axis=0)

    for t in range(len(S_res) - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    S_res = np.maximum(S_res, 0.0)
    w_se = sequential_equalizing(demands, S_res)

    return w_da + w_se


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="SE / DA / DASE Allocator", layout="wide")
st.title("Sequential Equalizing (SE) and DASE Allocator")

st.markdown(
    """
**Instructions**

• Enter **demands** as a matrix: rows = agents, columns = time steps  
• All rows must sum to the **same value** (normalized demands)  
• Enter **supply** as a single row or column of length T  
• Infinite storage is assumed  

No example is pre-filled — insert your own instance.
"""
)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands")
    demands_text = st.text_area(
        "Paste demands matrix",
        height=200,
        placeholder="e.g.\n1 2 1 0 0\n3 1 0 0 0\n0 2 1 0 1\n..."
    )

with col2:
    st.subheader("Supply")
    supply_text = st.text_area(
        "Paste supply vector",
        height=200,
        placeholder="e.g.\n36 36 36 42 50"
    )

run = st.button("Compute allocations", type="primary")

if run:
    try:
        demands = parse_matrix_from_text(demands_text)
        supply = parse_vector_from_text(supply_text)

        n, T = validate_inputs(demands, supply)

        w_se = sequential_equalizing(demands, supply)
        w_da = da_allocation(demands, supply)
        w_dase = dase(demands, supply)

        idx = [f"agent_{i}" for i in range(n)]
        cols = [f"t{t}" for t in range(T)]

        st.subheader("SE Allocation")
        st.dataframe(pd.DataFrame(w_se, index=idx, columns=cols))

        st.subheader("DA Allocation")
        st.dataframe(pd.DataFrame(w_da, index=idx, columns=cols))

        st.subheader("DASE Allocation")
        st.dataframe(pd.DataFrame(w_dase, index=idx, columns=cols))

    except Exception as e:
        st.error(str(e))
