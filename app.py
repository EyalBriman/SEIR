# app.py
# Streamlit app for:
#   DA   (Decentralized with infinite storage, per-agent LP)
#   SE   (Sequential Equalizing – progressive filling)
#   DASE (DA + SE on residual supply)
#
# Assumes infinite storage.
# No built-in examples. User must input data.

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
import pulp


# ============================
# Parsing & validation
# ============================

def parse_matrix(text: str) -> np.ndarray:
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    data = []
    for r in rows:
        r = r.replace(",", " ")
        data.append([float(x) for x in r.split()])
    return np.array(data, dtype=float)


def parse_vector(text: str) -> np.ndarray:
    arr = parse_matrix(text)
    if arr.shape[0] == 1:
        return arr.flatten()
    if arr.shape[1] == 1:
        return arr.flatten()
    raise ValueError("Supply must be a single row or column.")


def validate(d: np.ndarray, S: np.ndarray):
    if d.ndim != 2:
        raise ValueError("Demands must be a matrix.")
    if S.ndim != 1:
        raise ValueError("Supply must be a vector.")
    if d.shape[1] != len(S):
        raise ValueError("Supply length must equal number of time steps.")

    if np.any(d < 0) or np.any(S < 0):
        raise ValueError("All values must be non-negative.")

    row_sums = d.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each agent must have positive total demand.")

    if not np.allclose(row_sums, row_sums[0]):
        raise ValueError("All demand rows must sum to the same total.")


# ============================
# DA (per-agent LP, infinite storage)
# ============================

def da_single_agent(d_i: np.ndarray, S: np.ndarray, n: int):
    """
    Agent receives S(t)/n each time.
    Chooses allocation w(t) and carry y(t) to maximize beta.
    """
    T = len(S)
    Si = S / n

    prob = pulp.LpProblem("DA", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", range(T), lowBound=0)
    y = pulp.LpVariable.dicts("y", range(T), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=0)

    prob += beta

    prob += w[0] + y[0] <= Si[0]
    for t in range(1, T):
        prob += w[t] + y[t] <= Si[t] + y[t - 1]

    for t in range(T):
        if d_i[t] > 0:
            prob += beta * d_i[t] <= w[t]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return np.zeros(T), 0.0

    w_val = np.array([pulp.value(w[t]) for t in range(T)])
    beta_val = float(pulp.value(beta))
    return w_val, beta_val


def DA(demands: np.ndarray, S: np.ndarray):
    n, T = demands.shape
    alloc = np.zeros((n, T))
    beta = np.zeros(n)

    for i in range(n):
        alloc[i], beta[i] = da_single_agent(demands[i], S, n)

    return alloc, beta


# ============================
# SE (progressive filling, EXACT)
# ============================

def SE(demands: np.ndarray, S: np.ndarray):
    n, T = demands.shape
    w = np.zeros((n, T))

    N = set(range(n))
    N_star = set()

    while N_star != N:
        # cumulative ratios
        ratios = []
        for t in range(T):
            Dhat = sum(demands[i, :t+1].sum() for i in N - N_star)
            Shat = sum(S[:t+1])
            ratios.append(np.inf if Dhat == 0 else Shat / Dhat)

        t_star = int(np.argmin(ratios))
        alpha = ratios[t_star]

        for i in N - N_star:
            for t in range(t_star + 1, T):
                w[i, t] += alpha * demands[i, t]

            if np.any(demands[i, :t_star + 1] > 0):
                for t in range(t_star + 1):
                    w[i, t] = alpha * demands[i, t]
                N_star.add(i)

    return w


# ============================
# DASE / SEIR
# ============================

def DASE(demands: np.ndarray, S: np.ndarray, w_DA: np.ndarray):
    S_res = S - w_DA.sum(axis=0)

    for t in range(len(S_res) - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    S_res = np.maximum(S_res, 0.0)
    w_SE = SE(demands, S_res)
    return w_DA + w_SE


def alpha_from_alloc(w: np.ndarray, d: np.ndarray):
    mask = d > 0
    return np.min(w[mask] / d[mask])


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="DA / SE / DASE", layout="wide")
st.title("DA / SE / DASE — Infinite Storage")

st.markdown("""
Insert your instance below.

• Demands: rows = agents, columns = time steps  
• Supply: one row or column  
• All demand rows must sum to the same value  
""")

dem_text = st.text_area(
    "Demands",
    placeholder="e.g.\n1 2 1 0\n3 1 0 0\n0 2 1 0",
    height=180,
)

sup_text = st.text_area(
    "Supply",
    placeholder="e.g.\n36 36 36 42",
    height=80,
)

if st.button("Compute allocations", type="primary"):
    try:
        demands = parse_matrix(dem_text)
        supply = parse_vector(sup_text)
        validate(demands, supply)

        w_DA, beta_DA = DA(demands, supply)
        w_SE = SE(demands, supply)
        w_DASE = DASE(demands, supply, w_DA)

        alpha_DA = [alpha_from_alloc(w_DA[i], demands[i]) for i in range(len(demands))]
        alpha_SE = [alpha_from_alloc(w_SE[i], demands[i]) for i in range(len(demands))]
        alpha_DASE = [alpha_from_alloc(w_DASE[i], demands[i]) for i in range(len(demands))]

        idx = [f"agent_{i}" for i in range(len(demands))]
        cols = [f"t{t}" for t in range(demands.shape[1])]

        tab1, tab2, tab3 = st.tabs(["SE", "DA", "DASE"])

        with tab1:
            st.dataframe(pd.DataFrame(w_SE, index=idx, columns=cols))
            st.write("Alphas:", alpha_SE)

        with tab2:
            st.dataframe(pd.DataFrame(w_DA, index=idx, columns=cols))
            st.write("Alphas:", alpha_DA)

        with tab3:
            st.dataframe(pd.DataFrame(w_DASE, index=idx, columns=cols))
            st.write("Alphas:", alpha_DASE)

    except Exception as e:
        st.error(str(e))

