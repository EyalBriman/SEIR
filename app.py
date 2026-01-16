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
import math  


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

def SE(demands: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Sequential Equalizing (SE) / progressive filling with forward-only storage.
    Feasibility is by prefix constraints:
        for all t:  sum_{tau<=t} sum_i w_i(tau) <= sum_{tau<=t} S(tau)
    """
    n, T = demands.shape
    w = np.zeros((n, T), dtype=float)

    active = set(range(n))  # agents not yet finalized
    p = 0                   # first (0-based) time index that is NOT saturated yet

    while active and p < T:
        cumS = np.cumsum(S)
        cumAlloc = np.cumsum(w.sum(axis=0))
        slack = cumS - cumAlloc  # remaining prefix slack

        ratios = [math.inf] * T
        for t in range(p, T):
            denom = 0.0
            for i in active:
                denom += demands[i, p:t+1].sum()  # only demand after current frontier p
            if denom > 0:
                ratios[t] = slack[t] / denom

        t_star = int(np.argmin(ratios))
        Delta = ratios[t_star]
        if not np.isfinite(Delta):
            break

        # Increase allocations proportionally from time p onward
        for i in active:
            w[i, p:] += Delta * demands[i, p:]

        # The bottleneck prefix up to t_star is now saturated; move frontier
        p = t_star + 1

        # Any agent with positive demand in the saturated prefix becomes final
        to_remove = [i for i in active if np.any(demands[i, :p] > 0)]
        for i in to_remove:
            active.remove(i)

    return w


# ============================
# DASE / SEIR
# ============================

def residualize_supply_forward_only(S_res: np.ndarray) -> np.ndarray:
    """
    Enforce forward-only storage feasibility for residual supply:
    all prefix sums must be nonnegative.
    We can carry surplus forward, but cannot borrow from the future.
    """
    S_res = S_res.astype(float).copy()
    running = 0.0
    for t in range(len(S_res)):
        running += S_res[t]
        if running < -1e-12:
            raise ValueError(
                "DA allocation violates prefix feasibility (borrows from the future). "
                "Cannot form a valid residual supply for forward-only storage."
            )
        # keep as-is; SE uses prefix sums anyway
    # You may still want to clamp tiny negatives due to numerical noise:
    S_res = np.maximum(S_res, 0.0)
    return S_res


def DASE(demands: np.ndarray, S: np.ndarray, w_DA: np.ndarray) -> np.ndarray:
    """
    DASE = DA (IR part) + SE run on the residual instance:
      - residual supply: S_res(t) = S(t) - sum_i w_DA(i,t)
      - residual demands: d_res(i,t) = max(d(i,t) - w_DA(i,t), 0)
    Then return w_DA + w_SE_res.
    """
    # 1) residual supply per time step
    S_res = S - w_DA.sum(axis=0)
    S_res = residualize_supply_forward_only(S_res)

    # 2) residual demands per agent/time (what is still missing)
    d_res = np.maximum(demands - w_DA, 0.0)

    # 3) run SE on the residual instance
    w_SE_res = SE(d_res, S_res)

    # 4) combine
    return w_DA + w_SE_res


def alpha_from_alloc(w: np.ndarray, d: np.ndarray) -> float:
    """
    Leontief-style alpha: min_{t: d(t)>0} w(t)/d(t).
    If an agent has no positive demands (shouldn't happen in your validation), return +inf.
    """
    mask = d > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(w[mask] / d[mask]))



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
    placeholder="e.g.\n1 2 1 0\n3 1 0 0\n0 2 2 0",
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




