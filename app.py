# app.py
# Streamlit UI for:
#   (1) SE   (Sequential Equalizing — progressive filling on a supply vector, prefix-feasible)
#   (2) DA   (Decentralized: each agent gets S(t)/n as private stock; chooses consumption plan to maximize beta)
#   (3) DASE (DA consumption + SE on residual supply after push-back)
#
# Run:
#   pip install streamlit pandas numpy openpyxl pulp
#   streamlit run app.py

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import pulp


# ============================
# Parsing helpers
# ============================

def parse_matrix_from_text(text: str) -> np.ndarray:
    if text.strip() == "":
        raise ValueError("Empty input.")
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    data = []
    for r in rows:
        r = r.replace(",", " ")
        parts = [p for p in r.split() if p != ""]
        data.append([float(x) for x in parts])
    arr = np.array(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Demands must be a 2D matrix.")
    return arr


def parse_vector_from_text(text: str) -> np.ndarray:
    arr = parse_matrix_from_text(text)
    if arr.shape[0] == 1:
        return arr.reshape(-1).astype(float)
    if arr.shape[1] == 1:
        return arr.reshape(-1).astype(float)
    raise ValueError("Supply must be a single row or a single column.")


def read_excel(file) -> Tuple[np.ndarray, np.ndarray]:
    xls = pd.ExcelFile(file)
    sheet_names_lower = [s.lower() for s in xls.sheet_names]

    if "demands" in sheet_names_lower:
        d_sheet = xls.sheet_names[sheet_names_lower.index("demands")]
        demands_df = pd.read_excel(xls, sheet_name=d_sheet, header=None)
    else:
        demands_df = pd.read_excel(xls, sheet_name=0, header=None)

    if "supply" in sheet_names_lower:
        s_sheet = xls.sheet_names[sheet_names_lower.index("supply")]
        supply_df = pd.read_excel(xls, sheet_name=s_sheet, header=None)
    else:
        supply_df = pd.read_excel(xls, sheet_name=1 if len(xls.sheet_names) > 1 else 0, header=None)

    demands = (
        demands_df.apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .dropna(axis=1, how="all")
        .values.astype(float)
    )
    supply_mat = (
        supply_df.apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .dropna(axis=1, how="all")
        .values.astype(float)
    )

    if supply_mat.shape[0] == 1:
        supply = supply_mat.reshape(-1)
    elif supply_mat.shape[1] == 1:
        supply = supply_mat.reshape(-1)
    else:
        raise ValueError("In Excel, supply must be a single row or a single column.")

    return demands, supply


def validate_inputs(demands: np.ndarray, supply: np.ndarray) -> Tuple[int, int]:
    if demands.ndim != 2:
        raise ValueError("Demands must be a 2D table (agents x time-steps).")
    n, T = demands.shape

    if supply.ndim != 1:
        raise ValueError("Supply must be a 1D vector.")
    if len(supply) != T:
        raise ValueError(f"Supply length ({len(supply)}) must match number of time steps in demands ({T}).")

    if np.any(demands < 0):
        raise ValueError("Demands must be >= 0 everywhere.")
    if np.any(supply < 0):
        raise ValueError("Supply must be >= 0 everywhere.")

    row_sums = demands.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each agent must have a positive total demand (row sum > 0).")

    tol = 1e-8
    if float(np.max(row_sums) - np.min(row_sums)) > tol:
        raise ValueError(
            f"All demand rows must have the same sum. "
            f"Got min={np.min(row_sums):.6g}, max={np.max(row_sums):.6g}."
        )

    return n, T


# ============================
# Storage feasibility (prefix constraints)
# ============================

def check_prefix_feasible(alloc: np.ndarray, supply: np.ndarray, name: str):
    used = np.cumsum(np.sum(alloc, axis=0))
    cap = np.cumsum(supply)
    bad = np.where(used > cap + 1e-8)[0]
    if len(bad) > 0:
        t = int(bad[0])
        raise ValueError(
            f"{name} is prefix-infeasible at t={t}: "
            f"used_prefix={used[t]:.6g} > supply_prefix={cap[t]:.6g}."
        )


# ============================
# DA: per-agent LP with infinite local storage
# ============================

def da_single_agent_lp(d_i: np.ndarray, S: np.ndarray, n: int):
    """
    DA interpretation (infinite local storage):

    - At each time t, agent i receives Si(t)=S(t)/n into their private stock.
    - Agent chooses consumption w(t) and remaining stock y(t) (carry to next time).
    - Objective: maximize beta such that beta*d_i(t) <= w(t) for all t with d_i(t)>0.

    Inventory constraints (no borrowing from the future):
        w(0) + y(0) <= Si(0)
        for t>=1: w(t) + y(t) <= Si(t) + y(t-1)
        w,y >= 0
    """
    T = len(S)
    Si = S.astype(float) / float(n)
    d = d_i.astype(float).reshape(-1)

    prob = pulp.LpProblem("DA_single_agent", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", list(range(T)), lowBound=0)
    y = pulp.LpVariable.dicts("y", list(range(T)), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=0)

    prob += beta

    prob += (w[0] + y[0] <= Si[0]), "inv_0"
    for t in range(1, T):
        prob += (w[t] + y[t] <= Si[t] + y[t - 1]), f"inv_{t}"

    for t in range(T):
        if d[t] > 0:
            prob += (beta * d[t] <= w[t]), f"beta_demand_{t}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        return np.zeros(T), 0.0

    w_arr = np.array([pulp.value(w[t]) for t in range(T)], dtype=float)
    beta_val = float(pulp.value(beta))
    return w_arr, beta_val


def run_DA(demands: np.ndarray, supply: np.ndarray):
    n, T = demands.shape
    alloc_da = np.zeros((n, T), dtype=float)
    beta_da = np.zeros(n, dtype=float)
    for i in range(n):
        alloc_da[i], beta_da[i] = da_single_agent_lp(demands[i], supply, n)
    # DA is only required to be prefix-feasible globally (which it is, because everyone only uses their own prefix stock)
    check_prefix_feasible(alloc_da, supply, "DA")
    return alloc_da, beta_da


# ============================
# SE: progressive filling on a supply vector (matches your multi-iteration example)
# ============================

def sequential_equalizing_progressive(demands: np.ndarray, supply_vec: np.ndarray):
    """
    SE (progressive / multi-iteration), consistent with your worked example:

    Maintain current allocations w and remaining set R=N\\N*.
    At each iteration:
      1) compute SURPLUS per time: s_sur(t) = supply_vec(t) - sum_i w_i(t)
      2) compute cumulative surplus hat{s_sur}(t)
      3) compute cumulative remaining demand hat{d_R}(t)
      4) pick t* minimizing Q(t)=hat{s_sur}(t)/hat{d_R}(t) over hat{d_R}(t)>0
      5) let delta = Q(t*)
      6) for each i in R: add delta*d_i(t) for all t >= t*
      7) update each remaining agent's alpha_i += delta
      8) finalize agents who have any demand in prefix <= t* by setting w_i(t)=alpha_i*d_i(t) for all t<=t*
         and adding them to N*.

    This uses ONLY surplus that remains after what was already allocated, so it never overspends.
    Feasibility is guaranteed in prefix sense if supply_vec is prefix-feasible (nonnegative cumulative).
    """
    n, T = demands.shape
    S = supply_vec.astype(float).reshape(-1)
    w = np.zeros((n, T), dtype=float)

    N_star = set()
    alpha_i = np.zeros(n, dtype=float)

    def cum(x):
        return np.cumsum(x)

    for _ in range(n + T + 10):
        if len(N_star) == n:
            break

        R = [i for i in range(n) if i not in N_star]

        # Current per-time surplus and cumulative surplus
        s_sur = S - np.sum(w, axis=0)
        hat_sur = cum(s_sur)

        # If cumulative surplus goes negative, the supply_vec itself is not prefix-feasible given current w
        if np.min(hat_sur) < -1e-8:
            # This should not happen if we compute delta correctly, but guard anyway
            raise ValueError(
                "SE internal error: cumulative surplus became negative. "
                "This indicates inconsistent supply_vec / previous allocations."
            )

        # Remaining cumulative demand
        dR_per_t = np.sum(demands[R, :], axis=0)
        hat_dR = cum(dR_per_t)

        # Compute Q(t)
        Q = []
        for t in range(T):
            if hat_dR[t] <= 1e-12:
                Q.append(float("inf"))
            else:
                Q.append(float(hat_sur[t] / hat_dR[t]))

        t_star = int(np.argmin(Q))
        if Q[t_star] == float("inf"):
            # No remaining demand anywhere
            for i in R:
                N_star.add(i)
            continue

        delta = float(Q[t_star])
        if delta < -1e-12:
            raise ValueError("SE found a negative delta, which should be impossible under prefix-feasible surplus.")

        # Add delta * demand for remaining agents from t_star..T-1 (1-based t* means include it)
        for i in R:
            w[i, t_star:] += delta * demands[i, t_star:]
            alpha_i[i] += delta

        # Finalize agents that have demand in prefix up to t_star
        for i in R:
            if np.any(demands[i, : t_star + 1] > 0):
                # Ensure prefix exactly matches alpha_i * demand (this is the “fixing” step)
                w[i, : t_star + 1] = alpha_i[i] * demands[i, : t_star + 1]
                N_star.add(i)

    # Final feasibility check in prefix sense
    check_prefix_feasible(w, S, "SE (on given supply vector)")
    return w


# ============================
# DASE: DA + SE on residual after push-back
# ============================

def push_back_negatives(S_res: np.ndarray) -> np.ndarray:
    """
    Given residual per-time supply that may have negatives, push each negative backward:
      for t=T-1..1:
        if S_res[t] < 0:
           S_res[t-1] += S_res[t]
           S_res[t] = 0
    This preserves cumulative residual and ensures nonnegativity per-time (except possibly at t=0, which must be >=0).
    """
    S_res = S_res.astype(float).copy()
    T = len(S_res)
    for t in range(T - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0
    if S_res[0] < -1e-9:
        raise ValueError(f"Residual infeasible even after push-back: S_res[0]={S_res[0]:.6g} < 0.")
    return np.maximum(S_res, 0.0)


def dase(demands: np.ndarray, supply: np.ndarray, da_alloc: np.ndarray):
    """
    DASE:
      w_IR := DA consumption plans
      S' := S - sum_i w_IR_i
      push-back negatives in S'
      w* := SE(d, S')
      return w_IR + w*
    """
    w_ir = da_alloc.astype(float).copy()
    S = supply.astype(float).reshape(-1)

    # residual (may be negative at some times, but cumulative should be >=0 if w_ir is prefix-feasible)
    S_res = S - np.sum(w_ir, axis=0)

    # make residual per-time nonnegative by push-back (as in your pseudocode)
    S_res_pb = push_back_negatives(S_res)

    # run progressive SE on residual surplus
    w_star = sequential_equalizing_progressive(demands, S_res_pb)

    w_total = w_ir + w_star
    check_prefix_feasible(w_total, S, "DASE")
    return w_total, S_res, S_res_pb, w_star


# ============================
# Utility
# ============================

def leontief_alpha(alloc: np.ndarray, demand: np.ndarray) -> float:
    mask = demand > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(alloc[mask] / demand[mask]))


def to_excel_bytes(**dfs: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    return buf.read()


# ============================
# Streamlit UI
# ============================

st.set_page_config(page_title="SE / DA / DASE (Infinite Storage)", layout="wide")
st.title("SE / DA / DASE — Infinite Storage (PuLP + Progressive SE)")

with st.expander("Built-in example from your paper (click to load)", expanded=False):
    st.write("This is the example you pasted (n=T=5).")
    st.code(
        "Demands:\n"
        "1 2 1 0 0\n"
        "3 1 0 0 0\n"
        "0 2 1 0 1\n"
        "0 0 2 2 0\n"
        "0 0 0 3 1\n\n"
        "Supply:\n"
        "36 36 36 42 50\n"
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands & Supply")
    mode = st.radio("Input method", ["Paste", "Excel"], horizontal=True)

    if mode == "Paste":
        d_text = st.text_area("Demands (rows=agents, cols=time)", height=180,
                              value="1 2 1 0 0\n3 1 0 0 0\n0 2 1 0 1\n0 0 2 2 0\n0 0 0 3 1")
        s_text = st.text_area("Supply (single row/column)", height=120,
                              value="36 36 36 42 50")
        up = None
    else:
        up = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
        d_text = ""
        s_text = ""

with col2:
    st.subheader("Run")
    run = st.button("Compute allocations", type="primary", use_container_width=True)

if run:
    try:
        if mode == "Paste":
            demands = parse_matrix_from_text(d_text)
            supply = parse_vector_from_text(s_text)
        else:
            if up is None:
                raise ValueError("Please upload an Excel file.")
            demands, supply = read_excel(up)

        n, T = validate_inputs(demands, supply)

        # --- SE on full supply (sometimes you want to see it)
        alloc_se = sequential_equalizing_progressive(demands, supply)
        check_prefix_feasible(alloc_se, supply, "SE")

        # --- DA
        alloc_da, beta_da = run_DA(demands, supply)

        # --- DASE
        alloc_dase, S_res_raw, S_res_pb, alloc_se_on_res = dase(demands, supply, alloc_da)

        # --- alphas
        alpha_se = np.array([leontief_alpha(alloc_se[i], demands[i]) for i in range(n)], dtype=float)
        alpha_da = np.array([leontief_alpha(alloc_da[i], demands[i]) for i in range(n)], dtype=float)
        alpha_dase = np.array([leontief_alpha(alloc_dase[i], demands[i]) for i in range(n)], dtype=float)

        idx = [f"agent_{i+1}" for i in range(n)]
        cols = [f"t{t+1}" for t in range(T)]

        df_dem = pd.DataFrame(demands, index=idx, columns=cols)
        df_sup = pd.DataFrame(supply.reshape(1, -1), index=["S"], columns=cols)

        df_se = pd.DataFrame(alloc_se, index=idx, columns=cols)
        df_da = pd.DataFrame(alloc_da, index=idx, columns=cols)
        df_dase = pd.DataFrame(alloc_dase, index=idx, columns=cols)

        df_res = pd.DataFrame(
            np.vstack([S_res_raw, S_res_pb, np.sum(alloc_se_on_res, axis=0)]),
            index=["S_res_raw = S - sum(DA)", "S_res_pushback", "sum(SE_on_residual)"],
            columns=cols
        )

        summary = pd.DataFrame(
            {
                "alpha_SE": alpha_se,
                "beta_DA": beta_da,
                "alpha_DA": alpha_da,
                "alpha_DASE": alpha_dase,
            },
            index=idx,
        )

        st.success("Computed allocations (all prefix-feasible).")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "SE", "DA", "DASE", "Residuals"])
        with tab1:
            st.dataframe(summary, use_container_width=True)

        with tab2:
            st.dataframe(df_se, use_container_width=True)

        with tab3:
            st.dataframe(df_da, use_container_width=True)

        with tab4:
            st.dataframe(df_dase, use_container_width=True)

        with tab5:
            st.dataframe(df_res, use_container_width=True)

        out_bytes = to_excel_bytes(
            demands=df_dem,
            supply=df_sup,
            SE=df_se,
            DA=df_da,
            DASE=df_dase,
            Residuals=df_res,
            Summary=summary,
        )
        st.download_button(
            "Download results (Excel)",
            data=out_bytes,
            file_name="allocations_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))
