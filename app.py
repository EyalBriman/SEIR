# app.py
# Streamlit UI for:
#   (1) SE  (Sequential Equalizing — prefix-feasible with storage)
#   (2) DA  (Decentralized: each agent gets S(t)/n and optimizes local storage to maximize beta)
#   (3) SEIR/DASE = DA allocations + SE on residual supply after push-back
#
# Run:
#   pip install streamlit pandas numpy openpyxl pulp
#   streamlit run app.py
#
# Notes:
# - SE here is implemented in a **prefix-feasible** way (cannot allocate more than cumulative supply).
# - DA is solved per-agent with PuLP+CBC using an inventory (carry) formulation (no borrowing from future).
# - Prefix-feasibility is the correct feasibility notion under (central) storage: for each prefix t,
#     sum_{j<=t} sum_i w_i(j) <= sum_{j<=t} S(j).

from __future__ import annotations

import io
from typing import Tuple, List, Set

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
    """
    Expected formats:
      Option A: Sheet 'demands' and sheet 'supply'
      Option B: first sheet demands, second sheet supply
    """
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
# Feasibility check (prefix)
# ============================

def check_prefix_feasible(alloc: np.ndarray, supply: np.ndarray, name: str):
    used = np.cumsum(np.sum(alloc, axis=0))
    cap = np.cumsum(supply)
    for t in range(len(supply)):
        if used[t] > cap[t] + 1e-7:
            raise ValueError(
                f"{name} cumulative infeasible at t={t}: used_prefix={used[t]:.6g} > supply_prefix={cap[t]:.6g}."
            )


# ============================
# Core algorithms
# ============================

def sequential_equalizing(demands: np.ndarray, supplies: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    """
    Sequential Equalizing (SE), implemented to be prefix-feasible with storage.

    This follows the spirit of your pseudocode (iterative, fixing agents once a prefix bottleneck hits),
    BUT also explicitly updates residual supply so the final allocation cannot exceed cumulative supply.

    At each iteration:
      - Compute t* = argmin_t  (cumS_res(t) / cumD_active(t))
      - alpha = that minimum ratio
      - Allocate alpha*d_i(t) for active agents:
          * for t <= t*: set w_i(t) = alpha*d_i(t) and FIX agent i (if it has any demand in [0..t*])
          * for t >  t*: add alpha*d_i(t) (keeps accumulating)
      - Subtract what was allocated from residual supply, continue.

    Returns:
      allocations (n,T), alpha_values (one per iteration)
    """
    n, T = demands.shape
    w = np.zeros((n, T), dtype=float)

    S_res = supplies.astype(float).copy()
    active: Set[int] = set(range(n))
    alpha_values: List[float] = []

    max_iters = n + T + 10
    for _ in range(max_iters):
        if not active:
            break

        D_active = np.sum(demands[list(active), :], axis=0)
        cumD = np.cumsum(D_active)
        cumS = np.cumsum(S_res)

        ratios = []
        for t in range(T):
            if cumD[t] <= 1e-12:
                ratios.append(float("inf"))
            else:
                ratios.append(float(cumS[t] / cumD[t]))

        t_star = int(np.argmin(ratios))
        alpha = float(ratios[t_star])
        if not np.isfinite(alpha):
            break

        alpha_values.append(alpha)

        delta = np.zeros((n, T), dtype=float)

        # First, add alpha*d for t > t*
        if t_star + 1 < T:
            for i in active:
                delta[i, t_star + 1 :] = alpha * demands[i, t_star + 1 :]

        # Then, for agents that have demand in the prefix, set prefix allocations and fix them
        newly_fixed: Set[int] = set()
        for i in active:
            if np.any(demands[i, : t_star + 1] > 0):
                newly_fixed.add(i)
                delta[i, : t_star + 1] = alpha * demands[i, : t_star + 1]

        w += delta
        S_res = S_res - np.sum(delta, axis=0)

        if np.min(S_res) < -1e-7:
            # If this triggers, something is inconsistent numerically.
            raise ValueError("SE produced negative residual supply (numerical issue).")

        S_res = np.maximum(S_res, 0.0)
        active -= newly_fixed

    return w, alpha_values


def seir(demands: np.ndarray, supply: np.ndarray, da_allocations: np.ndarray) -> np.ndarray:
    """
    SEIR / DASE as in your pseudocode:

      1) w_IR := DA
      2) S'(t) = S(t) - sum_i w_IR_i(t)
      3) while some S'(t) < 0, take latest negative time t-hat and push it back to t-hat-1
      4) w* := SE(d, S')
      5) return w_IR + w*

    Important:
      - w_IR might use more than S(t) at a particular time t (because agents can store from earlier),
        but it must satisfy the PREFIX constraint. We validate prefix-feasibility outside.
    """
    w_da = da_allocations.copy().astype(float)
    S_res = supply.astype(float) - np.sum(w_da, axis=0)

    # push negatives backward (latest first)
    for t in range(len(S_res) - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    if S_res[0] < -1e-9:
        raise ValueError(
            f"SEIR residual infeasible: S_res[0]={S_res[0]:.6g}<0 after push-back. "
            "This means DA already used more than total supply in the first prefix (should not happen)."
        )

    S_res = np.maximum(S_res, 0.0)
    w_se, _ = sequential_equalizing(demands, S_res)
    return w_da + w_se


# ============================
# DA via PuLP (CBC) using inventory formulation
# ============================

def _cbc_solver_or_die():
    """
    Streamlit Cloud often does NOT have CBC unless installed.
    If CBC is missing, we fail with a clear message.
    """
    solver = pulp.PULP_CBC_CMD(msg=False)
    # crude availability check: pulp will error at solve-time if missing; we keep message clear then
    return solver


def da_optimization_pulp(single_agent_d: np.ndarray, S: np.ndarray, T: int, n: int) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    DA with infinite local storage.

    Agent i receives Si(t)=S(t)/n at each time.
    Agent chooses consumption w(t) and leftover y(t) (carried to next time) to maximize beta:

      w(0) + y(0) <= Si(0)
      for t>=1: w(t) + y(t) <= Si(t) + y(t-1)
      beta*d(t) <= w(t)   for d(t)>0
      w(t), y(t) >= 0

    beta is bounded to avoid "unbounded" issues if some d(t)=0:
      beta <= sum(Si) / min_{t:d(t)>0} d(t)
    """
    d = single_agent_d.reshape(-1).astype(float)
    if len(d) != T:
        raise ValueError("single_agent_d has wrong length.")
    S = S.astype(float)
    Si = S / float(n)

    d_pos = d[d > 0]
    if d_pos.size == 0:
        # should not happen due to validation
        return np.zeros(T), 0.0, np.zeros(T)

    beta_ub = float(np.sum(Si) / np.min(d_pos))

    prob = pulp.LpProblem("DA_single_agent", pulp.LpMaximize)
    w = pulp.LpVariable.dicts("w", list(range(T)), lowBound=0)
    y = pulp.LpVariable.dicts("y", list(range(T)), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=0.0, upBound=beta_ub)

    prob += beta
    prob += (w[0] + y[0] <= Si[0]), "inv_0"
    for t in range(1, T):
        prob += (w[t] + y[t] <= Si[t] + y[t - 1]), f"inv_{t}"

    for t in range(T):
        if d[t] > 0:
            prob += (beta * d[t] <= w[t]), f"beta_demand_{t}"

    solver = _cbc_solver_or_die()
    try:
        status = prob.solve(solver)
    except Exception as e:
        raise ValueError(
            "PuLP could not run CBC on this machine. "
            "If you deploy on Streamlit Cloud, you typically need to install CBC via packages.txt "
            "(e.g., 'coinor-cbc')."
        ) from e

    if pulp.LpStatus[status] != "Optimal":
        return np.zeros(T), 0.0, np.zeros(T)

    w_arr = np.array([pulp.value(w[t]) for t in range(T)], dtype=float)
    y_arr = np.array([pulp.value(y[t]) for t in range(T)], dtype=float)
    beta_val = float(pulp.value(beta))
    return w_arr, beta_val, y_arr


def leontief_alpha(alloc: np.ndarray, demand: np.ndarray) -> float:
    mask = demand > 0
    if not np.any(mask):
        return float("inf")
    return float(np.min(alloc[mask] / demand[mask]))


# ============================
# Export helper
# ============================

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

st.set_page_config(page_title="SE / DA / SEIR (DASE)", layout="wide")
st.title("SE / DA / SEIR (DASE) — Infinite Storage (PuLP+CBC)")

with st.sidebar:
    st.subheader("Demo")
    if st.button("Load demo (your earlier failing cases)", use_container_width=True):
        # Demo 1: the 3x3 small one you showed
        st.session_state["demo_demands"] = "1 2 3\n3 2 1\n2 3 1"
        st.session_state["demo_supply"] = "1 2 1"
    if st.button("Load demo (20 20 20 case)", use_container_width=True):
        st.session_state["demo_demands"] = "7 2 1\n3 0 7\n0 8 2\n4 4 2"
        st.session_state["demo_supply"] = "20 20 20"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands & Supply")
    mode = st.radio("Input method", ["Paste", "Excel"], horizontal=True)

    if mode == "Paste":
        d_default = st.session_state.get("demo_demands", "")
        s_default = st.session_state.get("demo_supply", "")
        st.caption("Demands: rows=agents, cols=time. Whitespace or commas.")
        d_text = st.text_area("Demands", height=180, value=d_default)
        st.caption("Supply: single row/column of length T.")
        s_text = st.text_area("Supply", height=120, value=s_default)
        up = None
    else:
        st.caption("Upload .xlsx with sheets 'demands' and 'supply' (or first=demands, second=supply).")
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

        total_d = float(np.sum(demands))
        total_s = float(np.sum(supply))
        if total_s + 1e-9 < total_d:
            st.warning(
                f"Total supply ({total_s:.6g}) is smaller than total demand ({total_d:.6g}). "
                f"Some alphas will necessarily be < 1."
            )

        # --- SE
        alloc_se, se_alpha_list = sequential_equalizing(demands, supply)
        check_prefix_feasible(alloc_se, supply, "SE")

        # --- DA (per-agent LP)
        alloc_da = np.zeros_like(demands, dtype=float)
        beta_da = np.zeros(n, dtype=float)
        for i in range(n):
            alloc_da[i, :], beta_da[i], _ = da_optimization_pulp(demands[i:i + 1], supply, T, n)
        check_prefix_feasible(alloc_da, supply, "DA")

        # --- SEIR/DASE
        alloc_seir = seir(demands, supply, alloc_da)
        check_prefix_feasible(alloc_seir, supply, "SEIR/DASE")

        # --- Alphas
        alpha_se = np.array([leontief_alpha(alloc_se[i], demands[i]) for i in range(n)], dtype=float)
        alpha_da = np.array([leontief_alpha(alloc_da[i], demands[i]) for i in range(n)], dtype=float)
        alpha_seir = np.array([leontief_alpha(alloc_seir[i], demands[i]) for i in range(n)], dtype=float)

        st.success("Computed allocations (prefix-feasible).")

        idx = [f"agent_{i}" for i in range(n)]
        cols = [f"t{t}" for t in range(T)]
        df_dem = pd.DataFrame(demands, index=idx, columns=cols)
        df_sup = pd.DataFrame(supply.reshape(1, -1), index=["supply"], columns=cols)
        df_se = pd.DataFrame(alloc_se, index=idx, columns=cols)
        df_da = pd.DataFrame(alloc_da, index=idx, columns=cols)
        df_seir = pd.DataFrame(alloc_seir, index=idx, columns=cols)

        # prefix diagnostics (helps you see feasibility quickly)
        prefix_used_se = np.cumsum(np.sum(alloc_se, axis=0))
        prefix_used_da = np.cumsum(np.sum(alloc_da, axis=0))
        prefix_used_seir = np.cumsum(np.sum(alloc_seir, axis=0))
        prefix_cap = np.cumsum(supply.astype(float))
        df_prefix = pd.DataFrame(
            {
                "prefix_supply": prefix_cap,
                "prefix_used_SE": prefix_used_se,
                "prefix_used_DA": prefix_used_da,
                "prefix_used_SEIR": prefix_used_seir,
            },
            index=cols,
        )

        summary = pd.DataFrame(
            {
                "alpha_SE": alpha_se,
                "alpha_DA": alpha_da,
                "beta_DA_LP": beta_da,
                "alpha_SEIR_DASE": alpha_seir,
                "SE_alpha_iters": [len(se_alpha_list)] * n,
            },
            index=idx,
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "SE", "DA", "SEIR/DASE", "Prefix check"])
        with tab1:
            st.dataframe(summary, use_container_width=True)
        with tab2:
            st.dataframe(df_se, use_container_width=True)
        with tab3:
            st.dataframe(df_da, use_container_width=True)
        with tab4:
            st.dataframe(df_seir, use_container_width=True)
        with tab5:
            st.dataframe(df_prefix, use_container_width=True)

        out_bytes = to_excel_bytes(
            demands=df_dem,
            supply=df_sup,
            SE=df_se,
            DA=df_da,
            SEIR_DASE=df_seir,
            PrefixCheck=df_prefix,
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
