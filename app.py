# app.py
# Streamlit UI for:
#   (1) SE (Sequential Equalizing)
#   (2) DA (Decentralized): each agent gets S(t)/n and optimizes beta_i with forward storage transfers
#   (3) SEIR = DA allocations + SE on residual supply
#
# Run locally:
#   pip install -r requirements.txt
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
    if text.strip() == "":
        raise ValueError("Empty input.")
    # Try CSV
    try:
        df = pd.read_csv(io.StringIO(text), header=None)
        arr = df.values.astype(float)
        return arr
    except Exception:
        # Fallback whitespace
        rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
        data = []
        for r in rows:
            parts = r.replace(",", " ").split()
            data.append([float(x) for x in parts])
        return np.array(data, dtype=float)

def parse_vector_from_text(text: str) -> np.ndarray:
    arr = parse_matrix_from_text(text)
    if arr.ndim != 2:
        raise ValueError("Supply input must be a 1D vector or a single-row/column table.")
    if arr.shape[0] == 1:
        return arr.reshape(-1).astype(float)
    if arr.shape[1] == 1:
        return arr.reshape(-1).astype(float)
    raise ValueError("Supply must be a single row or a single column.")

def read_excel(file) -> tuple[np.ndarray, np.ndarray]:
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
        .dropna(how="all").dropna(axis=1, how="all").values.astype(float)
    )
    supply_mat = (
        supply_df.apply(pd.to_numeric, errors="coerce")
        .dropna(how="all").dropna(axis=1, how="all").values.astype(float)
    )

    if supply_mat.shape[0] == 1:
        supply = supply_mat.reshape(-1)
    elif supply_mat.shape[1] == 1:
        supply = supply_mat.reshape(-1)
    else:
        raise ValueError("In Excel, supply must be a single row or a single column.")

    return demands, supply

def validate_inputs(demands: np.ndarray, supply: np.ndarray) -> tuple[int, int]:
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
# Core algorithms
# ============================

def sequential_equalizing(demands: np.ndarray, supplies: np.ndarray):
    """
    SE (as in your pseudocode).
    demands: (n,T), supplies: (T,)
    Returns allocations (n,T) and alpha_values list.
    """
    n, T = demands.shape
    allocations = np.zeros((n, T))
    N_star = set()
    N = set(range(n))
    alpha_values = []

    while N_star != N:
        Q = [
            np.sum(demands[:, :t+1]) / np.sum(supplies[:t+1])
            if np.sum(supplies[:t+1]) > 0 else float('inf')
            for t in range(T)
        ]
        t_star = int(np.argmin(Q))
        alpha = float(Q[t_star])
        alpha_values.append(alpha)

        for i in (N - N_star):
            for t in range(t_star + 1, T):
                allocations[i, t] += alpha * demands[i, t]

            if np.any(demands[i, :t_star+1] > 0):
                for t in range(t_star + 1):
                    allocations[i, t] = alpha * demands[i, t]
                N_star.add(i)

        # safety
        if len(alpha_values) > n + T + 5:
            break

    return allocations, alpha_values


def seir(demands: np.ndarray, supply: np.ndarray, da_allocations: np.ndarray):
    """
    SEIR:
      1) start with DA allocations (paper calls this "minimal IR allocation" via DA),
      2) compute residual supply and push any negative residual backward,
      3) run SE on residual, and add.
    """
    n, T = demands.shape
    w_da = da_allocations.copy()

    # residual
    S_res = supply - np.sum(w_da, axis=0)

    # push negative residual backward
    for t in range(T - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    S_res = np.maximum(S_res, 0.0)

    # SE on residual
    w_se, _ = sequential_equalizing(demands, S_res)
    return w_da + w_se


# ============================
# DA (Decentralized) via PuLP
# ============================

def _pick_solver():
    """
    Prefer HiGHS if available (great for Streamlit Cloud via highspy),
    otherwise fall back to CBC if present.
    """
    # HiGHS (recommended)
    if hasattr(pulp, "HiGHS_CMD"):
        return pulp.HiGHS_CMD(msg=False)
    if hasattr(pulp, "HiGHS"):
        return pulp.HiGHS(msg=False)

    # CBC fallback
    return pulp.PULP_CBC_CMD(msg=False)


def da_optimization_pulp(single_agent_d: np.ndarray, S: np.ndarray, T: int, n: int):
    """
    DA per Definition 8 in the PDF:
      - agent gets fixed share S_i(t)=S(t)/n
      - chooses w_i(t) and forward transfers z_{t,t'} (t < t') to maximize beta_i
      - constraints:
          beta_i * d_i(t) <= w_i(t)
          S'_i(t) = S_i(t) - sum_{t'>t} z_{t,t'} + sum_{t'<t} z_{t',t}
          w_i(t) <= S'_i(t)
    (The 1_{t<T} term is automatic because when t is last step there is no t'>t.)
    """
    d = single_agent_d.reshape(-1).astype(float)
    if len(d) != T:
        raise ValueError("single_agent_d has wrong length.")
    S = S.astype(float)

    prob = pulp.LpProblem("DA_single_agent", pulp.LpMaximize)

    # Variables
    w = pulp.LpVariable.dicts("w", list(range(T)), lowBound=0)
    Sprime = pulp.LpVariable.dicts("Sprime", list(range(T)), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=1e-9, upBound=1.0)

    # Only allow forward transfers z_{t,t'} for t < t'
    z = {}
    for t in range(T):
        for tp in range(t + 1, T):
            z[(t, tp)] = pulp.LpVariable(f"z_{t}_{tp}", lowBound=0)

    # Objective
    prob += beta

    # Sprime balance (S_i(t)=S(t)/n)
    for t in range(T):
        incoming = pulp.lpSum(z[(tp, t)] for tp in range(0, t) if (tp, t) in z)
        outgoing = pulp.lpSum(z[(t, tp)] for tp in range(t + 1, T) if (t, tp) in z)
        prob += (Sprime[t] == (S[t] / n) - outgoing + incoming), f"Sprime_balance_{t}"

    # Feasibility and tight allocation constraints
    for t in range(T):
        prob += (w[t] <= Sprime[t]), f"w_le_Sprime_{t}"
        prob += (beta * d[t] <= w[t]), f"beta_demand_{t}"

    # Solve
    solver = _pick_solver()
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        return np.zeros(T), 0.0, np.zeros(T)

    w_arr = np.array([pulp.value(w[t]) for t in range(T)], dtype=float)
    Sprime_arr = np.array([pulp.value(Sprime[t]) for t in range(T)], dtype=float)
    beta_val = float(pulp.value(beta))
    return w_arr, beta_val, Sprime_arr


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

st.set_page_config(page_title="SE / SEIR / DA Allocator", layout="wide")
st.title("SE / SEIR / DA — Streamlit (License-free LP, Infinite Storage)")

with st.expander("Input format (quick)", expanded=False):
    st.write(
        """
- **Demands**: matrix (agents × time steps). Rows are agents; columns are time steps.
- **Supply**: vector of length T (single row or single column).
- Validation:
  - all numbers must be **>= 0**
  - each demand row must sum to the **same total** (otherwise error)
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands & Supply")
    mode = st.radio("Input method", ["Paste", "Excel"], horizontal=True)

    demands = None
    supply = None

    if mode == "Paste":
        st.caption("Paste demands: rows=agents, cols=time. Commas or spaces work.")
        d_text = st.text_area("Demands", height=180, placeholder="e.g.\n1 2 3\n1 2 3\n1 2 3")
        st.caption("Paste supply: a single row or column of length T.")
        s_text = st.text_area("Supply", height=120, placeholder="e.g.\n10 10 10")
    else:
        st.caption("Upload .xlsx with sheets 'demands' and 'supply' (or first=demands, second=supply).")
        up = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])

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

        # SE
        alloc_se, _ = sequential_equalizing(demands, supply)

        # DA for each agent
        alloc_da = np.zeros_like(demands, dtype=float)
        beta_da = np.zeros(n, dtype=float)
        for i in range(n):
            alloc_da[i, :], beta_da[i], _ = da_optimization_pulp(demands[i:i+1], supply, T, n)

        # SEIR
        alloc_seir = seir(demands, supply, alloc_da)

        # Per-agent Leontief alphas
        alpha_se = np.array([leontief_alpha(alloc_se[i], demands[i]) for i in range(n)], dtype=float)
        alpha_da = np.array([leontief_alpha(alloc_da[i], demands[i]) for i in range(n)], dtype=float)
        alpha_seir = np.array([leontief_alpha(alloc_seir[i], demands[i]) for i in range(n)], dtype=float)

        st.success("Computed allocations.")

        idx = [f"agent_{i}" for i in range(n)]
        cols = [f"t{t}" for t in range(T)]
        df_dem = pd.DataFrame(demands, index=idx, columns=cols)
        df_sup = pd.DataFrame(supply.reshape(1, -1), index=["supply"], columns=cols)
        df_se = pd.DataFrame(alloc_se, index=idx, columns=cols)
        df_da = pd.DataFrame(alloc_da, index=idx, columns=cols)
        df_seir = pd.DataFrame(alloc_seir, index=idx, columns=cols)

        summary = pd.DataFrame(
            {
                "alpha_SE": alpha_se,
                "beta_DA": beta_da,          # the LP objective value
                "alpha_DA": alpha_da,        # computed from final DA allocation
                "alpha_SEIR": alpha_seir,
            },
            index=idx,
        )

        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "SE", "DA (Decentralized)", "SEIR"])

        with tab1:
            st.write("**Sizes**", {"n": n, "T": T})
            st.write("**Demand row sum (must be identical):**", float(df_dem.sum(axis=1).iloc[0]))
            st.dataframe(summary, use_container_width=True)

        with tab2:
            st.dataframe(df_se, use_container_width=True)

        with tab3:
            st.dataframe(df_da, use_container_width=True)

        with tab4:
            st.dataframe(df_seir, use_container_width=True)

        out_bytes = to_excel_bytes(
            demands=df_dem,
            supply=df_sup,
            SE=df_se,
            DA=df_da,
            SEIR=df_seir,
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
