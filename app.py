# app.py
# Streamlit UI for:
#   (1) SE (Sequential Equalizing)  [FIXED: alpha = cumulative_supply / cumulative_demand]
#   (2) DA (Decentralized): each agent gets S(t)/n and can carry unused supply forward (infinite storage)
#   (3) SEIR = DA allocations + SE on residual supply (with backward push of negative residual)
#
# Run locally:
#   pip install streamlit pandas numpy openpyxl pulp
#   streamlit run app.py
#
# IMPORTANT:
# - With storage, it is NORMAL that sum_i w_i(t) can exceed S(t) at some t
#   (it uses stored supply from earlier times). So we check feasibility cumulatively:
#       for all t: sum_{j<=t} sum_i w_i(j) <= sum_{j<=t} S(j)
#
# This version:
# - avoids HiGHS (prevents "PuLP: cannot execute highs") and uses CBC
# - validates: nonnegative, numeric, demand row sums equal, supply length matches
# - checks cumulative feasibility for SE, DA, SEIR
# - exports all tables to Excel

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
    # Try CSV first
    try:
        df = pd.read_csv(io.StringIO(text), header=None)
        return df.values.astype(float)
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

    if not np.isfinite(demands).all():
        raise ValueError("Demands contain non-numeric or infinite values.")
    if not np.isfinite(supply).all():
        raise ValueError("Supply contains non-numeric or infinite values.")

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

    # Guard: some prefix has positive demand but zero cumulative supply
    cumD = np.cumsum(demands.sum(axis=0))
    cumS = np.cumsum(supply)
    if np.any((cumS == 0) & (cumD > 0)):
        raise ValueError("Some prefix has zero cumulative supply but positive cumulative demand.")

    return n, T


# ============================
# Feasibility checks (WITH STORAGE)
# ============================

def check_cumulative_feasible(alloc: np.ndarray, supply: np.ndarray, eps: float = 1e-6) -> None:
    """
    With storage, per-time feasibility is not required.
    The correct check is prefix feasibility:
      for all t: sum_{j<=t} sum_i alloc[i,j] <= sum_{j<=t} supply[j] + eps
    """
    used_prefix = np.cumsum(np.sum(alloc, axis=0))
    supply_prefix = np.cumsum(supply)
    bad = used_prefix > supply_prefix + eps
    if np.any(bad):
        t_bad = int(np.argmax(bad))
        raise RuntimeError(
            f"Cumulative infeasible at t={t_bad}: "
            f"used_prefix={used_prefix[t_bad]:.6g} > supply_prefix={supply_prefix[t_bad]:.6g}."
        )


# ============================
# Core algorithms
# ============================

def sequential_equalizing(demands: np.ndarray, supplies: np.ndarray):
    """
    Sequential Equalizing (SE) — FIXED.

    Satisfaction alpha is allocation/demand.
    Therefore:
      alpha(t) = cumulative_supply(1..t) / cumulative_demand(1..t)
      t* = argmin_t alpha(t)
      alpha = alpha(t*)
    """
    n, T = demands.shape
    allocations = np.zeros((n, T))
    N_star: set[int] = set()
    N: set[int] = set(range(n))
    alpha_values: list[float] = []

    cumD = np.cumsum(np.sum(demands, axis=0))
    cumS = np.cumsum(supplies)

    while N_star != N:
        Q = [
            (cumS[t] / cumD[t]) if cumD[t] > 0 else float("inf")
            for t in range(T)
        ]
        t_star = int(np.argmin(Q))
        alpha = float(Q[t_star])
        alpha_values.append(alpha)

        for i in (N - N_star):
            # allocate for t > t*
            for t in range(t_star + 1, T):
                allocations[i, t] += alpha * demands[i, t]

            # fix agents with positive demand in prefix
            if np.any(demands[i, :t_star + 1] > 0):
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
      1) start with DA allocations,
      2) residual supply S_res(t) = S(t) - sum_i w_DA_i(t),
      3) push negative residual backward,
      4) run SE on residual and add.
    """
    w_da = da_allocations.copy()
    S_res = supply - np.sum(w_da, axis=0)

    for t in range(len(S_res) - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    S_res = np.maximum(S_res, 0.0)

    w_se, _ = sequential_equalizing(demands, S_res)
    return w_da + w_se


# ============================
# DA via PuLP (CBC) using stock/inventory formulation
# ============================

def da_optimization_pulp(single_agent_d: np.ndarray, S: np.ndarray, T: int, n: int):
    """
    DA with infinite storage (forward carry):
      - per-time share Si(t) = S(t)/n
      - variables: w(t) allocation, y(t) stock carried to next time, beta
      - constraints:
          w(0) + y(0) <= Si(0)
          for t>=1: w(t) + y(t) <= Si(t) + y(t-1)
          beta*d(t) <= w(t) for d(t)>0
      - objective: maximize beta
    """
    d = single_agent_d.reshape(-1).astype(float)
    if len(d) != T:
        raise ValueError("single_agent_d has wrong length.")
    S = S.astype(float)
    Si = S / n

    prob = pulp.LpProblem("DA_single_agent", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", list(range(T)), lowBound=0)
    y = pulp.LpVariable.dicts("y", list(range(T)), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=1e-9, upBound=1.0)

    prob += beta

    # inventory constraints
    prob += (w[0] + y[0] <= Si[0]), "inv_0"
    for t in range(1, T):
        prob += (w[t] + y[t] <= Si[t] + y[t-1]), f"inv_{t}"

    # satisfaction constraints
    for t in range(T):
        if d[t] > 0:
            prob += (beta * d[t] <= w[t]), f"beta_demand_{t}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

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

st.set_page_config(page_title="SE / SEIR / DA Allocator", layout="wide")
st.title("SE / SEIR / DA (Infinite Storage)")

with st.expander("Input format (quick)", expanded=False):
    st.write(
        """
- **Demands**: matrix (agents × time steps). Rows are agents; columns are time steps.
- **Supply**: vector of length T (single row or single column).
- Validation:
  - all numbers must be **>= 0**
  - each demand row must sum to the **same total** (otherwise error)
- With storage, feasibility is checked **cumulatively** (prefix sums), not per-time.
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands & Supply")
    mode = st.radio("Input method", ["Paste", "Excel"], horizontal=True)

    if mode == "Paste":
        st.caption("Paste demands: rows=agents, cols=time. Commas or spaces work.")
        d_text = st.text_area("Demands", height=180, placeholder="e.g.\n1 2 3\n1 2 3\n1 2 3")
        st.caption("Paste supply: a single row or column of length T.")
        s_text = st.text_area("Supply", height=120, placeholder="e.g.\n10 10 10")
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

        # SE
        alloc_se, _ = sequential_equalizing(demands, supply)
        check_cumulative_feasible(alloc_se, supply)

        # DA for each agent
        alloc_da = np.zeros_like(demands, dtype=float)
        beta_da = np.zeros(n, dtype=float)
        for i in range(n):
            alloc_da[i, :], beta_da[i], _ = da_optimization_pulp(demands[i:i+1], supply, T, n)
        check_cumulative_feasible(alloc_da, supply)

        # SEIR
        alloc_seir = seir(demands, supply, alloc_da)
        check_cumulative_feasible(alloc_seir, supply)

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

        # Helpful diagnostics (not errors) about per-time exceedance due to storage
        per_time_used_da = np.sum(alloc_da, axis=0)
        per_time_used_se = np.sum(alloc_se, axis=0)
        per_time_used_seir = np.sum(alloc_seir, axis=0)

        storage_note = False
        if np.any(per_time_used_da > supply + 1e-6) or np.any(per_time_used_se > supply + 1e-6) or np.any(per_time_used_seir > supply + 1e-6):
            storage_note = True

        summary = pd.DataFrame(
            {
                "alpha_SE": alpha_se,
                "beta_DA": beta_da,          # DA LP objective
                "alpha_DA": alpha_da,        # computed from DA allocation
                "alpha_SEIR": alpha_seir,
            },
            index=idx,
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "SE", "DA (Decentralized)", "SEIR", "Diagnostics"])

        with tab1:
            st.write("**Sizes**", {"n": n, "T": T})
            st.write("**Demand row sum (must be identical):**", float(df_dem.sum(axis=1).iloc[0]))
            st.dataframe(summary, use_container_width=True)
            if storage_note:
                st.info(
                    "Note: Some allocations exceed per-time supply S(t). This is allowed with storage "
                    "(it uses stored supply from earlier time steps). Feasibility is checked cumulatively."
                )

        with tab2:
            st.dataframe(df_se, use_container_width=True)

        with tab3:
            st.dataframe(df_da, use_container_width=True)

        with tab4:
            st.dataframe(df_seir, use_container_width=True)

        with tab5:
            diag = pd.DataFrame(
                {
                    "S(t)": supply,
                    "sum_i SE(t)": per_time_used_se,
                    "sum_i DA(t)": per_time_used_da,
                    "sum_i SEIR(t)": per_time_used_seir,
                    "prefix S(<=t)": np.cumsum(supply),
                    "prefix SE used": np.cumsum(per_time_used_se),
                    "prefix DA used": np.cumsum(per_time_used_da),
                    "prefix SEIR used": np.cumsum(per_time_used_seir),
                },
                index=cols,
            )
            st.dataframe(diag, use_container_width=True)

        out_bytes = to_excel_bytes(
            demands=df_dem,
            supply=df_sup,
            SE=df_se,
            DA=df_da,
            SEIR=df_seir,
            Summary=summary,
            Diagnostics=diag,
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
