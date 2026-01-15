# app.py
# Streamlit UI for:
#   (1) SE  (Sequential Equalizing surplus allocation with storage feasibility via prefixes)
#   (2) DA  (Decentralized Allocation: each agent gets S(t)/n and optimizes beta with local storage)
#   (3) SEIR/DASE = DA allocations + SE on residual supply after push-back
#
# Run:
#   pip install streamlit pandas numpy openpyxl pulp
#   streamlit run app.py
#
# Notes:
# - Uses PuLP + CBC (license-free).
# - Storage feasibility is checked cumulatively (prefix constraints), consistent with storage/carry.

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
    """
    Accepts:
      - whitespace-separated rows
      - comma-separated rows
    We avoid pandas' default CSV parsing because it misreads whitespace-only input.
    """
    if text.strip() == "":
        raise ValueError("Empty input.")

    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    data = []
    for r in rows:
        # allow commas OR whitespace
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
# Feasibility check (prefix)
# ============================

def check_prefix_feasible(alloc: np.ndarray, supply: np.ndarray, name: str):
    """
    Under storage, feasibility is:
      for all t: sum_{j<=t} sum_i alloc(i,j) <= sum_{j<=t} supply(j)
    """
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

def sequential_equalizing(demands: np.ndarray, supplies: np.ndarray):
    """
    Sequential Equalizing (SE) EXACTLY in the spirit of your pseudocode,
    but FIXED so it terminates and stays prefix-feasible even when some agents
    have zero demand in early prefixes.

    Key fix:
      When computing t* and alpha, use ONLY agents in (N \\ N_star).
      Otherwise agents with zero prefix-demand never become fixable and the
      loop keeps adding alpha*d forever.

    t* = argmin_t  hatS(t) / sum_{i in U} hatd_i(t)
    alpha = that minimum ratio.
    """
    n, T = demands.shape
    allocations = np.zeros((n, T), dtype=float)
    N_star: set[int] = set()
    N: set[int] = set(range(n))
    alpha_values: list[float] = []

    cumS = np.cumsum(supplies.astype(float))

    # Safety guard against accidental non-progress
    max_iters = n + T + 10

    for _ in range(max_iters):
        if N_star == N:
            break

        U = sorted(list(N - N_star))  # unfixed agents
        # cumulative demand of unfixed agents
        cumD_U = np.cumsum(np.sum(demands[U, :], axis=0))

        Q = []
        for t in range(T):
            if cumD_U[t] <= 0:
                Q.append(float("inf"))
            else:
                Q.append(float(cumS[t] / cumD_U[t]))

        t_star = int(np.argmin(Q))
        alpha = float(Q[t_star])
        alpha_values.append(alpha)

        # Allocate for all unfixed agents
        for i in U:
            # For t > t*: add
            for t in range(t_star + 1, T):
                allocations[i, t] += alpha * demands[i, t]

            # If agent has any positive demand in prefix, fix them and set prefix allocations
            if np.any(demands[i, :t_star + 1] > 0):
                for t in range(t_star + 1):
                    allocations[i, t] = alpha * demands[i, t]
                N_star.add(i)

    if N_star != N:
        # If this happens, something degenerate occurred (e.g., supplies all zero after some point).
        # We fail loudly instead of returning nonsense.
        raise ValueError("SE did not converge (degenerate input: likely zero residual supply with remaining demand).")

    return allocations, alpha_values


def seir(demands: np.ndarray, supply: np.ndarray, da_allocations: np.ndarray):
    """
    SEIR / DASE exactly as you wrote:

      1) w_IR := DA
      2) S'(t) = S(t) - sum_i w_IR_i(t)
      3) while some S'(t) < 0: take latest t with negative and push it backward
      4) w* := SE(d, S')
      5) w := w_IR + w*
    """
    w_da = da_allocations.copy().astype(float)
    S_res = supply.astype(float) - np.sum(w_da, axis=0)

    # push negative residual backward
    for t in range(len(S_res) - 1, 0, -1):
        if S_res[t] < 0:
            S_res[t - 1] += S_res[t]
            S_res[t] = 0.0

    # If the very first entry is negative, even storage cannot fix it.
    if S_res[0] < -1e-9:
        raise ValueError(
            f"SEIR residual infeasible: after push-back, S_res[0]={S_res[0]:.6g}<0. "
            f"This means DA consumed more than total prefix supply at t=0 (should not happen)."
        )

    S_res = np.maximum(S_res, 0.0)

    w_se, _ = sequential_equalizing(demands, S_res)
    return w_da + w_se


# ============================
# DA via PuLP (CBC) using inventory formulation
# ============================

def da_optimization_pulp(single_agent_d: np.ndarray, S: np.ndarray, T: int, n: int):
    """
    DA with infinite local storage.

    Each agent i receives Si(t)=S(t)/n at each time t.
    Agent chooses allocations w(t) and carry y(t) (inventory) to maximize beta:

      maximize beta
      s.t.
        w(0) + y(0) <= Si(0)
        for t>=1: w(t) + y(t) <= Si(t) + y(t-1)
        beta * d(t) <= w(t)   (for d(t)>0)
        w(t), y(t) >= 0

    This is equivalent to allowing forward transfers and is numerically cleaner.
    """
    d = single_agent_d.reshape(-1).astype(float)
    if len(d) != T:
        raise ValueError("single_agent_d has wrong length.")

    S = S.astype(float)
    Si = S / float(n)

    prob = pulp.LpProblem("DA_single_agent", pulp.LpMaximize)

    w = pulp.LpVariable.dicts("w", list(range(T)), lowBound=0)
    y = pulp.LpVariable.dicts("y", list(range(T)), lowBound=0)
    beta = pulp.LpVariable("beta", lowBound=0.0, upBound=None)  # do not cap at 1

    prob += beta

    # inventory constraints
    prob += (w[0] + y[0] <= Si[0]), "inv_0"
    for t in range(1, T):
        prob += (w[t] + y[t] <= Si[t] + y[t - 1]), f"inv_{t}"

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
st.title("SE / SEIR (DASE) / DA — Infinite Storage (license-free)")

with st.expander("Input format (quick)", expanded=False):
    st.write(
        """
- **Demands**: matrix (agents × time steps). Rows=agents, cols=time.
- **Supply**: vector of length T (single row or single column).
- Checks:
  - all numbers **>= 0**
  - all demand rows have the **same total sum**
  - cumulative feasibility (prefix) is used for storage
        """
    )

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demands & Supply")
    mode = st.radio("Input method", ["Paste", "Excel"], horizontal=True)

    if mode == "Paste":
        st.caption("Paste demands: whitespace or commas. Example row: `7 2 1`")
        d_text = st.text_area("Demands", height=180)
        st.caption("Paste supply as one row/column, length T. Example: `20 20 20`")
        s_text = st.text_area("Supply", height=120)
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
        # Parse
        if mode == "Paste":
            demands = parse_matrix_from_text(d_text)
            supply = parse_vector_from_text(s_text)
        else:
            if up is None:
                raise ValueError("Please upload an Excel file.")
            demands, supply = read_excel(up)

        n, T = validate_inputs(demands, supply)

        # SE
        alloc_se, _ = sequential_equalizing(demands, supply)
        check_prefix_feasible(alloc_se, supply, "SE")

        # DA
        alloc_da = np.zeros_like(demands, dtype=float)
        beta_da = np.zeros(n, dtype=float)
        for i in range(n):
            alloc_da[i, :], beta_da[i], _ = da_optimization_pulp(demands[i:i+1], supply, T, n)
        check_prefix_feasible(alloc_da, supply, "DA")

        # SEIR/DASE
        alloc_seir = seir(demands, supply, alloc_da)
        check_prefix_feasible(alloc_seir, supply, "SEIR/DASE")

        # Alphas
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
                "beta_DA": beta_da,
                "alpha_DA": alpha_da,
                "alpha_SEIR_DASE": alpha_seir,
            },
            index=idx,
        )

        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "SE", "DA", "SEIR/DASE"])

        with tab1:
            st.write("**Sizes**", {"n": n, "T": T})
            st.write("**Demand row sum:**", float(df_dem.sum(axis=1).iloc[0]))
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
            SEIR_DASE=df_seir,
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
