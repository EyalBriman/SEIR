# SE / SEIR Streamlit App

This repository contains a lightweight Streamlit application for computing
allocations in a continuous fair division setting over time.

The app implements and compares the following allocation rules:

- **Sequential Equalizing (SE)**
- **Decentralized Strong Individual Rationality (Strong-IR)**  
  Each agent receives an equal share (1/n) of supply and optimizes its allocation
  over time under infinite storage.
- **SEIR**  
  Strong-IR allocations followed by Sequential Equalizing on the residual supply.

All computations assume **infinite storage**, as in the current theoretical model.

---

## Input

Users provide:

- **Demands**: a matrix of shape *(n agents × T time steps)*  
  All rows must sum to the same total demand.
- **Supply**: a vector of length *T* with non-negative entries.

Inputs can be provided either:
- by pasting numeric tables directly into the app, or
- by uploading an Excel file (`.xlsx`) with sheets:
  - `demands`
  - `supply`

---

## Output

The app returns:

- Allocation tables for each rule (agent × time step)
- Per-agent Leontief utility values (α)
- A downloadable Excel file containing all results

---

## Algorithms

- **SE** follows the standard Sequential Equalizing procedure.
- **Strong-IR** is solved via a linear program using **PuLP (CBC solver)**,
  avoiding commercial solvers.
- **SEIR** combines Strong-IR allocations with SE on the remaining supply.

---

## Deployment

This app is designed to be deployed directly on **Streamlit Community Cloud**.

The entry point is:
