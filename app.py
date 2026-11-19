import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

# ============================================================
# APFC ENGINE - calculation logic reused from original notebook
# ============================================================

PF_LOW_LIMIT = 0.85
TARGET_PF = 0.95
CAPACITOR_STAGES_KVAR = [0.5, 0.5, 1.0, 2.0, 2.0]

def electrical_state(system, voltage_v, active_power_kw, pf, frequency_hz=50.0):
    pf = float(np.clip(pf, 0.01, 1.0))
    p_kw = max(float(active_power_kw), 0.0)
    q_kvar = p_kw * np.tan(np.arccos(pf))
    s_kva = np.sqrt(p_kw**2 + q_kvar**2)

    if system == "Single-phase":
        current_a = (s_kva * 1000.0) / max(voltage_v, 1e-9)
    elif system == "Three-phase":
        current_a = (s_kva * 1000.0) / (
            np.sqrt(3.0) * max(voltage_v, 1e-9)
        )
    else:
        raise ValueError("Invalid electrical system")

    return {
        "P_kW": p_kw,
        "Q_kVAR": q_kvar,
        "S_kVA": s_kva,
        "PF": pf,
        "Current_A": current_a,
        "Frequency_Hz": frequency_hz
    }

def choose_capacitor_stages(required_kvar, stages_kvar):
    stages = [float(x) for x in stages_kvar if float(x) > 0]
    best_total, best_combo = 0.0, []
    for mask in range(1 << len(stages)):
        combo = [stages[i] for i in range(len(stages))
                 if mask & (1 << i)]
        total = sum(combo)
        if total <= required_kvar + 1e-9 and total > best_total:
            best_total, best_combo = total, combo
    return best_total, best_combo

def apfc_correct(system, voltage_v, active_power_kw, initial_pf,
                 target_pf=TARGET_PF, stages_kvar=None):
    if not 0 < target_pf <= 1:
        raise ValueError("Target PF must be above 0 and at most 1")

    if stages_kvar is None:
        stages_kvar = CAPACITOR_STAGES_KVAR

    before = electrical_state(
        system, voltage_v, active_power_kw, initial_pf
    )

    target_q = active_power_kw * np.tan(np.arccos(target_pf))
    required_qc = max(before["Q_kVAR"] - target_q, 0.0)

    applied_qc, selected = choose_capacitor_stages(
        required_qc, stages_kvar
    )

    remaining_q = max(before["Q_kVAR"] - applied_qc, 0.0)

    after_pf = (
        active_power_kw /
        np.sqrt(active_power_kw**2 + remaining_q**2)
        if active_power_kw > 0 else 1.0
    )

    after_s = np.sqrt(active_power_kw**2 + remaining_q**2)

    if system == "Single-phase":
        after_current = after_s * 1000 / max(voltage_v, 1e-9)
    else:
        after_current = after_s * 1000 / (
            np.sqrt(3) * max(voltage_v, 1e-9)
        )

    return {
        "System": system,
        "Voltage_V": voltage_v,
        "P_kW": active_power_kw,
        "Initial_PF": initial_pf,
        "Initial_Q_kVAR": before["Q_kVAR"],
        "Required_Qc_kVAR": required_qc,
        "Applied_Qc_kVAR": applied_qc,
        "Selected_Stages_kVAR": selected,
        "Remaining_Q_kVAR": remaining_q,
        "Corrected_PF": after_pf,
        "Initial_Current_A": before["Current_A"],
        "Corrected_Current_A": after_current,
        "Status": "Low PF" if initial_pf < PF_LOW_LIMIT
                  else "Acceptable PF"
    }

def generate_alarms(row):
    alarms = []
    if row["Initial_PF"] < PF_LOW_LIMIT:
        alarms.append("LOW POWER FACTOR")
    if row["Corrected_PF"] < TARGET_PF - 0.02:
        alarms.append("TARGET PF NOT REACHED")
    if row["Corrected_PF"] < row["Initial_PF"]:
        alarms.append("CHECK COMPENSATION LOGIC")
    if row["Applied_Qc_kVAR"] > row["Initial_Q_kVAR"] + 1e-9:
        alarms.append("OVER-COMPENSATION WARNING")
    return alarms if alarms else ["NORMAL"]

# ============================================================
# PAGE CONFIGURATION AND LIGHT THEME
# ============================================================

st.set_page_config(
    page_title="APFC Management System",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.main {background-color: #f7f9fc;}
.block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e2e8f0;
    padding: 16px;
    border-radius: 12px;
}
h1, h2, h3 {color: #17365d;}
</style>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.title("⚡ APFC SYSTEM")
    st.caption("Automatic Power Factor Correction")
    st.divider()

    page = st.radio(
        "MAIN MENU",
        [
            "Overview",
            "Live Simulator",
            "Capacitor Bank",
            "Analytics",
            "Alarms & Events",
            "Reports",
            "About"
        ]
    )

    st.divider()
    st.success("Software simulation active")
    st.caption("Educational model | 50 Hz")

st.title("⚡ APFC Management System")
st.caption("Power monitoring • Capacitor control • Power factor analysis")

# ============================================================
# HELPER: DISPLAY RESULT
# ============================================================

def show_result(result):
    st.subheader("Electrical Performance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial Power Factor",
              f'{result["Initial_PF"]:.3f}')
    c2.metric("Corrected Power Factor",
              f'{result["Corrected_PF"]:.3f}',
              delta=f'{result["Corrected_PF"] - result["Initial_PF"]:+.3f}')
    c3.metric("Capacitor Applied",
              f'{result["Applied_Qc_kVAR"]:.2f} kVAR')
    c4.metric("Active Power",
              f'{result["P_kW"]:.2f} kW')

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Initial Current",
              f'{result["Initial_Current_A"]:.2f} A')
    c6.metric("Corrected Current",
              f'{result["Corrected_Current_A"]:.2f} A')
    c7.metric("Initial Reactive Power",
              f'{result["Initial_Q_kVAR"]:.2f} kVAR')
    c8.metric("Remaining Reactive Power",
              f'{result["Remaining_Q_kVAR"]:.2f} kVAR')

    st.subheader("Before and After Correction")

    comparison = pd.DataFrame({
        "Parameter": ["Power Factor", "Current (A)", "Reactive Power (kVAR)"],
        "Before": [
            result["Initial_PF"],
            result["Initial_Current_A"],
            result["Initial_Q_kVAR"]
        ],
        "After": [
            result["Corrected_PF"],
            result["Corrected_Current_A"],
            result["Remaining_Q_kVAR"]
        ]
    })

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before APFC",
        x=comparison["Parameter"],
        y=comparison["Before"]
    ))
    fig.add_trace(go.Bar(
        name="After APFC",
        x=comparison["Parameter"],
        y=comparison["After"]
    ))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Capacitor Bank Status")
    st.write("Selected capacitor stages:",
             result["Selected_Stages_kVAR"])
    st.write(f'Required compensation: {result["Required_Qc_kVAR"]:.2f} kVAR')
    st.write(f'Applied compensation: {result["Applied_Qc_kVAR"]:.2f} kVAR')

    alarms = generate_alarms(result)
    if alarms == ["NORMAL"]:
        st.success("System status: NORMAL")
    else:
        for alarm in alarms:
            st.warning(alarm)

# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":
    st.subheader("System Overview")
    st.write(
        "Welcome to the Automatic Power Factor Correction "
        "software simulation dashboard."
    )

    if st.session_state.result:
        show_result(st.session_state.result)
    else:
        a, b, c, d = st.columns(4)
        a.metric("System", "Ready")
        b.metric("Simulation Runs", len(st.session_state.history))
        c.metric("PF Limit", "0.85")
        d.metric("Target PF", "0.95")
        st.info("Open Live Simulator to run your first APFC calculation.")

    st.subheader("Application Modules")
    cards = st.columns(3)
    modules = [
        ("Live Monitoring", "Electrical parameter simulation"),
        ("Capacitor Control", "AUTO and MANUAL operation"),
        ("Analytics", "Performance charts and history"),
        ("Alarms", "Low PF and correction warnings"),
        ("Reports", "Download simulation records"),
        ("System", "Educational balanced-load model")
    ]
    for i, (name, description) in enumerate(modules):
        with cards[i % 3]:
            st.markdown(f"**{name}**")
            st.caption(description)

# ============================================================
# LIVE SIMULATOR
# ============================================================

elif page == "Live Simulator":
    st.subheader("Live APFC Simulator")
    st.caption("Enter electrical values and run the correction simulation.")

    with st.form("apfc_form"):
        col1, col2 = st.columns(2)

        with col1:
            system = st.selectbox(
                "Electrical System",
                ["Single-phase", "Three-phase"]
            )
            voltage_v = st.number_input(
                "Voltage (V)",
                min_value=1.0,
                value=415.0 if system == "Three-phase" else 230.0
            )
            power_kw = st.number_input(
                "Active Power (kW)",
                min_value=0.1,
                value=12.0,
                step=0.5
            )

        with col2:
            initial_pf = st.slider(
                "Initial Power Factor",
                min_value=0.10,
                max_value=1.00,
                value=0.72,
                step=0.01
            )
            target_pf = st.slider(
                "Target Power Factor",
                min_value=0.80,
                max_value=1.00,
                value=0.95,
                step=0.01
            )
            mode = st.selectbox(
                "Operating Mode",
                ["AUTO", "MANUAL"]
            )

        if mode == "MANUAL":
            selected_manual = st.multiselect(
                "Select capacitor stages (kVAR)",
                CAPACITOR_STAGES_KVAR,
                default=[]
            )
        else:
            selected_manual = []

        submitted = st.form_submit_button(
            "▶ RUN APFC CALCULATION",
            use_container_width=True
        )

    if submitted:
        result = apfc_correct(
            system,
            voltage_v,
            power_kw,
            initial_pf,
            target_pf,
            CAPACITOR_STAGES_KVAR
        )

        if mode == "MANUAL":
            applied = min(sum(selected_manual), result["Initial_Q_kVAR"])
            remaining = max(result["Initial_Q_kVAR"] - applied, 0.0)
            corrected_pf = (
                power_kw / np.sqrt(power_kw**2 + remaining**2)
                if power_kw > 0 else 1.0
            )
            corrected_s = np.sqrt(power_kw**2 + remaining**2)
            if system == "Single-phase":
                corrected_i = corrected_s * 1000 / voltage_v
            else:
                corrected_i = corrected_s * 1000 / (
                    np.sqrt(3) * voltage_v
                )

            result.update({
                "Applied_Qc_kVAR": applied,
                "Selected_Stages_kVAR": selected_manual,
                "Remaining_Q_kVAR": remaining,
                "Corrected_PF": corrected_pf,
                "Corrected_Current_A": corrected_i
            })

        result["Mode"] = mode
        result["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.result = result
        st.session_state.history.append(result)

    if st.session_state.result:
        show_result(st.session_state.result)

# ============================================================
# CAPACITOR BANK
# ============================================================

elif page == "Capacitor Bank":
    st.subheader("Capacitor Bank Management")
    st.write("Available simulated capacitor stages")

    stages_df = pd.DataFrame({
        "Stage": [f"Stage {i+1}" for i in range(len(CAPACITOR_STAGES_KVAR))],
        "Rating (kVAR)": CAPACITOR_STAGES_KVAR,
        "Status": ["Available"] * len(CAPACITOR_STAGES_KVAR)
    })
    st.dataframe(stages_df, use_container_width=True, hide_index=True)

    st.metric("Total Installed Simulated Capacity",
              f"{sum(CAPACITOR_STAGES_KVAR):.2f} kVAR")

    if st.session_state.result:
        r = st.session_state.result
        st.write("Last selected stages:", r["Selected_Stages_kVAR"])
        st.write(f'Applied compensation: {r["Applied_Qc_kVAR"]:.2f} kVAR')
    else:
        st.info("Run a simulation to view selected capacitor stages.")

# ============================================================
# ANALYTICS
# ============================================================

elif page == "Analytics":
    st.subheader("APFC Analytics")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)

        st.metric("Total Simulation Records", len(df))
        st.metric("Average Corrected PF",
                  f'{df["Corrected_PF"].mean():.3f}')

        fig = px.line(
            df.reset_index(),
            x="index",
            y=["Initial_PF", "Corrected_PF"],
            markers=True,
            title="Power Factor Improvement Across Simulations",
            labels={"index": "Simulation", "value": "Power Factor"}
        )
        fig.update_layout(template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            df.reset_index(),
            x="index",
            y="Applied_Qc_kVAR",
            title="Capacitor Compensation per Simulation"
        )
        fig2.update_layout(template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No simulation history yet. Run simulations first.")

# ============================================================
# ALARMS AND EVENTS
# ============================================================

elif page == "Alarms & Events":
    st.subheader("Alarms and Event Monitoring")

    if st.session_state.history:
        for i, row in enumerate(reversed(st.session_state.history), 1):
            alarms = generate_alarms(row)
            with st.expander(f"Simulation event {i} — {row.get('Timestamp', '')}"):
                st.write("Operating mode:", row.get("Mode", "AUTO"))
                st.write("Initial PF:", row["Initial_PF"])
                st.write("Corrected PF:", row["Corrected_PF"])
                for alarm in alarms:
                    if alarm == "NORMAL":
                        st.success(alarm)
                    else:
                        st.warning(alarm)
    else:
        st.success("No events recorded. Run a simulation to generate events.")

# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":
    st.subheader("Simulation Reports")

    if st.session_state.history:
        report = pd.DataFrame(st.session_state.history)
        st.dataframe(report, use_container_width=True)

        csv = report.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download APFC Report (CSV)",
            data=csv,
            file_name="APFC_Simulation_Report.csv",
            mime="text/csv"
        )
    else:
        st.info("No records available. Run a simulation first.")

# ============================================================
# ABOUT
# ============================================================

elif page == "About":
    st.subheader("About the APFC Management System")
    st.write("""
    This academic application demonstrates Automatic Power Factor
    Correction through software simulation.

    **Main functions**
    - Single-phase and balanced three-phase calculations
    - Reactive power compensation
    - Automatic capacitor stage selection
    - Manual and automatic operating modes
    - Before-and-after performance analysis
    - Alarm generation and CSV reporting

    **Model assumptions**
    - Balanced three-phase loads
    - Ideal sinusoidal electrical quantities
    - 50 Hz frequency
    - Simulated capacitor stages

    This software is not connected to physical electrical equipment.
    It does not simulate switching transients, harmonics, resonance,
    inrush current, discharge delays or protection coordination.
    """)
    st.info("Academic simulation only — not for controlling real equipment.")

st.divider()
st.caption("APFC Management System | Academic Software Simulation")
