import streamlit as st
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm

st.set_page_config(page_title="Nobian MPC Simulation", layout="wide")
st.title("Chlorine Production Flexibility: Forecast Horizon Analysis")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload 'Price duration curves ETM & heat maps.xlsx'", type=['xlsx'])

st.sidebar.header("2. Physical Constants")
C_max = st.sidebar.number_input("C_max (Max Capacity in tons/h)", value=68.0, step=1.0)
L_avg = st.sidebar.slider("L_avg (Average Annual Load)", min_value=0.1, max_value=1.0, value=0.70, step=0.01)
L_min_ratio = st.sidebar.slider("L_min_ratio (Min Safe Part-Load)", min_value=0.1, max_value=1.0, value=0.30, step=0.01)
V_max = st.sidebar.number_input("V_max (Max Storage Volume in tons)", value=7000.0, step=100.0)
energy_intensity = st.sidebar.number_input("Energy Intensity (MWh/ton)", value=2.875, step=0.001)

st.sidebar.header("3. Economic Parameters")
lookahead_input = st.sidebar.text_input(
    "Dynamic Threshold Lookahead in Days (comma-separated):",
    value="1, 2, 7, 14, 30"
)

# Parse the comma-separated string into a list of integers
selected_lookahead_days = []
if lookahead_input:
    try:
        selected_lookahead_days = [int(x.strip()) for x in lookahead_input.split(',')]
    except ValueError:
        st.sidebar.error("Please enter valid numbers separated by commas (e.g., 7, 14, 30)")

st.sidebar.header("4. Select Scenarios")
all_scenarios = {
    'Dumb 1h (Static)': {'h': 1, 'dyn': False},
    '1h (Dynamic)': {'h': 1, 'dyn': True},
    '12h (Dynamic)': {'h': 12, 'dyn': True},
    '24h (Dynamic)': {'h': 24, 'dyn': True},
    '48h (Dynamic)': {'h': 48, 'dyn': True},
    '168h (Dynamic)': {'h': 168, 'dyn': True}
}
selected_scenario_names = st.sidebar.multiselect(
    "Choose scenarios to simulate:",
    options=list(all_scenarios.keys()),
    default=['Dumb 1h (Static)', '24h (Dynamic)']
)

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if st.sidebar.button("Run Simulation") and uploaded_file is not None and len(selected_scenario_names) > 0 and (len(selected_lookahead_days) > 0 or 'Dumb 1h (Static)' in selected_scenario_names):

    # Calculate derived constants
    D_const = C_max * L_avg
    L_min_tons = C_max * L_min_ratio

    # Load Data
    with st.spinner('Loading ENTSO-E Price Data...'):
        df_prices = pd.read_excel(uploaded_file, sheet_name='ENTSOE-E dayahead mid2425')
        df_prices['datetime'] = pd.to_datetime(df_prices['date'].dt.strftime('%Y-%m-%d') + ' ' + df_prices['time'].str[:8])
        df_prices.set_index('datetime', inplace=True)
        df_prices.sort_index(inplace=True)
        prices_eur_mwh = df_prices['price €/MWh'].values
        hours_in_year = len(prices_eur_mwh)

    # Calculate Global Static Threshold
    global_hours_at_max = round(hours_in_year * (L_avg - L_min_ratio) / (1.0 - L_min_ratio))
    global_sorted_prices = np.sort(prices_eur_mwh)
    global_price_threshold = global_sorted_prices[global_hours_at_max - 1]
    global_terminal_value = global_price_threshold * energy_intensity

    # Core Simulation Function
    def run_simulation(forecast_horizon, use_dynamic_threshold, dyn_lookahead_hrs):
        prod_array = np.zeros(hours_in_year)
        storage_array = np.zeros(hours_in_year)
        cost_array = np.zeros(hours_in_year)
        actual_storage = V_max / 2.0

        for t in range(hours_in_year):
            end_t = min(t + forecast_horizon, hours_in_year)
            window_length = end_t - t
            forecast_prices = prices_eur_mwh[t:end_t]

            if use_dynamic_threshold:
                thresh_end_t = min(t + dyn_lookahead_hrs, hours_in_year)
                future_prices = prices_eur_mwh[t:thresh_end_t]
                local_hours = len(future_prices)
                local_hours_at_max = round(local_hours * (L_avg - L_min_ratio) / (1.0 - L_min_ratio))
                idx = max(0, local_hours_at_max - 1)
                dynamic_price_threshold = np.sort(future_prices)[idx]
                active_terminal_value = dynamic_price_threshold * energy_intensity
            else:
                active_terminal_value = global_terminal_value

            model = pulp.LpProblem(f"MPC_{t}", pulp.LpMinimize)
            P = pulp.LpVariable.dicts("Prod", range(window_length), lowBound=L_min_tons, upBound=C_max)
            S = pulp.LpVariable.dicts("Stor", range(window_length), lowBound=0, upBound=V_max)

            model += pulp.lpSum([P[i] * energy_intensity * forecast_prices[i] for i in range(window_length)]) - (S[window_length - 1] * active_terminal_value)

            for i in range(window_length):
                if i == 0:
                    model += S[i] == actual_storage + P[i] - D_const
                else:
                    model += S[i] == S[i-1] + P[i] - D_const

            model.solve(pulp.PULP_CBC_CMD(msg=0))
            optimal_P_now = P[0].varValue

            prod_array[t] = optimal_P_now
            actual_storage = actual_storage + optimal_P_now - D_const
            storage_array[t] = actual_storage
            cost_array[t] = optimal_P_now * energy_intensity * forecast_prices[0]

        return np.sum(cost_array), prod_array, storage_array

    # Generate Run Combinations (Cartesian Product)
    runs_to_execute = []
    for name in selected_scenario_names:
        s = all_scenarios[name]
        if not s['dyn']:
            runs_to_execute.append({'label': name, 'h': s['h'], 'dyn': False, 'lh_hrs': 336})
        else:
            for lh_days in selected_lookahead_days:
                lh_hrs = lh_days * 24
                label = f"{name} | {lh_days}d Lookahead"
                runs_to_execute.append({'label': label, 'h': s['h'], 'dyn': True, 'lh_hrs': lh_hrs})

    # Execute Scenarios
    results_data = {}
    rigid_cost_array = D_const * energy_intensity * prices_eur_mwh
    total_rigid_cost = np.sum(rigid_cost_array)

    # Dynamic color mapping
    cmap = cm.get_cmap('tab10')
    colors = {run['label']: cmap(i % 10) for i, run in enumerate(runs_to_execute)}

    if 'Dumb 1h (Static)' in colors:
        colors['Dumb 1h (Static)'] = 'red'

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, run in enumerate(runs_to_execute):
        label = run['label']
        status_text.text(f"Running {label}...")

        cost, prod, stor = run_simulation(run['h'], run['dyn'], run['lh_hrs'])

        prices_at_max = prices_eur_mwh[prod >= C_max - 0.1]
        prices_at_min = prices_eur_mwh[prod <= L_min_tons + 0.1]

        results_data[label] = {
            'cost': cost,
            'prod': prod,
            'stor': stor,
            'prices_at_max': prices_at_max,
            'prices_at_min': prices_at_min,
            'color': colors[label]
        }
        progress_bar.progress((i + 1) / len(runs_to_execute))

    status_text.text("Simulation Complete!")

    # Financial Summary Table (Inventory Adjusted)
    st.subheader("Financial Performance (Inventory Adjusted)")
    summary_data = []
    avg_price_per_ton = np.mean(prices_eur_mwh) * energy_intensity

    for label in results_data.keys():
        raw_cost = results_data[label]['cost']
        final_storage = results_data[label]['stor'][-1]
        starting_storage = V_max / 2.0

        inventory_delta_tons = starting_storage - final_storage
        inventory_adjustment_eur = inventory_delta_tons * avg_price_per_ton

        true_adjusted_cost = raw_cost + inventory_adjustment_eur
        true_savings = total_rigid_cost - true_adjusted_cost

        summary_data.append({
            "Scenario Combination": label,
            "Final Tank Level (Tons)": f"{final_storage:,.0f}",
            "Adjusted True Cost (€)": f"€{true_adjusted_cost:,.0f}",
            "True Savings (€)": f"€{true_savings:,.0f}"
        })

    st.table(pd.DataFrame(summary_data))

    # Visualizations
    st.subheader("Comparative Analysis")
    fig, axes = plt.subplots(4, 1, figsize=(14, 24))

    for label in results_data.keys():
        c = results_data[label]['color']
        axes[0].plot(np.arange(hours_in_year), results_data[label]['stor'], color=c, alpha=0.6, label=label)
        axes[1].plot(np.arange(hours_in_year), np.sort(results_data[label]['stor'])[::-1], color=c, linewidth=2, label=label)

    axes[0].set_title('Chronological Storage Level', fontweight='bold')
    axes[0].set_ylabel('Storage Level (Tons)')
    axes[0].axhline(V_max, color='black', linestyle='--', label='Max Capacity')
    axes[0].legend(loc='upper right', fontsize='small')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Storage Level Duration Curve', fontweight='bold')
    axes[1].set_ylabel('Storage Level (Tons)')
    axes[1].axhline(V_max, color='black', linestyle='--')
    axes[1].legend(loc='upper right', fontsize='small')
    axes[1].grid(True, alpha=0.3)

    prod_arrays = [results_data[label]['prod'] for label in results_data.keys()]
    c_list = [results_data[label]['color'] for label in results_data.keys()]
    axes[2].hist(prod_arrays, bins=10, color=c_list, label=list(results_data.keys()), edgecolor='black')
    axes[2].set_title('Hourly Production Frequency Distribution', fontweight='bold')
    axes[2].set_xlabel('Production Rate (Tons/Hour)')
    axes[2].set_ylabel('Frequency (Hours)')
    axes[2].legend(loc='upper center', fontsize='small')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(np.arange(hours_in_year), np.sort(prices_eur_mwh)[::-1], color='black', linewidth=3, label='Overall Market Price', zorder=1)

    for label in results_data.keys():
        c = results_data[label]['color']
        p_max = results_data[label]['prices_at_max']
        p_min = results_data[label]['prices_at_min']

        axes[3].plot(np.arange(len(p_min)), np.sort(p_min)[::-1], color=c, linestyle=':', linewidth=2, alpha=0.8, label=f'{label} (30% Load)', zorder=2)

        x_offset = hours_in_year - len(p_max)
        axes[3].plot(np.arange(x_offset, hours_in_year), np.sort(p_max)[::-1], color=c, linestyle='-', linewidth=2, label=f'{label} (100% Load)', zorder=2)

    axes[3].set_title('Electricity Price Duration Curves based on Operational State', fontweight='bold')
    axes[3].set_xlabel('Cumulative Hours')
    axes[3].set_ylabel('Electricity Price (€/MWh)')
    axes[3].legend(loc='upper right', fontsize='small', ncol=2)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

elif uploaded_file is None:
    st.info("Please upload the price dataset via the sidebar to begin.")
else:
    st.warning("Please enter at least one valid number for lookahead days and select a scenario.")
