import streamlit as st
import numpy as np
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import time

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
dynamic_lookahead_hours = st.sidebar.number_input("Dynamic Threshold Lookahead (Hours)", value=336, step=24)

st.sidebar.header("4. Select Scenarios")
all_scenarios = {
    'Dumb 1h (Static)': {'h': 1, 'dyn': False, 'color': 'red'},
    '1h (Dynamic)': {'h': 1, 'dyn': True, 'color': 'gray'},
    '12h (Dynamic)': {'h': 12, 'dyn': True, 'color': 'orange'},
    '24h (Dynamic)': {'h': 24, 'dyn': True, 'color': 'blue'},
    '48h (Dynamic)': {'h': 48, 'dyn': True, 'color': 'green'},
    '168h (Dynamic)': {'h': 168, 'dyn': True, 'color': 'purple'}
}
selected_scenario_names = st.sidebar.multiselect(
    "Choose scenarios to simulate:",
    options=list(all_scenarios.keys()),
    default=['Dumb 1h (Static)', '24h (Dynamic)']
)

# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if st.sidebar.button("Run Simulation") and uploaded_file is not None and len(selected_scenario_names) > 0:
    
    # Calculate derived constants based on user input
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
    def run_simulation(forecast_horizon, use_dynamic_threshold=True):
        prod_array = np.zeros(hours_in_year)
        storage_array = np.zeros(hours_in_year)
        cost_array = np.zeros(hours_in_year)
        actual_storage = V_max / 2.0  
        
        for t in range(hours_in_year):
            end_t = min(t + forecast_horizon, hours_in_year)
            window_length = end_t - t
            forecast_prices = prices_eur_mwh[t:end_t]
            
            if use_dynamic_threshold:
                thresh_end_t = min(t + dynamic_lookahead_hours, hours_in_year)
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

    # Execute Selected Scenarios
    results_data = {}
    rigid_cost_array = D_const * energy_intensity * prices_eur_mwh
    total_rigid_cost = np.sum(rigid_cost_array)

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, name in enumerate(selected_scenario_names):
        status_text.text(f"Running {name}...")
        s = all_scenarios[name]
        cost, prod, stor = run_simulation(s['h'], s['dyn'])
        results_data[name] = {'cost': cost, 'prod': prod, 'stor': stor, 'color': s['color']}
        progress_bar.progress((i + 1) / len(selected_scenario_names))
        
    status_text.text("Simulation Complete!")

    # Financial Summary Table
    st.subheader("Financial Performance")
    summary_data = []
    for name in selected_scenario_names:
        cost = results_data[name]['cost']
        savings = total_rigid_cost - cost
        summary_data.append({"Scenario": name, "Total Cost (€)": f"€{cost:,.0f}", "Total Savings vs Rigid (€)": f"€{savings:,.0f}"})
    
    st.table(pd.DataFrame(summary_data))

    # Visualizations
    st.subheader("Comparative Analysis")
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    for name in selected_scenario_names:
        c = results_data[name]['color']
        # Chronological
        axes[0].plot(np.arange(hours_in_year), results_data[name]['stor'], color=c, alpha=0.6, label=name)
        # Duration Curve
        axes[1].plot(np.arange(hours_in_year), np.sort(results_data[name]['stor'])[::-1], color=c, linewidth=2, label=name)
        
    axes[0].set_title('Chronological Storage Level', fontweight='bold')
    axes[0].set_ylabel('Storage Level (Tons)')
    axes[0].axhline(V_max, color='black', linestyle='--', label='Max Capacity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Storage Level Duration Curve', fontweight='bold')
    axes[1].set_ylabel('Storage Level (Tons)')
    axes[1].axhline(V_max, color='black', linestyle='--')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Grouped Histogram
    prod_arrays = [results_data[name]['prod'] for name in selected_scenario_names]
    colors = [results_data[name]['color'] for name in selected_scenario_names]
    axes[2].hist(prod_arrays, bins=10, color=colors, label=selected_scenario_names, edgecolor='black')
    axes[2].set_title('Hourly Production Frequency Distribution', fontweight='bold')
    axes[2].set_xlabel('Production Rate (Tons/Hour)')
    axes[2].set_ylabel('Frequency (Hours)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

elif uploaded_file is None:
    st.info("Please upload the price dataset via the sidebar to begin.")
elif len(selected_scenario_names) == 0:
    st.warning("Please select at least one scenario to run.")
