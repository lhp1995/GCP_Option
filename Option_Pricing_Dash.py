# %% Imports
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import time
import traceback # For detailed error messages

# ==============================================================================
# %% Block 1: Option Valuation Class Definition (Unchanged from original)
# ==============================================================================
class option_valuation:
    """
    Class to value European and American options using Monte Carlo simulation
    under different underlying asset models (GBM, Jump Diffusion, GARCH).
    (Full class code as previously defined)
    """
    # Class attributes:
    def __init__(self, S0, E, Tt, Vol, rf, dt, call_put, sim_number,
                 option_style='European',  # 'European' or 'American'
                 dividend_yield=0.0,     # Continuous dividend yield (q)
                 model_type='GBM',         # 'GBM', 'JumpDiffusion', 'GARCH'
                 # Jump Diffusion parameters (only used if model_type='JumpDiffusion')
                 jump_lambda=0.0, jump_mu=0.0, jump_sigma=0.1, # Added default sigma
                 # GARCH(1,1) parameters (only used if model_type='GARCH')
                 garch_omega=1e-6, garch_alpha=0.1, garch_beta=0.85): # Added defaults

        # --- Input Validation ---
        if dt <= 0: raise ValueError("dt (time step) must be positive.")
        if Tt < 0: raise ValueError("Tt (Time to Maturity) cannot be negative.")
        if sim_number <= 0: raise ValueError("sim_number must be positive.")
        if Vol < 0: raise ValueError("Volatility cannot be negative.")
        # Add more validation as needed...

        # --- Store Core Parameters ---
        self.S0 = S0
        self.E = E
        self.Tt = Tt
        self.Vol = Vol
        self.rf = rf
        self.dt = dt
        self.call_put = call_put
        self.sim_number = int(sim_number) # Ensure integer
        self.option_style = option_style
        self.dividend_yield = dividend_yield # Used internally by class methods
        self.model_type = model_type

        # --- Store Model Specific Parameters ---
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma
        self.garch_omega = garch_omega
        self.garch_alpha = garch_alpha
        self.garch_beta = garch_beta

        # --- Derived Parameters ---
        self.time_steps = int(Tt / dt) if Tt > 0 and dt > 0 else 0 # Ensure Tt/dt > 0

        # Handle Tt=0 or Tt<dt case gracefully
        if self.time_steps <= 0:
             self.time_steps = 0
             self.price_path = pd.DataFrame(np.full((1, self.sim_number), self.S0))
             self.vol_path = None
             payoff_at_expiry = np.maximum(self.call_put * (self.S0 - self.E), 0)
             self.option_value = payoff_at_expiry # No discounting needed if Tt=0
             return # Skip simulation and rest of init

        # --- Seed for reproducibility ---
        # Simple seed for demonstration; consider more robust seeding if needed
        np.random.seed(self.sim_number + int(self.S0 * 100) + int(self.E * 100) + int(self.Tt * 100))

        # --- Simulate Paths ---
        self.simulate_paths() # Generates self.price_path (and self.vol_path if GARCH)

        # --- Value Option ---
        if self.option_style == 'European':
            self.option_value = self.european_option_valuation()
        elif self.option_style == 'American':
            if self.model_type != 'GBM':
                 # In Streamlit, we might show a warning using st.warning if needed
                 pass
            self.option_value = self.american_option_valuation_LS()
        else:
            raise ValueError("option_style must be 'European' or 'American'")

    # --- Path Simulation Methods ---
    def simulate_paths(self):
        if self.time_steps <= 0: return
        try:
            if self.model_type == 'GBM': self._simulate_gbm()
            elif self.model_type == 'JumpDiffusion': self._simulate_jump_diffusion()
            elif self.model_type == 'GARCH': self._simulate_garch()
            else: raise ValueError(f"Unsupported model_type: {self.model_type}")
        except Exception as e:
            # Catch potential numerical errors during simulation
            raise RuntimeError(f"Error during path simulation ({self.model_type}): {e}") from e

    def _generate_random_factors(self, num_sets=1):
        factors = []
        for _ in range(num_sets):
            # Ensure size is positive even if time_steps=0 (though handled earlier)
            rows = max(1, self.time_steps)
            Z_arr = np.zeros((rows + 1, self.sim_number))
            # Generate standard normal random numbers for steps 1 to time_steps
            Z_arr[1:, :] = norm.rvs(size=(rows, self.sim_number))
            factors.append(pd.DataFrame(Z_arr))
        return factors

    def _simulate_gbm(self):
        Z, = self._generate_random_factors(1)
        S = np.zeros((self.time_steps + 1, self.sim_number))
        S[0] = self.S0
        # Use self.dividend_yield here
        drift = (self.rf - self.dividend_yield - 0.5 * self.Vol**2) * self.dt
        diffusion = self.Vol * np.sqrt(self.dt)
        for i in range(1, self.time_steps + 1):
            # Ensure S doesn't go negative (though exp should prevent this)
            S[i] = S[i-1] * np.exp(drift + diffusion * Z.iloc[i, :])
            S[i] = np.maximum(S[i], 0) # Floor at zero
        self.price_path = pd.DataFrame(S)
        self.vol_path = None

    def _simulate_jump_diffusion(self):
        # Validate Jump Sigma
        if self.jump_sigma <= 0: raise ValueError("jump_sigma must be positive for Jump Diffusion.")
        Z_diff, Z_jump_size = self._generate_random_factors(2)
        poisson_draws = poisson.rvs(self.jump_lambda * self.dt, size=(self.time_steps, self.sim_number))

        # Merton's jump correction term E[Y-1] = exp(mu + 0.5*sig^2) - 1
        jump_drift_comp = self.jump_lambda * (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1)

        S = np.zeros((self.time_steps + 1, self.sim_number))
        S[0] = self.S0
        # Use self.dividend_yield here
        drift = (self.rf - self.dividend_yield - jump_drift_comp - 0.5 * self.Vol**2) * self.dt
        diffusion = self.Vol * np.sqrt(self.dt)

        for i in range(1, self.time_steps + 1):
            num_jumps = poisson_draws[i-1, :]
            safe_num_jumps = np.maximum(num_jumps, 0) # Poisson shouldn't be neg, but safety

            # Sum of k N(mu, sig^2) variables is N(k*mu, k*sig^2)
            # Jump component is sum(log(Y_j)) ~ N(num_jumps*mu, num_jumps*sigma^2)
            log_jump_sum_mean = safe_num_jumps * self.jump_mu
            log_jump_sum_std = np.sqrt(safe_num_jumps) * self.jump_sigma

            # Generate jump sum directly from normal distribution
            # Add small epsilon to scale to avoid scale=0 warning if num_jumps=0
            log_jump_sum = norm.rvs(loc=log_jump_sum_mean, scale=log_jump_sum_std + 1e-12)
            log_jump_sum = np.nan_to_num(log_jump_sum) # Handle cases where std dev might be zero if num_jumps=0

            # Apply update S(t) = S(t-1) * exp(gbm_part) * exp(jump_part)
            S[i] = S[i-1] * np.exp(drift + diffusion * Z_diff.iloc[i, :] + log_jump_sum)
            S[i] = np.maximum(S[i], 0) # Floor at zero

        self.price_path = pd.DataFrame(S)
        self.vol_path = None

    def _simulate_garch(self):
         # Validate GARCH params
        if self.garch_omega <= 0 or self.garch_alpha < 0 or self.garch_beta < 0:
            st.warning(f"GARCH params omega={self.garch_omega}, alpha={self.garch_alpha}, beta={self.garch_beta} may be invalid (non-positive).")
        # if self.garch_alpha + self.garch_beta >= 1:
        #     st.warning("GARCH process may not be stationary (alpha + beta >= 1).")

        Z, = self._generate_random_factors(1)
        S = np.zeros((self.time_steps + 1, self.sim_number))
        V = np.zeros((self.time_steps + 1, self.sim_number)) # Variance path
        S[0] = self.S0
        V[0] = self.Vol**2 # Use Vol as initial variance

        for i in range(1, self.time_steps + 1):
            current_var = V[i-1]
            current_var = np.maximum(current_var, 1e-9) # Ensure positive variance
            current_sigma = np.sqrt(current_var)

            # Use self.dividend_yield here
            drift_term = (self.rf - self.dividend_yield - 0.5 * current_var) * self.dt
            diffusion_term = current_sigma * np.sqrt(self.dt) * Z.iloc[i, :]
            log_return = drift_term + diffusion_term
            S[i] = S[i-1] * np.exp(log_return)
            S[i] = np.maximum(S[i], 0) # Floor at zero

            # Update variance for next step using squared log return
            # Apply GARCH(1,1) update rule
            V[i] = self.garch_omega + self.garch_alpha * log_return**2 + self.garch_beta * current_var
            V[i] = np.maximum(V[i], 1e-9) # Ensure variance stays positive

        self.price_path = pd.DataFrame(S)
        self.vol_path = pd.DataFrame(np.sqrt(V)) # Store volatility

    # --- Option Valuation Methods ---
    def european_option_valuation(self):
        if self.time_steps <= 0: return self.option_value
        final_prices = self.price_path.iloc[self.time_steps, :]
        payoffs = np.maximum(self.call_put * (final_prices - self.E), 0)
        # Handle potential NaNs or Infs in payoffs if simulation failed
        valid_payoffs = payoffs[np.isfinite(payoffs)]
        if len(valid_payoffs) == 0: return np.nan # Indicate calculation failure
        expected_payoff = np.mean(valid_payoffs)
        V0 = np.exp(-self.rf * self.Tt) * expected_payoff
        return V0

    def american_option_valuation_LS(self, poly_degree=3):
        if self.time_steps <= 0: return self.option_value
        if not isinstance(self.price_path, pd.DataFrame) or self.price_path.empty:
             raise ValueError("Price path is invalid for American valuation.")

        prices = self.price_path.values
        # Ensure prices are finite
        if not np.all(np.isfinite(prices)):
            # Maybe show a warning in Streamlit context
            # st.warning("Non-finite values found in price paths. Replacing with S0.")
            prices = np.nan_to_num(prices, nan=self.S0, posinf=self.S0, neginf=0) # Replace NaNs/Infs

        payoffs = np.maximum(self.call_put * (prices - self.E), 0)
        cash_flow = np.zeros_like(payoffs)
        cash_flow[self.time_steps, :] = payoffs[self.time_steps, :]
        discount = np.exp(-self.rf * self.dt)

        for t in range(self.time_steps - 1, 0, -1):
            current_S = prices[t, :]
            value_if_held = cash_flow[t+1, :] * discount

            in_the_money_paths = payoffs[t, :] > 1e-6 # Check if significantly ITM
            idx_itm = np.where(in_the_money_paths)[0]
            continuation_value = np.zeros(self.sim_number)

            if len(idx_itm) > poly_degree: # Need points for regression
                S_itm = current_S[idx_itm]
                future_val_itm = value_if_held[idx_itm]

                # Filter out potential non-finite values before regression
                mask = np.isfinite(S_itm) & np.isfinite(future_val_itm)
                if np.sum(mask) > poly_degree:
                    S_clean = S_itm[mask]
                    future_val_clean = future_val_itm[mask]

                    # Check if S values are distinct enough for polynomial regression
                    if len(np.unique(np.round(S_clean, 5))) > poly_degree:
                        try:
                            coeffs = np.polyfit(S_clean, future_val_clean, poly_degree)
                            estimated_continuation_clean = np.polyval(coeffs, S_clean)

                            # Map results back to original ITM indices
                            temp_continuation = np.zeros_like(S_itm) * np.nan
                            temp_continuation[mask] = estimated_continuation_clean
                            # Fill NaNs with 0 where estimation failed or wasn't possible
                            continuation_value[idx_itm] = np.nan_to_num(temp_continuation, nan=0.0)

                        except (np.linalg.LinAlgError, TypeError, ValueError) as fit_err:
                             # Fallback: maybe use intrinsic value or zero? Using zero.
                             # Consider logging: print(f"Warning: Regression failed at step {t}: {fit_err}")
                             pass # continuation_value remains 0
                    # else: print(f"Warning: Not enough distinct S values for regression at step {t}")
                # else: print(f"Warning: Not enough finite points for regression at step {t}")


            # Ensure continuation value is non-negative
            continuation_value = np.maximum(continuation_value, 0)
            exercise_now = payoffs[t, :] > continuation_value
            exercise_decision_paths = np.where(exercise_now & in_the_money_paths)[0]

            cash_flow[t, :] = value_if_held
            cash_flow[t, exercise_decision_paths] = payoffs[t, exercise_decision_paths]

        # Final value is the mean of the cash flows at t=1, discounted to t=0
        final_cashflows = cash_flow[1, :]
        valid_final_cashflows = final_cashflows[np.isfinite(final_cashflows)]
        if len(valid_final_cashflows) == 0: return np.nan
        V0 = np.mean(valid_final_cashflows) * discount
        return V0

# ==============================================================================
# %% Block 2: Streamlit App Setup & UI Definition
# ==============================================================================

st.set_page_config(layout="wide", page_title="Monte Carlo Option Pricer")

st.title("Monte Carlo Option Pricer & Analyzer")

# --- Input Sidebar ---
st.sidebar.header("Configuration")

# --- Option Parameters ---
st.sidebar.subheader("Option Parameters")
s0 = st.sidebar.number_input("Spot (S0):", min_value=0.01, value=100.0, step=1.0, format="%.2f")
e = st.sidebar.number_input("Strike (E):", min_value=0.01, value=100.0, step=1.0, format="%.2f")
tt = st.sidebar.number_input("Maturity (Tt, yrs):", min_value=0.01, value=1.0, step=0.1, format="%.2f")
vol = st.sidebar.number_input("Volatility (Vol):", min_value=0.0, value=0.20, step=0.01, format="%.2f")
rf = st.sidebar.number_input("Risk-Free Rate (rf):", min_value=0.0, value=0.05, step=0.005, format="%.3f")
# 'div' is the variable holding the dividend yield input
div = st.sidebar.number_input("Dividend Yield (q):", min_value=0.0, value=0.01, step=0.005, format="%.3f")

call_put_str = st.sidebar.selectbox("Type:", options=["Call", "Put"], index=0)
option_style = st.sidebar.selectbox("Style:", options=["European", "American"], index=0)

# Map string selection to numerical value expected by the class
cp = 1 if call_put_str == "Call" else -1

st.sidebar.divider()

# --- Model Selection ---
st.sidebar.subheader("Underlying Model")
model_type = st.sidebar.selectbox("Model:", options=["GBM", "JumpDiffusion", "GARCH"], index=0)

# --- Conditional Model Parameters ---
model_args = {}
jump_params = {}
garch_params = {}

if model_type == 'JumpDiffusion':
    st.sidebar.subheader("Jump Diffusion Params:")
    jump_params['jump_lambda'] = st.sidebar.number_input("Lambda (λ):", min_value=0.0, value=0.5, step=0.1, format="%.2f")
    jump_params['jump_mu'] = st.sidebar.number_input("Mean (μ):", value=-0.05, step=0.01, format="%.2f")
    jump_params['jump_sigma'] = st.sidebar.number_input("Sigma (σj):", min_value=0.001, value=0.25, step=0.01, format="%.3f")
    model_args = jump_params
elif model_type == 'GARCH':
    st.sidebar.subheader("GARCH(1,1) Params:")
    # Use text input for scientific notation flexibility, convert later
    garch_omega_str = st.sidebar.text_input("Omega (ω):", value='1e-6')
    garch_params['garch_alpha'] = st.sidebar.number_input("Alpha (α):", min_value=0.0, value=0.09, step=0.01, format="%.3f")
    garch_params['garch_beta'] = st.sidebar.number_input("Beta (β):", min_value=0.0, value=0.90, step=0.01, format="%.3f")
    # Attempt conversion for validation inside button click
    model_args = garch_params # Store user inputs for now

st.sidebar.divider()

# --- Simulation Settings ---
st.sidebar.subheader("Simulation Settings")
simulation_runs_app = st.sidebar.number_input("Simulations (Price):", min_value=100, value=1000, step=100)
ts_year = st.sidebar.number_input("Time Steps/Year:", min_value=1, value=50, step=1)
sim_plot = st.sidebar.number_input("Sims per Graph Pt:", min_value=50, value=500, step=50)
num_points_plot_app = st.sidebar.number_input("Points per Graph:", min_value=5, max_value=50, value=15, step=1)

st.sidebar.divider()

# --- Action Button ---
calculate_button = st.sidebar.button("Calculate Price & Update Graphs", type="primary", use_container_width=True)

# ==============================================================================
# %% Block 3: Calculation and Output Area
# ==============================================================================

# --- Helper function to run valuation - cached for performance ---
# Note: The 'div' argument is expected here by this function definition
@st.cache_data(show_spinner=False) # Use spinner within the button logic
def run_valuation(s0, e, tt, vol, rf, base_dt, cp, sim_num, style, div, model, **model_params):
    """Runs a single option valuation. Cached by Streamlit."""
    try:
        # The option_valuation class internally expects 'dividend_yield'
        option_instance = option_valuation(
            S0=s0, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, call_put=cp, sim_number=sim_num,
            option_style=style, dividend_yield=div, model_type=model, **model_params
        )
        return option_instance.option_value
    except Exception as err:
        # Return the error to be handled by the caller
        return err

# --- Define placeholders for outputs ---
price_placeholder = st.empty()
graph_cols = st.columns(2) # Create columns for graphs
graph_placeholders = {
    's0': graph_cols[0].empty(),
    'e': graph_cols[1].empty(),
    'tt': graph_cols[0].empty(),
    'vol': graph_cols[1].empty(),
    'rf': graph_cols[0].empty(),
    'div': graph_cols[1].empty(),
}
log_expander = st.expander("Show Calculation Logs", expanded=False)
log_placeholder = log_expander.empty()


# --- Calculation Logic (Runs when button is clicked) ---
if calculate_button:
    start_calc_time = time.time()
    status_messages = []
    error_occurred = False
    graphs = {} # To store generated figures

    # --- Initial Price Display ---
    price_placeholder.info("Calculating...", icon="⏳")
    for key in graph_placeholders:
        graph_placeholders[key].info(f"Generating Graph {key}...", icon="⏳")
    log_placeholder.info("Starting calculations...")

    with st.spinner(f"Running Monte Carlo ({model_type})... Please wait."):
        try:
            # --- Input Validation ---
            if tt <= 0 or vol < 0 or simulation_runs_app <= 0 or ts_year <= 0 or sim_plot <= 0 or num_points_plot_app <= 0:
                raise ValueError("Tt, Vol, Simulations, Time Steps, Plot Sims/Points must be positive.")
            if e <= 0: raise ValueError("Strike (E) must be positive.")
            if s0 <= 0: raise ValueError("Spot (S0) must be positive.")
            base_dt = 1.0 / ts_year

            # --- Parse Model Specific Args (with validation) ---
            if model_type == 'GARCH':
                try:
                    # Convert omega here
                    model_args['garch_omega'] = float(garch_omega_str)
                    if model_args['garch_omega'] < 0:
                         st.warning("GARCH Omega (ω) should ideally be non-negative.")
                except ValueError:
                    raise ValueError(f"Invalid GARCH Omega (ω): '{garch_omega_str}'. Must be a number.")
                if model_args['garch_alpha'] < 0 or model_args['garch_beta'] < 0:
                     st.warning("GARCH Alpha (α) and Beta (β) should ideally be non-negative.")
            elif model_type == 'JumpDiffusion':
                 if model_args['jump_sigma'] <= 0:
                     raise ValueError("Jump Sigma (σj) must be positive.")

            status_messages.append(f"Using Model: {model_type} with params: {model_args}")
            status_messages.append(f"Calculating base price ({option_style} {call_put_str})...")


            # --- 1. Calculate Single Price using Cached Function ---
            # Call run_valuation with the 'div' argument it expects
            base_price_result = run_valuation(
                s0=s0, e=e, tt=tt, vol=vol, rf=rf, base_dt=base_dt, cp=cp, sim_num=simulation_runs_app,
                style=option_style, div=div, model=model_type, **model_args
            )

            if isinstance(base_price_result, Exception):
                 raise base_price_result # Raise error caught in cached function
            elif np.isnan(base_price_result):
                 raise ValueError("Base price calculation returned NaN.")
            else:
                 option_value_base = base_price_result
                 price_placeholder.metric(
                     label=f"{option_style} {call_put_str} Price ({model_type})",
                     value=f"{option_value_base:.4f}"
                 )
                 status_messages.append(f"-> Base price calculated: {option_value_base:.4f}")


            # --- 2. Calculate Data for Graphs ---
            status_messages.append(f"\nGenerating sensitivity graphs (Sims/Pt={sim_plot}, Pts={num_points_plot_app})...")
            sim_plot_actual = max(50, int(sim_plot))
            num_points_actual = max(5, int(num_points_plot_app))

            # Define common arguments for graph valuation calls
            # These are passed via **kwargs to run_valuation
            common_plot_args_graphs = {
                'sim_num': sim_plot_actual,
                'style': option_style,
                'model': model_type,
                **model_args
            }

            # --- S0 Graph ---
            status_messages.append(" Calculating S0 sensitivity...")
            s0_range_plot = np.linspace(max(0.1, s0 * 0.7), s0 * 1.3, num_points_actual)
            # CORRECTED CALL: Use 'div=div' matching run_valuation definition
            prices_s0_call = [run_valuation(s0=s_val, e=e, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div, cp=1, **common_plot_args_graphs) for s_val in s0_range_plot]
            prices_s0_put = [run_valuation(s0=s_val, e=e, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div, cp=-1, **common_plot_args_graphs) for s_val in s0_range_plot]

            fig_s0 = go.Figure()
            fig_s0.add_trace(go.Scatter(x=s0_range_plot, y=prices_s0_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_s0.add_trace(go.Scatter(x=s0_range_plot, y=prices_s0_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_s0.add_vline(x=e, line_dash="dash", line_color="grey", annotation_text=f"E={e:.2f}")
            fig_s0.update_layout(title=f"vs Spot Price ({model_type})", xaxis_title="Spot Price (S0)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            graphs['s0'] = fig_s0

            # --- E Graph ---
            status_messages.append(" Calculating E sensitivity...")
            e_range_plot = np.linspace(max(0.1, e * 0.7), e * 1.3, num_points_actual)
            # CORRECTED CALL: Use 'div=div'
            prices_e_call = [run_valuation(s0=s0, e=e_val, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div, cp=1, **common_plot_args_graphs) for e_val in e_range_plot]
            prices_e_put = [run_valuation(s0=s0, e=e_val, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div, cp=-1, **common_plot_args_graphs) for e_val in e_range_plot]

            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(x=e_range_plot, y=prices_e_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_e.add_trace(go.Scatter(x=e_range_plot, y=prices_e_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_e.add_vline(x=s0, line_dash="dash", line_color="grey", annotation_text=f"S0={s0:.2f}")
            fig_e.update_layout(title=f"vs Strike Price ({model_type})", xaxis_title="Strike Price (E)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
            graphs['e'] = fig_e

            # --- Tt Graph ---
            status_messages.append(" Calculating Tt sensitivity...")
            min_tt_plot = max(base_dt * 5, 0.02) # Ensure Tt > 0 for plotting
            tt_range_plot = np.linspace(min_tt_plot, tt * 1.5 + min_tt_plot, num_points_actual)
            prices_tt_call = []
            prices_tt_put = []
            for tt_val in tt_range_plot:
                 # Adjust dt for short maturities if needed, or keep base_dt for consistency
                 current_dt = min(base_dt, tt_val / 10 if tt_val > 0 else base_dt) # Ensure dt <= Tt/10 roughly
                 # CORRECTED CALL: Use 'div=div'
                 prices_tt_call.append(run_valuation(s0=s0, e=e, tt=tt_val, vol=vol, rf=rf, base_dt=current_dt, div=div, cp=1, **common_plot_args_graphs))
                 prices_tt_put.append(run_valuation(s0=s0, e=e, tt=tt_val, vol=vol, rf=rf, base_dt=current_dt, div=div, cp=-1, **common_plot_args_graphs))

            fig_tt = go.Figure()
            fig_tt.add_trace(go.Scatter(x=tt_range_plot, y=prices_tt_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_tt.add_trace(go.Scatter(x=tt_range_plot, y=prices_tt_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_tt.update_layout(title=f"vs Maturity ({model_type})", xaxis_title="Time to Maturity (Tt, years)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            graphs['tt'] = fig_tt

            # --- Vol Graph ---
            status_messages.append(" Calculating Vol sensitivity...")
            vol_range_plot = np.linspace(max(0.01, vol * 0.2), vol * 2.0 + 0.01, num_points_actual)
            # CORRECTED CALL: Use 'div=div'
            prices_vol_call = [run_valuation(s0=s0, e=e, tt=tt, vol=vol_val, rf=rf, base_dt=base_dt, div=div, cp=1, **common_plot_args_graphs) for vol_val in vol_range_plot]
            prices_vol_put = [run_valuation(s0=s0, e=e, tt=tt, vol=vol_val, rf=rf, base_dt=base_dt, div=div, cp=-1, **common_plot_args_graphs) for vol_val in vol_range_plot]

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(x=vol_range_plot, y=prices_vol_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_vol.add_trace(go.Scatter(x=vol_range_plot, y=prices_vol_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_vol.update_layout(title=f"vs Volatility ({model_type})", xaxis_title="Volatility (Vol)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            graphs['vol'] = fig_vol

            # --- rf Graph ---
            status_messages.append(" Calculating rf sensitivity...")
            rf_range_plot = np.linspace(max(0.0, rf - 0.04), rf + 0.05, num_points_actual)
            # CORRECTED CALL: Use 'div=div'
            prices_rf_call = [run_valuation(s0=s0, e=e, tt=tt, vol=vol, rf=rf_val, base_dt=base_dt, div=div, cp=1, **common_plot_args_graphs) for rf_val in rf_range_plot]
            prices_rf_put = [run_valuation(s0=s0, e=e, tt=tt, vol=vol, rf=rf_val, base_dt=base_dt, div=div, cp=-1, **common_plot_args_graphs) for rf_val in rf_range_plot]

            fig_rf = go.Figure()
            fig_rf.add_trace(go.Scatter(x=rf_range_plot, y=prices_rf_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_rf.add_trace(go.Scatter(x=rf_range_plot, y=prices_rf_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_rf.update_layout(title=f"vs Risk-Free Rate ({model_type})", xaxis_title="Risk-Free Rate (rf)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            graphs['rf'] = fig_rf

            # --- div Graph ---
            status_messages.append(" Calculating Dividend sensitivity...")
            div_range_plot = np.linspace(0.0, max(0.1, div * 2.0 + 0.02) , num_points_actual)
            # CORRECTED CALL: Use 'div=div_val' matching the loop variable
            prices_div_call = [run_valuation(s0=s0, e=e, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div_val, cp=1, **common_plot_args_graphs) for div_val in div_range_plot]
            prices_div_put = [run_valuation(s0=s0, e=e, tt=tt, vol=vol, rf=rf, base_dt=base_dt, div=div_val, cp=-1, **common_plot_args_graphs) for div_val in div_range_plot]

            fig_div = go.Figure()
            fig_div.add_trace(go.Scatter(x=div_range_plot, y=prices_div_call, mode='lines+markers', name=f'{option_style} Call'))
            fig_div.add_trace(go.Scatter(x=div_range_plot, y=prices_div_put, mode='lines+markers', name=f'{option_style} Put'))
            fig_div.update_layout(title=f"vs Dividend Yield ({model_type})", xaxis_title="Dividend Yield (q)", yaxis_title="Option Price", height=350, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
            graphs['div'] = fig_div

            # --- Update Graph Placeholders ---
            for key, fig in graphs.items():
                # Check if any results were exceptions (returned by cached func)
                has_error = False
                for trace in fig.data:
                    # Check if y-values exist and handle potential errors/NaNs
                    if hasattr(trace, 'y') and trace.y is not None:
                        if any(isinstance(y, Exception) for y in trace.y):
                            has_error = True
                            break
                        if any(y is None or np.isnan(y) for y in trace.y): # Check for None or NaN
                            has_error = True # Treat NaN/None as error for display
                            break
                    else: # Handle cases where y might be missing (unlikely for Scatter but safe)
                        has_error = True
                        break

                if has_error:
                    graph_placeholders[key].error(f"Error generating graph '{key}'. Check logs.", icon="⚠️")
                    status_messages.append(f"ERROR generating graph '{key}'. Possible NaN or Exception in results.")
                    error_occurred = True
                else:
                    graph_placeholders[key].plotly_chart(fig, use_container_width=True)


        except Exception as e:
            error_occurred = True
            tb_str = traceback.format_exc()
            price_placeholder.error(f"Calculation Error: {e}", icon="🚨")
            status_messages.append(f"\n--- ERROR ---")
            status_messages.append(str(e))
            status_messages.append(tb_str)
            # Clear any potentially partially drawn graphs on error
            for key in graph_placeholders:
                 if key not in graphs: # If graph wasn't even generated before error
                    graph_placeholders[key].error(f"Graph '{key}' cancelled due to error.", icon="⚠️")

        finally:
            # --- Final Touches ---
            end_calc_time = time.time()
            calc_duration = end_calc_time - start_calc_time
            status_messages.append(f"\nCalculation finished in {calc_duration:.2f} seconds.")
            log_placeholder.code("\n".join(status_messages))
            if error_occurred:
                log_expander.expanded = True # Auto-expand logs on error
            else:
                st.success("Calculations complete!")


# Add some instructions or info at the bottom
st.divider()
st.markdown(
    """
    **Instructions:**
    1. Configure parameters in the sidebar.
    2. Select the underlying asset model and adjust specific parameters if needed.
    3. Set simulation settings (higher simulations increase accuracy but take longer).
    4. Click "Calculate Price & Update Graphs".
    5. Results and sensitivity graphs will appear in the main area. Check logs for details.

    *Note: Calculations (especially sensitivity graphs) can take time due to Monte Carlo simulations.*
    *Caching is enabled: Re-running with the exact same parameters will be much faster.*
    """
)
