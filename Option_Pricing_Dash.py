# %% Imports
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback # Using current Dash import syntax
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson
import time
import traceback # For detailed error messages

# ==============================================================================
# %% Block 1: Option Valuation Class Definition
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
        self.dividend_yield = dividend_yield
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
                 # Suppress warning in app context or log it
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
            print(f"Warning: GARCH params omega={self.garch_omega}, alpha={self.garch_alpha}, beta={self.garch_beta} may be invalid.")
        # if self.garch_alpha + self.garch_beta >= 1:
        #     print("Warning: GARCH process may not be stationary (alpha + beta >= 1).")

        Z, = self._generate_random_factors(1)
        S = np.zeros((self.time_steps + 1, self.sim_number))
        V = np.zeros((self.time_steps + 1, self.sim_number)) # Variance path
        S[0] = self.S0
        V[0] = self.Vol**2 # Use Vol as initial variance

        for i in range(1, self.time_steps + 1):
            current_var = V[i-1]
            current_var = np.maximum(current_var, 1e-9) # Ensure positive variance
            current_sigma = np.sqrt(current_var)

            drift_term = (self.rf - self.dividend_yield - 0.5 * current_var) * self.dt
            diffusion_term = current_sigma * np.sqrt(self.dt) * Z.iloc[i, :]
            log_return = drift_term + diffusion_term
            S[i] = S[i-1] * np.exp(log_return)
            S[i] = np.maximum(S[i], 0) # Floor at zero

            # Update variance for next step using squared log return
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
            print("Warning: Non-finite values found in price paths. Replacing with S0.")
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
                             # print(f"Warning: Regression failed at step {t}: {fit_err}")
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

print("Block 1 executed: option_valuation class defined.")
# ==============================================================================
# %% Block 2: App Initialization and Global Settings
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
server = app.server # For potential deployment integration

# --- Global App Settings ---
# Reduced simulations/points for app responsiveness
simulation_runs_app = 1000 # Default sims for main price calc
sim_plot = 500          # Sims used for graph calculations (KEEP LOW!)
num_points_plot_app = 15 # Points per sensitivity graph (KEEP LOW!)

print("Block 2 executed: App initialized.")
# ==============================================================================
# %% Block 3: App Layout Definition
# ==============================================================================

# --- Reusable Input Group Function ---
def make_input_group(label_text, input_id, input_type='number', input_value=None, input_step=None, input_min=None, input_max=None, width=7):
    return dbc.Row([
        dbc.Label(label_text, html_for=input_id, width=12-width),
        dbc.Col(
            dcc.Input(id=input_id, type=input_type, value=input_value, step=input_step, min=input_min, max=input_max, className="form-control form-control-sm") # Smaller form control
        , width=width)
    ], className="mb-2 align-items-center")

# --- App Layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Monte Carlo Option Pricer & Analyzer"), width=12), className="mb-4 mt-2 text-center"),
    dbc.Row([
        # --- INPUT COLUMN ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Configuration", className="card-title")),
                dbc.CardBody([
                    # --- Option Parameters ---
                    html.H6("Option Parameters", className="card-subtitle mb-2 text-muted"),
                    make_input_group("Spot (S0):", 'input-s0', input_value=100, input_step='any', input_min=0.01),
                    make_input_group("Strike (E):", 'input-e', input_value=100, input_step='any', input_min=0.01),
                    make_input_group("Maturity (Tt, yrs):", 'input-tt', input_value=1.0, input_step='any', input_min=0.01),
                    make_input_group("Volatility (Vol):", 'input-vol', input_value=0.20, input_step='any', input_min=0.0),
                    make_input_group("Risk-Free Rate (rf):", 'input-rf', input_value=0.05, input_step='any', input_min=0.0),
                    make_input_group("Dividend Yield (q):", 'input-div', input_value=0.01, input_step='any', input_min=0.0),

                    dbc.Row([
                        dbc.Label("Type:", html_for='input-callput', width=5),
                        dbc.Col(dcc.Dropdown(id='input-callput', options=[{'label': 'Call', 'value': 1}, {'label': 'Put', 'value': -1}], value=1), width=7)
                    ], className="mb-2 align-items-center"),
                    dbc.Row([
                         dbc.Label("Style:", html_for='input-style', width=5),
                         dbc.Col(dcc.Dropdown(id='input-style', options=[{'label': 'European', 'value': 'European'}, {'label': 'American', 'value': 'American'}], value='European'), width=7)
                    ], className="mb-2 align-items-center"),

                    html.Hr(),
                    # --- Model Selection ---
                    html.H6("Underlying Model", className="card-subtitle mb-2 text-muted"),
                     dbc.Row([
                         dbc.Label("Model:", html_for='input-model', width=5),
                         dbc.Col(dcc.Dropdown(id='input-model', options=[
                             {'label': 'GBM', 'value': 'GBM'},
                             {'label': 'Jump Diffusion', 'value': 'JumpDiffusion'},
                             {'label': 'GARCH(1,1)', 'value': 'GARCH'}], value='GBM'), width=7)
                    ], className="mb-2 align-items-center"),

                    # --- Conditional Model Parameters ---
                    html.Div(id='jump-params-div', children=[
                        html.H6("Jump Diffusion Params:", className="text-muted mt-2"),
                        make_input_group("Lambda (λ):", 'input-jlambda', input_value=0.5, input_step='any', input_min=0),
                        make_input_group("Mean (μ):", 'input-jmu', input_value=-0.05, input_step='any'),
                        make_input_group("Sigma (σj):", 'input-jsigma', input_value=0.25, input_step='any', input_min=0.001),
                    ], style={'display': 'none', 'border':'1px solid #eee', 'padding':'10px', 'border-radius':'5px', 'margin-top':'10px'}), # Initially hidden

                    html.Div(id='garch-params-div', children=[
                         html.H6("GARCH(1,1) Params:", className="text-muted mt-2"),
                         make_input_group("Omega (ω):", 'input-gomega', input_value='1e-6', input_step='any', input_min=0, input_type='text'), # Text for sci notation
                         make_input_group("Alpha (α):", 'input-galpha', input_value=0.09, input_step='any', input_min=0),
                         make_input_group("Beta (β):", 'input-gbeta', input_value=0.90, input_step='any', input_min=0),
                    ], style={'display': 'none', 'border':'1px solid #eee', 'padding':'10px', 'border-radius':'5px', 'margin-top':'10px'}), # Initially hidden

                    html.Hr(),
                    # --- Simulation Settings ---
                    html.H6("Simulation Settings", className="card-subtitle mb-2 text-muted"),
                    make_input_group("Simulations (Price):", 'input-simnum', input_value=simulation_runs_app, input_step='any', input_min=100),
                    make_input_group("Time Steps/Year:", 'input-ts-year', input_value=50, input_step='any', input_min=1),
                    make_input_group(f"Sims per Graph Pt:", 'input-sim-plot', input_value=sim_plot, input_step='any', input_min=50),
                    make_input_group(f"Points per Graph:", 'input-points-plot', input_value=num_points_plot_app, input_step='any', input_min=5, input_max=50),

                    html.Hr(),
                    # --- Action Button and Output ---
                    dbc.Button("Calculate Price & Update Graphs", id='calculate-button', n_clicks=0, color="primary", className="w-100 mb-3 fw-bold"),
                    dbc.Alert("Price results will appear here.", id='output-price', color="info", className="text-center fw-bold fs-5"),
                    dbc.Collapse( # Collapsible section for status/logs
                         dbc.Card(dbc.CardBody(html.Pre(id='calc-status', style={'max-height':'150px', 'overflow-y':'scroll', 'font-size':'0.8em'}))),
                         id="collapse-status", is_open=False,
                    ),
                    dbc.Button("Show/Hide Logs", id="collapse-button", color="secondary", size="sm", className="mt-1 w-100"),

                ]) # End CardBody
            ]), # End Card
        ], md=4), # End Input Column

        # --- GRAPH COLUMN ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Sensitivity Analysis", className="card-title")),
                dbc.CardBody([
                    dcc.Loading( # Wrap all graphs in a single Loading component
                        type="default",
                        children=[
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-s0'), md=12)]),
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-e'), md=12)]),
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-tt'), md=12)]),
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-vol'), md=12)]),
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-rf'), md=12)]),
                             dbc.Row([dbc.Col(dcc.Graph(id='graph-div'), md=12)]),
                             # --- REMOVED GRAPH 7 (graph-compare) ---
                        ]
                    ) # End Loading
                ]) # End CardBody
            ]) # End Card
        ], md=8) # End Graph Column
    ]) # End Main Row
], fluid=True)

print("Block 3 executed: App layout defined.")
# ==============================================================================
# %% Block 4: App Callbacks
# ==============================================================================

# --- Callback to toggle log visibility ---
@callback(
    Output("collapse-status", "is_open"),
    Input("collapse-button", "n_clicks"),
    State("collapse-status", "is_open"),
    prevent_initial_call=True,
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# --- Callback to show/hide model-specific parameters ---
@callback(
    Output('jump-params-div', 'style'),
    Output('garch-params-div', 'style'),
    Input('input-model', 'value')
)
def toggle_model_params(selected_model):
    """Shows/hides the input sections for Jump or GARCH parameters."""
    if selected_model == 'JumpDiffusion':
        return {'display': 'block', 'border':'1px solid #eee', 'padding':'10px', 'border-radius':'5px', 'margin-top':'10px'}, {'display': 'none'}
    elif selected_model == 'GARCH':
        return {'display': 'none'}, {'display': 'block', 'border':'1px solid #eee', 'padding':'10px', 'border-radius':'5px', 'margin-top':'10px'}
    else: # GBM
        return {'display': 'none'}, {'display': 'none'}

# --- Main callback to calculate price and update all graphs ---
@callback(
    # Outputs (Removed graph-compare)
    Output('output-price', 'children'),
    Output('output-price', 'color'),
    Output('calc-status', 'children'),
    Output('graph-s0', 'figure'),
    Output('graph-e', 'figure'),
    Output('graph-tt', 'figure'),
    Output('graph-vol', 'figure'),
    Output('graph-rf', 'figure'),
    Output('graph-div', 'figure'),

    # Trigger
    Input('calculate-button', 'n_clicks'),

    # State Inputs (Get current values when button is clicked)
    State('input-s0', 'value'), State('input-e', 'value'), State('input-tt', 'value'),
    State('input-vol', 'value'), State('input-rf', 'value'), State('input-div', 'value'),
    State('input-callput', 'value'), State('input-style', 'value'),
    State('input-model', 'value'),
    State('input-simnum', 'value'), State('input-ts-year', 'value'),
    State('input-sim-plot', 'value'), State('input-points-plot', 'value'),
    # Model Params State
    State('input-jlambda', 'value'), State('input-jmu', 'value'), State('input-jsigma', 'value'),
    State('input-gomega', 'value'), State('input-galpha', 'value'), State('input-gbeta', 'value'),
    prevent_initial_call=True # Don't run when app loads
)
def update_price_and_graphs(
    n_clicks, s0, e, tt, vol, rf, div, cp, style, model, simnum, ts_year,
    sim_plot_val, points_plot_val,
    jlambda, jmu, jsigma, gomega_str, galpha, gbeta):
    """
    This function runs when the button is clicked. It calculates the option price
    and generates data for all sensitivity graphs, then updates the UI.
    """
    start_calc_time = time.time()
    status_messages = []
    error_occurred = False

    # --- Helper Function for Plotting ---
    def create_empty_figure(title="Error generating graph"):
        fig = go.Figure()
        fig.update_layout(
            title=title, height=300, margin=dict(t=40, b=20, l=30, r=30),
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[{'text': title, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]
        )
        return fig

    # --- Create default empty figures for all graphs ---
    fig_s0 = create_empty_figure("Sensitivity to Spot (Pending Calculation)")
    fig_e = create_empty_figure("Sensitivity to Strike (Pending Calculation)")
    fig_tt = create_empty_figure("Sensitivity to Maturity (Pending Calculation)")
    fig_vol = create_empty_figure("Sensitivity to Volatility (Pending Calculation)")
    fig_rf = create_empty_figure("Sensitivity to Rate (Pending Calculation)")
    fig_div = create_empty_figure("Sensitivity to Dividend (Pending Calculation)")
    # Removed: fig_compare

    # --- Input Validation ---
    if n_clicks == 0: # Failsafe
        return "Click button to calculate.", "info", "", fig_s0, fig_e, fig_tt, fig_vol, fig_rf, fig_div

    try:
        core_params = [s0, e, tt, vol, rf, div, cp, style, model, simnum, ts_year, sim_plot_val, points_plot_val]
        if None in core_params: raise ValueError("Core input parameters cannot be empty.")
        s0, e, tt, vol, rf, div = float(s0), float(e), float(tt), float(vol), float(rf), float(div)
        simnum, ts_year = int(simnum), int(ts_year)
        sim_plot_val, points_plot_val = int(sim_plot_val), int(points_plot_val)
        if tt <= 0 or vol < 0 or simnum <= 0 or ts_year <= 0 or sim_plot_val <=0 or points_plot_val <= 0:
             raise ValueError("Tt, Vol, Simulations, Time Steps, Plot Sims/Points must be positive.")
        if e <= 0: raise ValueError("Strike (E) must be positive.")
        if s0 <= 0: raise ValueError("Spot (S0) must be positive.")
        base_dt = 1.0 / ts_year
    except (ValueError, TypeError) as verr:
         status_messages.append(f"Input Error: {verr}")
         # Return empty figures for all graphs
         return "Input Error", "danger", html.Pre("\n".join(status_messages)), fig_s0, fig_e, fig_tt, fig_vol, fig_rf, fig_div


    # --- Parse Model Specific Args ---
    model_args = {}
    try:
        if model == 'JumpDiffusion':
            if None in [jlambda, jmu, jsigma]: raise ValueError("Jump parameters cannot be empty.")
            model_args = {'jump_lambda': float(jlambda), 'jump_mu': float(jmu), 'jump_sigma': float(jsigma)}
            if model_args['jump_sigma'] <= 0: raise ValueError("Jump Sigma must be positive.")
        elif model == 'GARCH':
            if None in [gomega_str, galpha, gbeta]: raise ValueError("GARCH parameters cannot be empty.")
            model_args = {'garch_omega': float(gomega_str), 'garch_alpha': float(galpha), 'garch_beta': float(gbeta)}
            if model_args['garch_omega'] < 0 or model_args['garch_alpha'] < 0 or model_args['garch_beta'] < 0:
                 print("Warning: Negative GARCH parameters provided.")
    except (ValueError, TypeError) as model_err:
        status_messages.append(f"Model Parameter Error: {model_err}")
        # Return empty figures for all graphs
        return "Model Parameter Error", "danger", html.Pre("\n".join(status_messages)), fig_s0, fig_e, fig_tt, fig_vol, fig_rf, fig_div


    # --- 1. Calculate Single Price ---
    price_output = "Error calculating price."
    price_color = "danger"
    option_value_base = np.nan
    try:
        status_messages.append(f"Calculating base price ({style} {model} {'Call' if cp==1 else 'Put'})...")
        option_base = option_valuation(
            S0=s0, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, call_put=cp, sim_number=simnum,
            option_style=style, dividend_yield=div, model_type=model, **model_args
        )
        option_value_base = option_base.option_value
        if np.isnan(option_value_base): raise ValueError("Calculation returned NaN.")
        price_output = f"{option_value_base:.4f}"
        price_color = "success"
        status_messages.append(f"-> Base price calculated: {option_value_base:.4f}")
    except Exception as err:
        error_occurred = True
        tb_str = traceback.format_exc()
        price_output = f"Pricing Error"
        status_messages.append(f"ERROR calculating base price: {err}\n{tb_str}")


    # --- 2. Calculate Data for Graphs ---
    status_messages.append(f"\nGenerating sensitivity graphs (Sims/Pt={sim_plot_val}, Pts={points_plot_val})...")
    sim_plot_actual = max(50, int(sim_plot_val))
    num_points_actual = max(5, int(points_plot_val))

    # --- CORRECTED common_plot_args definition ---
    # Include only arguments that are CONSTANT across ALL graph loops below
    common_plot_args_graphs = {'sim_number': sim_plot_actual,
                             'option_style': style,
                             'model_type': model,
                             **model_args}

    # --- S0 Graph ---
    try:
        status_messages.append(" Calculating S0 sensitivity...")
        s0_range_plot = np.linspace(max(0.1, s0 * 0.7), s0 * 1.3, num_points_actual)
        # Explicitly pass all necessary option params, varying S0
        prices_s0_call = [option_valuation(S0=s_val, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div, call_put=1, **common_plot_args_graphs).option_value for s_val in s0_range_plot]
        prices_s0_put = [option_valuation(S0=s_val, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div, call_put=-1, **common_plot_args_graphs).option_value for s_val in s0_range_plot]

        fig_s0 = go.Figure()
        fig_s0.add_trace(go.Scatter(x=s0_range_plot, y=prices_s0_call, mode='lines+markers', name=f'{style} Call'))
        fig_s0.add_trace(go.Scatter(x=s0_range_plot, y=prices_s0_put, mode='lines+markers', name=f'{style} Put'))
        fig_s0.add_vline(x=e, line_dash="dash", line_color="grey", annotation_text=f"E={e:.2f}")
        fig_s0.update_layout(title=f"vs Spot Price ({model})", xaxis_title="Spot Price (S0)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: S0 Graph: {graph_err}"); fig_s0 = create_empty_figure(f"Error")

    # --- E Graph ---
    try:
        status_messages.append(" Calculating E sensitivity...")
        e_range_plot = np.linspace(max(0.1, e * 0.7), e * 1.3, num_points_actual)
        # Explicitly pass all necessary option params, varying E
        prices_e_call = [option_valuation(S0=s0, E=e_val, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div, call_put=1, **common_plot_args_graphs).option_value for e_val in e_range_plot]
        prices_e_put = [option_valuation(S0=s0, E=e_val, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div, call_put=-1, **common_plot_args_graphs).option_value for e_val in e_range_plot]

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=e_range_plot, y=prices_e_call, mode='lines+markers', name=f'{style} Call'))
        fig_e.add_trace(go.Scatter(x=e_range_plot, y=prices_e_put, mode='lines+markers', name=f'{style} Put'))
        fig_e.add_vline(x=s0, line_dash="dash", line_color="grey", annotation_text=f"S0={s0:.2f}")
        fig_e.update_layout(title=f"vs Strike Price ({model})", xaxis_title="Strike Price (E)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: E Graph: {graph_err}"); fig_e = create_empty_figure(f"Error")

    # --- Tt Graph ---
    try:
        status_messages.append(" Calculating Tt sensitivity...")
        min_tt = max(base_dt * 5, 0.02)
        tt_range_plot = np.linspace(min_tt, tt * 1.5 + min_tt, num_points_actual)
        prices_tt_call = []
        prices_tt_put = []
        for tt_val in tt_range_plot:
            # Calculate dt specific to this tt_val for accuracy at short Tt
            current_dt = min(base_dt, tt_val / 10 if tt_val > 0 else base_dt) # Ensure dt <= Tt/10 roughly
            # Explicitly pass all necessary option params, varying Tt and dt
            call_opt = option_valuation(S0=s0, E=e, Tt=tt_val, Vol=vol, rf=rf, dt=current_dt, dividend_yield=div, call_put=1, **common_plot_args_graphs)
            put_opt = option_valuation(S0=s0, E=e, Tt=tt_val, Vol=vol, rf=rf, dt=current_dt, dividend_yield=div, call_put=-1, **common_plot_args_graphs)
            prices_tt_call.append(call_opt.option_value)
            prices_tt_put.append(put_opt.option_value)

        fig_tt = go.Figure()
        fig_tt.add_trace(go.Scatter(x=tt_range_plot, y=prices_tt_call, mode='lines+markers', name=f'{style} Call'))
        fig_tt.add_trace(go.Scatter(x=tt_range_plot, y=prices_tt_put, mode='lines+markers', name=f'{style} Put'))
        fig_tt.update_layout(title=f"vs Maturity ({model})", xaxis_title="Time to Maturity (Tt, years)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: Tt Graph: {graph_err}"); fig_tt = create_empty_figure(f"Error")

    # --- Vol Graph ---
    try:
        status_messages.append(" Calculating Vol sensitivity...")
        vol_range_plot = np.linspace(max(0.01, vol * 0.2), vol * 2.0 + 0.01, num_points_actual)
        # Explicitly pass all necessary option params, varying Vol
        prices_vol_call = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol_val, rf=rf, dt=base_dt, dividend_yield=div, call_put=1, **common_plot_args_graphs).option_value for vol_val in vol_range_plot]
        prices_vol_put = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol_val, rf=rf, dt=base_dt, dividend_yield=div, call_put=-1, **common_plot_args_graphs).option_value for vol_val in vol_range_plot]

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=vol_range_plot, y=prices_vol_call, mode='lines+markers', name=f'{style} Call'))
        fig_vol.add_trace(go.Scatter(x=vol_range_plot, y=prices_vol_put, mode='lines+markers', name=f'{style} Put'))
        fig_vol.update_layout(title=f"vs Volatility ({model})", xaxis_title="Volatility (Vol)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: Vol Graph: {graph_err}"); fig_vol = create_empty_figure(f"Error")

    # --- rf Graph ---
    try:
        status_messages.append(" Calculating rf sensitivity...")
        rf_range_plot = np.linspace(max(0.0, rf - 0.04), rf + 0.05, num_points_actual)
        # Explicitly pass all necessary option params, varying rf
        prices_rf_call = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol, rf=rf_val, dt=base_dt, dividend_yield=div, call_put=1, **common_plot_args_graphs).option_value for rf_val in rf_range_plot]
        prices_rf_put = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol, rf=rf_val, dt=base_dt, dividend_yield=div, call_put=-1, **common_plot_args_graphs).option_value for rf_val in rf_range_plot]

        fig_rf = go.Figure()
        fig_rf.add_trace(go.Scatter(x=rf_range_plot, y=prices_rf_call, mode='lines+markers', name=f'{style} Call'))
        fig_rf.add_trace(go.Scatter(x=rf_range_plot, y=prices_rf_put, mode='lines+markers', name=f'{style} Put'))
        fig_rf.update_layout(title=f"vs Risk-Free Rate ({model})", xaxis_title="Risk-Free Rate (rf)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: rf Graph: {graph_err}"); fig_rf = create_empty_figure(f"Error")

    # --- div Graph ---
    try:
        status_messages.append(" Calculating Dividend sensitivity...")
        div_range_plot = np.linspace(0.0, max(0.1, div * 2.0 + 0.02) , num_points_actual)
        # Explicitly pass all necessary option params, varying dividend_yield
        prices_div_call = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div_val, call_put=1, **common_plot_args_graphs).option_value for div_val in div_range_plot]
        prices_div_put = [option_valuation(S0=s0, E=e, Tt=tt, Vol=vol, rf=rf, dt=base_dt, dividend_yield=div_val, call_put=-1, **common_plot_args_graphs).option_value for div_val in div_range_plot]

        fig_div = go.Figure()
        fig_div.add_trace(go.Scatter(x=div_range_plot, y=prices_div_call, mode='lines+markers', name=f'{style} Call'))
        fig_div.add_trace(go.Scatter(x=div_range_plot, y=prices_div_put, mode='lines+markers', name=f'{style} Put'))
        fig_div.update_layout(title=f"vs Dividend Yield ({model})", xaxis_title="Dividend Yield (q)", yaxis_title="Option Price", height=300, margin=dict(t=40, b=20, l=30, r=30), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    except Exception as graph_err: error_occurred = True; status_messages.append(f" ERROR: Div Graph: {graph_err}"); fig_div = create_empty_figure(f"Error")

    # --- REMOVED: Calculation for Comparison Graph ---

    # --- Final Touches ---
    end_calc_time = time.time()
    calc_duration = end_calc_time - start_calc_time
    status_messages.append(f"\nCalculation finished in {calc_duration:.2f} seconds.")
    status_html = html.Pre("\n".join(status_messages)) # Use Pre for multi-line status

    # Determine final price color based on errors during graph generation too
    if error_occurred and price_color != "danger":
        price_color = "warning"
        if "Error" not in price_output: # Avoid duplicating Error prefix
             price_output += " (Graph errors)"

    # Return statement updated (Removed fig_compare)
    return price_output, price_color, status_html, fig_s0, fig_e, fig_tt, fig_vol, fig_rf, fig_div


# ==============================================================================
# %% Block 5: Run the App Server
# ==============================================================================
if __name__ == '__main__':
    print("Block 5: Starting Dash server...")
    print(f"Access the app at http://127.0.0.1:8050/")
    # Use app.run() instead of app.run_server() for newer Dash versions
    app.run(debug=True, port=8050) # Set debug=False for production / deployment