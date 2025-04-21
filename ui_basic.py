import numpy as np
from scipy.integrate import solve_ivp
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="COVID SIR model UI", layout="wide")
st.title("Interactive COVID-19 SIR Model Simulation")

# parameters, explained in the presentation ppt
st.sidebar.header("Model Parameters")
base_beta = st.sidebar.slider("Base transmission rate (β)", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
gamma = st.sidebar.slider("Recovery rate (γ)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
eff = st.sidebar.slider("Vaccine efficacy", min_value=0.0, max_value=1.0, value=0.8, step=0.05)
temp = st.sidebar.slider("Ambient temperature (°C)", min_value=-20.0, max_value=40.0, value=0.0, step=1.0)
mob = st.sidebar.slider("Mobility index (× baseline)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
T0 = st.sidebar.number_input("Reference temperature T0 (°C)", value=10.0)
alpha = st.sidebar.number_input("Temperature sensitivity α", value=0.05)

st.sidebar.header("Population & Initial Conditions")
N = st.sidebar.number_input("Total population", value=1_000_000, step=10000)
vac_frac = st.sidebar.slider("Initial vaccinated fraction", min_value=0.0, max_value=1.0, value=0.3)
I0 = st.sidebar.number_input("Initial infected (I0)", value=1000, step=100)

st.sidebar.header("Simulation Settings")
days = st.sidebar.slider("Simulation days", min_value=1, max_value=365, value=180)

# helper functions
def virus_survival_factor(temperature_c, T0, alpha):
    return 1 + alpha * max(0, T0 - temperature_c)

def mobility_factor(mobility_index):
    return mobility_index

def effective_beta(base_beta, temp, mob, T0, alpha):
    return base_beta * virus_survival_factor(temp, T0, alpha) * mobility_factor(mob)

def sirv_ode(t, y, params):
    Su, Sv, I, R = y
    N = Su + Sv + I + R
    beta = effective_beta(
        params['base_beta'], params['temperature_c'], params['mobility_index'],
        params['T0'], params['alpha']
    )
    lambda_u = beta * I / N
    lambda_v = lambda_u * (1 - params['vaccine_efficacy'])
    dSu = - lambda_u * Su
    dSv = - lambda_v * Sv
    dI  =   lambda_u * Su + lambda_v * Sv - params['gamma'] * I
    dR  =   params['gamma'] * I
    return [dSu, dSv, dI, dR]

# initial state
y0 = [(N - I0) * (1 - vac_frac), (N - I0) * vac_frac, I0, 0.0]
params = dict(
    base_beta=base_beta,
    gamma=gamma,
    vaccine_efficacy=eff,
    temperature_c=temp,
    mobility_index=mob,
    T0=T0,
    alpha=alpha
)

t_eval = np.linspace(0, days, days + 1)

# solve the ODE system
sol = solve_ivp(
    lambda t, y: sirv_ode(t, y, params),
    [0, days], y0, t_eval=t_eval, rtol=1e-6
)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(sol.t, sol.y[2], label='Infected')
ax.plot(sol.t, sol.y[0] + sol.y[1], label='Susceptible')
ax.plot(sol.t, sol.y[3], label='Recovered')
ax.plot(sol.t, sol.y[2] + sol.y[3], label='Total Infected')  # Cumulative infected = I + R
ax.set_xlabel('Days')
ax.set_ylabel('Population')
ax.set_title('SIR-V Model Dynamics')
ax.legend()
ax.grid(True)

st.pyplot(fig)
st.markdown("---")
st.markdown(
    "**Notes:** Tweak sliders to see in real time how temperature, mobility, and vaccination change the outbreak dynamics. "
    "Now includes a cumulative 'total infected' curve (I + R)."
)
