import streamlit as st
import networkx as nx
import random
import plotly.graph_objects as go

st.set_page_config(page_title="Network SIR Simulation 3D", layout="wide")
st.title("Interactive Network‑based SIR Model (3D Visualization)")

# network & epidemic parameters
st.sidebar.header("Network Parameters")
num_nodes = st.sidebar.slider("Number of nodes", 50, 500, 200, step=50)
edge_prob = st.sidebar.slider("Edge probability", 0.01, 0.2, 0.05, step=0.01)

st.sidebar.header("Epidemic Parameters")
beta = st.sidebar.slider("Infection probability per contact (β)", 0.0, 1.0, 0.3, step=0.05)
gamma = st.sidebar.slider("Recovery probability per step (γ)", 0.0, 1.0, 0.1, step=0.05)
init_inf_frac = st.sidebar.slider("Initial infected fraction", 0.0, 0.2, 0.02, step=0.01)
steps = st.sidebar.slider("Simulation steps", 1, 50, 55, step=1)

def initialize_network(n, p, init_frac):
    G = nx.erdos_renyi_graph(n, p)
    status = {node: 'S' for node in G.nodes()}
    infected = random.sample(list(G.nodes()), max(1, int(init_frac * n)))
    for node in infected:
        status[node] = 'I'
    return G, status

# discrete-time SIR simulation (deterministic update)
def run_simulation(G, status0, beta, gamma, max_steps):
    history = []
    curr = status0.copy()
    for t in range((max_steps + 1)*5):
        history.append(curr.copy())
        new = curr.copy()
        for node in G.nodes():
            if curr[node] == 'I':
                # recovery
                if random.random() < gamma:
                    new[node] = 'R'
                else:
                    # infection
                    for nbr in G.neighbors(node):
                        if curr[nbr] == 'S' and random.random() < beta:
                            new[nbr] = 'I'
        curr = new
    return history

# build network & simulate
G, init_status = initialize_network(num_nodes, edge_prob, init_inf_frac)
history = run_simulation(G, init_status, beta, gamma, steps)

time_idx = st.sidebar.slider("Time step", 0, steps, 0)
status = history[time_idx]

# compute 3D layout
pos = nx.spring_layout(G, dim=3, seed=42)
x_nodes = [pos[n][0] for n in G.nodes()]
y_nodes = [pos[n][1] for n in G.nodes()]
z_nodes = [pos[n][2] for n in G.nodes()]
colors = ["#1f77b4" if status[n]=='S' else "#d62728" if status[n]=='I' else "#2ca02c" for n in G.nodes()]

# build edge traces
edge_x, edge_y, edge_z = [], [], []
for u, v in G.edges():
    x0, y0, z0 = pos[u]
    x1, y1, z1 = pos[v]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]
    edge_z += [z0, z1, None]

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z,
    mode='lines', line=dict(color='gray', width=1), hoverinfo='none'
)
node_trace = go.Scatter3d(
    x=x_nodes, y=y_nodes, z=z_nodes,
    mode='markers', marker=dict(size=8, color=colors),
    text=[str(n) for n in G.nodes()], hoverinfo='text'
)

fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    width=1000, height=800,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    )
)

st.plotly_chart(fig, use_container_width=True)

# stats
totals = {s: sum(1 for v in status.values() if v==s) for s in ['S','I','R']}
st.write(f"**Step {time_idx}:** S={totals['S']}, I={totals['I']}, R={totals['R']}")
st.markdown("*Rotate the 3D view by dragging; zoom with scroll; adjust sliders to rerun simulation.*")
