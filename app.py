# DATS 6401: Visualization of Complex Data
# Week 6 Homework: Graph & Network Data
# Author: Soumay Patidar
# Dataset: Formula 1 World Championship
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from pyvis.network import Network
from itertools import combinations

st.set_page_config(page_title="F1 Teammate Network", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Teammate Network")
st.markdown(
    """
    ### What This Network Represents

    This graph shows **F1 driver teammate relationships**.  Two drivers are
    connected by an edge if they raced for the **same constructor in the
    same season**.  The edge weight counts how many seasons they were
    teammates.

    For example, Lewis Hamilton and Nico Rosberg were teammates at Mercedes
    from 2013–2016, so their edge has weight 4.  Drivers who moved between
    many teams (like Daniel Ricciardo or Kimi Raikkonen) become **hubs**
    connecting different parts of the network.
    """
)

st.markdown("---")
st.sidebar.title("Controls")

year_start = st.sidebar.slider("Start year", 1950, 2020, 2010)

layout_choice = st.sidebar.selectbox(
    "Layout algorithm",
    ["force-directed", "circular"],
)

min_degree = st.sidebar.slider(
    "Min connections to show a driver", 1, 10, 1,
)

centrality_choice = st.sidebar.selectbox(
    "Size nodes by",
    ["degree", "betweenness"],
)
@st.cache_data
def load_and_build(start_year):
    """Load F1 data and build the teammate network."""
    results = pd.read_csv("data/results.csv")
    races = pd.read_csv("data/races.csv")
    drivers = pd.read_csv("data/drivers.csv")
    constructors = pd.read_csv("data/constructors.csv")

    df = (
        results
        .merge(races[["raceId", "year"]], on="raceId")
        .merge(drivers[["driverId", "forename", "surname"]], on="driverId")
        .merge(constructors[["constructorId", "name"]], on="constructorId")
    )
    df["driver"] = df["forename"] + " " + df["surname"]
    df = df[df["year"] >= start_year]

    # find teammates: same constructor + same season
    team_groups = df.groupby(["year", "name"])["driver"].apply(set).reset_index()
    team_groups = team_groups[team_groups["driver"].apply(len) >= 2]

    edges = []
    for _, row in team_groups.iterrows():
        for d1, d2 in combinations(sorted(row["driver"]), 2):
            edges.append({"driver1": d1, "driver2": d2})

    edge_df = pd.DataFrame(edges)
    agg = edge_df.groupby(["driver1", "driver2"]).size().reset_index(name="weight")

    G = nx.from_pandas_edgelist(agg, "driver1", "driver2", ["weight"])
    return G


G = load_and_build(year_start)

# filter by minimum degree
nodes_to_keep = [n for n in G.nodes() if G.degree(n) >= min_degree]
G_filtered = G.subgraph(nodes_to_keep).copy()
degrees = dict(G_filtered.degree())

if centrality_choice == "betweenness":
    centrality = nx.betweenness_centrality(G_filtered)
else:
    centrality = {n: d for n, d in degrees.items()}

# community detection
comms = list(greedy_modularity_communities(G_filtered))
community_map = {}
for cid, comm in enumerate(comms):
    for node in comm:
        community_map[node] = cid
c1, c2, c3 = st.columns(3)
c1.metric("Drivers (nodes)", G_filtered.number_of_nodes())
c2.metric("Teammate links (edges)", G_filtered.number_of_edges())
c3.metric("Communities detected", len(comms))

palette = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabebe", "#469990",
    "#e6beff", "#9A6324", "#800000", "#aaffc3",
]

net = Network(height="600px", width="100%", cdn_resources="in_line")

# scale centrality for node size
max_cent = max(centrality.values()) if centrality else 1
for n in G_filtered.nodes():
    size = 10 + (centrality[n] / max_cent) * 40
    color = palette[community_map.get(n, 0) % len(palette)]
    net.add_node(
        n, label=n.split()[-1],  # last name only for readability
        size=size,
        color=color,
        title=f"{n}\nDegree: {degrees[n]}\nCommunity: {community_map.get(n, '?')}",
    )

for u, v, d in G_filtered.edges(data=True):
    net.add_edge(u, v, value=d.get("weight", 1), color="#d5d5d5")

# apply layout
if layout_choice == "circular":
    pos = nx.circular_layout(G_filtered, scale=400)
    for node in net.nodes:
        node["x"], node["y"] = map(float, pos[node["id"]])
        node["physics"] = False
    net.toggle_physics(False)

# render
html = net.generate_html(notebook=False)
components.html(html, height=620)

st.caption(
    "Node size = " + centrality_choice + " centrality · "
    "Node color = detected community · "
    "Hover for details · Drag to rearrange"
)
st.markdown("---")
st.markdown("### Why These Encodings")
st.markdown(
    f"""
    **Size = {centrality_choice} centrality:** {"Degree centrality counts how many different teammates a driver has had. Drivers with high degree (like Daniel Ricciardo or Nico Hulkenberg) drove for many teams, making them hubs in the network." if centrality_choice == "degree" else "Betweenness centrality measures how often a driver lies on the shortest path between two other drivers. High betweenness means the driver is a bridge connecting otherwise separate clusters, often drivers who moved between rival teams."}

    **Color = community (greedy modularity):** The community detection
    algorithm groups drivers who share more internal connections than
    expected by chance.  In this network, communities roughly correspond
    to **team families**, groups of drivers who circulated among the same
    set of constructors.  Color makes these clusters pop out
    preattentively (we detect hue differences in under 200ms).

    **Edge thickness = seasons as teammates:** Thicker edges mean the pair
    raced together for more years, highlighting stable partnerships like
    Hamilton-Rosberg or Vettel-Webber.
    """
)

st.markdown("### Limitation")
st.markdown(
    """
    The main limitation is the **hairball problem**.  When the start year is
    set too early (e.g., 1950), the graph has 800+ nodes and thousands of
    edges, it becomes an unreadable tangle where no structure is visible.
    The year slider and degree filter partially mitigate this, but for the
    full historical network, an **adjacency matrix** or **aggregated view**
    (grouping by constructor instead of individual driver) would
    communicate the structure more honestly than a node-link diagram.

    A second limitation is that **layout position has no inherent meaning**.
    The force-directed algorithm places connected nodes closer together,
    but the x/y coordinates don't map to any data variable.  A viewer
    might mistakenly read spatial proximity as similarity beyond just
    "these drivers share teammates," which is the only thing the layout
    encodes.
    """
)
st.markdown("---")