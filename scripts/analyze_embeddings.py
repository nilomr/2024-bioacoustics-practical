import json
from pathlib import Path

import cuml
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy

rootdir = Path(__file__).parent.parent
detections_file = Path(rootdir, "data", "derived", "detections.json")
embeddings_file = Path(rootdir, "data", "derived", "embeddings.json")
figs_dir = Path(rootdir, "figures", "embeddings")
figs_dir.mkdir(parents=True, exist_ok=True)

# ──── PLOT SETTINGS ──────────────────────────────────────────────────────────

# spines should be white
plt.rcParams.update(
    {
        "axes.facecolor": "#1d1d1d",
        "figure.facecolor": "#1d1d1d",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "axes.edgecolor": "white",
        "axes.titlecolor": "white",
        "axes.titlepad": 15,
    }
)

site_palette = plt.get_cmap("Spectral", 7)
cluster_palette = ["#ed6a5a", "#f4f1bb", "#9bc1bc"]
cluster_palette_4 = ["#335c67", "#fff3b0", "#e09f3e", "#9e2a2b"]

figwidth = 8
textsize = 15

# ──── DATA INGEST ────────────────────────────────────────────────────────────


# Read the detections from the json file and convert to a pandas DataFrame
with open(detections_file, encoding="utf-8") as file:
    detections = json.load(file)

with open(embeddings_file, encoding="utf-8") as file:
    embeddings = json.load(file)

detections = [
    {
        "file_name": d[0],
        "site": d[1],
        "common_name": e["common_name"],
        "scientific_name": e["scientific_name"],
        "start_time": e["start_time"],
        "end_time": e["end_time"],
        "confidence": e["confidence"],
        "label": e["label"],
    }
    for d in detections
    for e in d[2]
]

embeddings = [
    {
        "file_name": e[0],
        "site": e[1],
        "start_time": f["start_time"],
        "end_time": f["end_time"],
        "embeddings": f["embeddings"],
    }
    for e in embeddings
    for f in e[2]
]

# Add detection information to the embeddings based on file name, site,
# and start and end times

df_detections = pd.DataFrame(detections)
df_embeddings = pd.DataFrame(embeddings)

df = pd.merge(
    df_embeddings,
    df_detections,
    on=["file_name", "site", "start_time", "end_time"],
    how="left",
)
# Change nans in common_name to "Unknown"
df["common_name"] = df["common_name"].fillna("Unknown")

# from the file_name column create a datatime column and a time column
df["datetime"] = pd.to_datetime(df["file_name"], format="%Y%m%d_%H%M%S")
df["time"] = df["datetime"].dt.time

# Project all embeddings into 2D using UMAP
X = np.array(df["embeddings"].tolist())
umap_embedding = cuml.UMAP(n_neighbors=30, min_dist=0.1, n_components=3).fit_transform(
    X
)

df.loc[:, "umap_x"] = umap_embedding[:, 0]
df.loc[:, "umap_y"] = umap_embedding[:, 1]

# Plot the embedding without colors (all points #eaeaea)
plt.figure(figsize=(figwidth, figwidth))
plt.scatter(df["umap_x"], df["umap_y"], color="#eaeaea", s=0.5, alpha=0.2)
plt.title("All embeddings")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(1.3)
plt.savefig(Path(figs_dir, "embeddings.png"), bbox_inches="tight")


# Plot the embeddings colored by species
plt.figure(figsize=(figwidth, figwidth))
palette = plt.get_cmap(
    "tab20b", len(df[df["common_name"] != "Unknown"]["common_name"].unique())
)
plt.scatter(
    df[df["common_name"] == "Unknown"]["umap_x"],
    df[df["common_name"] == "Unknown"]["umap_y"],
    label="Unknown",
    alpha=0.5,
    color="#eaeaea",
    s=0.1,
)
for i, (species, group) in enumerate(
    df[df["common_name"] != "Unknown"].groupby("common_name")
):
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=species,
        alpha=0.5,
        s=1.5,
        color=palette(i),
    )
plt.title("Embeddings colored by species")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
legend = plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize * 0.7,
    title="Species",
    title_fontsize=textsize * 0.7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(1.3)
plt.savefig(Path(figs_dir, "embeddings_species.png"), bbox_inches="tight")


# Plot the embeddings colored by site
plt.figure(figsize=(figwidth, figwidth))
for i, (site, group) in enumerate(df.groupby("site")):
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=site,
        color=site_palette(i),
        edgecolors="none",
        alpha=0.5,
        s=1.5,
    )
plt.title("Embeddings colored by site")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
legend = plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize * 0.7,
    title="Site",
    title_fontsize=textsize * 0.7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(1.3)
plt.savefig(Path(figs_dir, "embeddings_site.png"), bbox_inches="tight")


# Plot the embeddings colored by time
time_palette = plt.get_cmap("Spectral", len(np.unique(df.time)) // 2)
time_palette = [time_palette(i) for i in range((len(np.unique(df.time)) // 2) + 1)][
    ::-1
]
time_palette += time_palette[::-1]

plt.figure(figsize=(figwidth, figwidth))
for i, (time, group) in enumerate(df.sort_values("time").groupby("time")):
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=time,
        alpha=0.5,
        s=1.5,
        color=time_palette[i],
    )
plt.title("Embeddings colored by time")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.colors.ListedColormap(time_palette),
    norm=plt.Normalize(vmin=0, vmax=len(df["time"].unique())),
)
sm.set_array([])
cbarticks = [time.strftime("%H:%M") for time in sorted(df["time"].unique())][::12][::-1]
cb = plt.colorbar(
    sm,
    fraction=0.046,
    pad=0.04,
    ax=plt.gca(),
    ticks=range(0, len(df["time"].unique()), 12),
)
cb.ax.set_yticklabels(cbarticks)
cb.ax.set_ylim(0, len(df["time"].unique()))  # Set the colorbar height
plt.gca().set_aspect(1.3)
plt.xticks([])
plt.yticks([])
plt.savefig(Path(figs_dir, "embeddings_time.png"), bbox_inches="tight")


# Cluster the embeddings using HDBSCAN
clusterer = cuml.HDBSCAN(min_samples=15, min_cluster_size=200)
df.loc[:, "cluster"] = clusterer.fit_predict(umap_embedding)

plt.figure(figsize=(figwidth, figwidth))
for i, (cluster, group) in enumerate(df.groupby("cluster")):
    if cluster == -1:
        plt.scatter(
            group["umap_x"],
            group["umap_y"],
            label=cluster,
            color="grey",
            alpha=0.1,
            s=0.5,
        )
    else:
        plt.scatter(
            group["umap_x"],
            group["umap_y"],
            label=cluster,
            alpha=0.5,
            s=0.5,
            color=cluster_palette_4[cluster],
        )
plt.title("Embeddings colored by cluster")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
legend = plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize * 0.7,
    title="Cluster",
    title_fontsize=textsize * 0.7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(1.3)
plt.savefig(Path(figs_dir, "embeddings_cluster.png"), bbox_inches="tight")


# Plot sample spectrograms for each cluster
num_clusters = len(df[df["cluster"] != -1]["cluster"].unique())
num_columns = 3
fig, axes = plt.subplots(
    num_clusters, num_columns, figsize=(figwidth, (figwidth / 3) * num_clusters)
)
for cluster, group in df.groupby("cluster"):
    if cluster == -1:
        continue
    group = group.sample(n=min(num_columns, len(group)))
    for i, (index, row) in enumerate(group.iterrows()):
        fpath = Path(rootdir, "data", "raw", row["site"], f"{row['file_name']}.WAV")
        y, sr = librosa.load(fpath, sr=None)
        start_sample = int(row["start_time"] * sr)
        end_sample = int(row["end_time"] * sr)
        y = y[start_sample:end_sample]
        S = np.abs(librosa.stft(y))
        S = librosa.amplitude_to_db(S, ref=np.max)

        ax = axes[cluster][i]
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="linear", ax=ax)
        ax.set_ylim([0, 15000])
        ax.text(
            0.05,
            0.89,
            row["site"],
            transform=ax.transAxes,
            fontsize=textsize * 0.8,
            color="white",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 0:
            ax.text(
                -0.1,
                0.5,
                f"Cluster {cluster}",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=textsize,
                fontweight="bold",
                color=cluster_palette_4[cluster],
            )
plt.tight_layout()
plt.savefig(Path(figs_dir, "embeddings_cluster_spectrograms.png"), bbox_inches="tight")

# get file names for cluster 2
cluster_2_files = df[df["cluster"] == 2][
    ["site", "file_name", "start_time", "end_time"]
]


# ──── PLOT EMBEDDINGS FOR BIRDS ONLY ─────────────────────────────────────────


df_bird_detections = df[df["common_name"] != "Unknown"]

# project embeddings to 2D using UMAP
X = np.array(df_bird_detections["embeddings"].tolist())
umap_embedding_birds = cuml.UMAP(
    n_neighbors=10, min_dist=0.1, n_components=2
).fit_transform(X)

df_bird_detections.loc[:, "umap_x"] = umap_embedding_birds[:, 0]
df_bird_detections.loc[:, "umap_y"] = umap_embedding_birds[:, 1]

# plot the embeddings colored by species
plt.figure(figsize=(figwidth, figwidth))
palette = plt.get_cmap("tab20", len(df_bird_detections["common_name"].unique()))
for i, (species, group) in enumerate(df_bird_detections.groupby("common_name")):
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=species,
        s=2,
        alpha=0.85,
        color=palette(i),
    )
plt.title("Bird embeddings colored by species")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize * 0.7,
    title="Species",
    title_fontsize=textsize * 0.7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.xticks([])
plt.yticks([])
for handle in plt.gca().get_legend().legendHandles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.gca().set_aspect(1)
plt.savefig(Path(figs_dir, "embeddings_birds_species.png"), bbox_inches="tight")

# Now color by site
plt.figure(figsize=(figwidth, figwidth))
for i, (site, group) in enumerate(df_bird_detections.groupby("site")):
    plt.scatter(
        group["umap_x"], group["umap_y"], label=site, s=2, alpha=0.85, c=site_palette(i)
    )
plt.title("Bird embeddings colored by site")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize * 0.7,
    title="Site",
    title_fontsize=textsize * 0.7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.xticks([])
plt.yticks([])
for handle in plt.gca().get_legend().legendHandles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.gca().set_aspect(1)
plt.savefig(Path(figs_dir, "embeddings_birds_site.png"), bbox_inches="tight")


# ──── WREN ANALYSIS ──────────────────────────────────────────────────────────

X = np.array(df[df["common_name"] == "Eurasian Wren"]["embeddings"].tolist())
umap_embedding_wren = cuml.UMAP(
    metric="euclidean", n_neighbors=10, min_dist=0.1, n_components=2
).fit_transform(X)

wren_df = df[df["common_name"] == "Eurasian Wren"]
wren_df.loc[:, "umap_x"] = umap_embedding_wren[:, 0]
wren_df.loc[:, "umap_y"] = umap_embedding_wren[:, 1]


plt.figure(figsize=(figwidth, figwidth))
for i, (site, group) in enumerate(wren_df.groupby("site")):
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=site,
        color=site_palette(i),
        alpha=0.7,
        s=30,
        edgecolors="none",
    )
plt.xlabel("UMAP 1", color="white", fontsize=textsize)
plt.ylabel("UMAP 2", color="white", fontsize=textsize)
legend = plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize,
    title="Site",
    title_fontsize=textsize,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(0.9)
plt.title("Eurasian Wren embeddings by site", fontsize=textsize)
plt.savefig(Path(figs_dir, "wren_sites.png"), bbox_inches="tight")

# ──── CLUSTER WREN DATA ──────────────────────────────────────────────────────

clusterer = cuml.HDBSCAN(min_samples=10, min_cluster_size=5)
wren_df.loc[:, "cluster"] = clusterer.fit_predict(umap_embedding_wren)

plt.figure(figsize=(figwidth, figwidth))
for i, (cluster, group) in enumerate(wren_df.groupby("cluster")):
    if cluster == -1:
        continue
    plt.scatter(
        group["umap_x"],
        group["umap_y"],
        label=cluster,
        color=cluster_palette[i],
        alpha=0.7,
        s=30,
        edgecolors="none",
    )
plt.xlabel("UMAP 1", color="white", fontsize=textsize)
plt.ylabel("UMAP 2", color="white", fontsize=textsize)
legend = plt.legend(
    frameon=False,
    handletextpad=0.1,
    fontsize=textsize,
    title="Cluster",
    title_fontsize=textsize,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
for text in legend.get_texts():
    text.set_color("white")
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.gca().set_aspect(0.9)
plt.title("Eurasian Wren embeddings by cluster", fontsize=textsize)
plt.savefig(Path(figs_dir, "wren_clusters.png"), bbox_inches="tight")


# get the file names and times for each cluster
cluster_files = wren_df.groupby("cluster")[
    ["site", "file_name", "start_time", "end_time"]
].apply(lambda x: x.values.tolist())


# Now plot sample spectrograms for each cluster
num_clusters = len(cluster_files)
num_columns = 3
fig, axes = plt.subplots(
    num_clusters, num_columns, figsize=(figwidth, (figwidth / 3) * num_clusters)
)
for cluster, files in cluster_files.items():
    if cluster == -1:
        continue
    for i, (site, file_name, start_time, end_time) in enumerate(files[:num_columns]):
        fpath = Path(rootdir, "data", "raw", site, f"{file_name}.WAV")
        y, sr = librosa.load(fpath, sr=None)
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        y = y[start_sample:end_sample]
        S = np.abs(librosa.stft(y))
        S = librosa.amplitude_to_db(S, ref=np.max)
        S[S < -40] = -40
        ax = axes[cluster][i]
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="linear", ax=ax)
        ax.set_ylim([0, 15000])
        ax.text(
            0.05,
            0.89,
            site,
            transform=ax.transAxes,
            fontsize=textsize * 0.8,
            color="white",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 0:
            ax.text(
                -0.1,
                0.5,
                f"Cluster {cluster}",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=textsize,
                fontweight="bold",
                color=cluster_palette[cluster],
            )
        # print file name and cluster
        print(f"File name: {file_name} from cluster {cluster} and site {site}")
plt.tight_layout()
plt.savefig(Path(figs_dir, "wren_clusters_spectrograms.png"), bbox_inches="tight")


# ──── DISTANCE BETWEEN EMBEDDINGS FOR EACH SITE ACROSS TIME ──────────────────

# Get the mean of the embeddings for each site and time, and order by time
site_time_embeddings = (
    df.groupby(["site", "datetime"])["embeddings"]
    .apply(lambda x: np.mean(x.tolist(), axis=0))
    .reset_index()
)

site_list = np.unique(site_time_embeddings.site)

site_time_embeddings = (
    site_time_embeddings.groupby("datetime")["embeddings"]
    .apply(lambda x: np.stack(x))
    .reset_index()
)

# Remove any time for which there are not embeddings for all sites
site_time_embeddings = site_time_embeddings[
    site_time_embeddings["embeddings"].apply(lambda x: x.shape[0]) == len(site_list)
]


# Calculate the pairwise distances between each pair of embeddings within each list of embeddings
def calculate_distances(embeddings):
    return np.array([np.linalg.norm(x - y) for x in embeddings for y in embeddings])


distances = site_time_embeddings["embeddings"].apply(calculate_distances)
distances_df = distances.explode().reset_index()
distances_df = distances_df.join(site_time_embeddings["datetime"], on="index")
site_combinations = np.array(np.meshgrid(site_list, site_list)).T.reshape(-1, 2)
site_combinations_df = pd.DataFrame(site_combinations, columns=["site1", "site2"])
mcombs = [site_combinations_df] * len(site_time_embeddings)
site_combinations_df = pd.concat(mcombs, ignore_index=True)
distances_df = pd.concat([distances_df, site_combinations_df], axis=1)
distances_df.rename(columns={"embeddings": "distance"}, inplace=True)
distances_df = distances_df[distances_df["site1"] != distances_df["site2"]]

# Plot the distance between sites over time

# Get the average distance for each site with all other sites for each time step
site_distances = (
    distances_df.groupby(["datetime", "site1"])["distance"].mean().reset_index()
)

# Plot the distances
plt.figure(figsize=(14, 6))
for i, (site, group) in enumerate(site_distances.groupby("site1")):
    plt.plot(
        group["datetime"].dt.strftime("%H:%M"),
        group["distance"],
        label=site,
        color=site_palette(i),
    )

plt.title("Average 'acoustic' distance between sites over time", fontsize=textsize)
plt.xlabel("Time", fontsize=textsize, labelpad=10)
plt.ylabel("Dissimilarity", fontsize=textsize, labelpad=10)
plt.xticks(rotation=0, fontsize=textsize)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.legend(
    fontsize=textsize,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    title="Site",
    title_fontsize=textsize,
    frameon=False,
    handletextpad=0.6,
    markerscale=5,
)
# make legend markers bigger and alpha 1
for handle in plt.gca().get_legend().legendHandles:
    handle.set_linewidth(7)
    handle.set_alpha(1)
plt.savefig(Path(figs_dir, "site_distances.png"), bbox_inches="tight")

# Get the point change for AM49
site_AM49 = site_distances[site_distances["site1"] == "AM49"].reset_index(drop=True)
site_AM49.loc[:, "change"] = site_AM49["distance"].diff()
site_AM49.loc[:, "change"] = site_AM49["change"].abs()
max_change = site_AM49.iloc[site_AM49["change"].idxmax()]
max_change_idx = site_AM49.index.get_loc(max_change.name)
max_change = site_AM49.iloc[max_change_idx - 2 : max_change_idx + 3].reset_index(
    drop=True
)
max_change["filename"] = max_change["datetime"].dt.strftime("%Y%m%d_%H%M%S")
# Answer: rain


# Calculate mean embedding per site
site_embeddings = (
    df[df["common_name"] != "Unknown"]
    .groupby("site")["embeddings"]
    .apply(lambda x: np.mean(x.tolist(), axis=0))
    .reset_index()
)

# Calculate a distance matrix between sites and plot it
site_distances = np.array(site_embeddings["embeddings"].tolist())
site_distances = np.linalg.norm(site_distances[:, None] - site_distances, axis=2)

plt.figure(figsize=(10, 10))
site_distances_no_diag = np.copy(site_distances)
np.fill_diagonal(site_distances_no_diag, np.nan)
# Scale the distances between 0 and 1
plt.imshow(site_distances_no_diag, cmap="viridis", interpolation="none")
colorbar = plt.colorbar(label="Distance", fraction=0.046, pad=0.04)
colorbar.ax.tick_params(labelsize=textsize)
colorbar.set_label("Distance", fontsize=textsize)
plt.xticks(range(len(site_list)), site_list, rotation=0, fontsize=textsize)
plt.yticks(range(len(site_list)), site_list, fontsize=textsize)
plt.title("Average distance between Sites", fontsize=textsize)
plt.savefig(Path(figs_dir, "site_distances_matrix.png"), bbox_inches="tight")
