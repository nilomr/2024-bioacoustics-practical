import dis
import json
from audioop import add
from pathlib import Path
from turtle import color, title

import cuml
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rootdir = Path(__file__).parent.parent
detections_file = Path(rootdir, "data", "derived", "detections.json")
embeddings_file = Path(rootdir, "data", "derived", "embeddings.json")

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


# Plot the embeddings colored by species
plt.figure(figsize=(figwidth, figwidth))
# Plot the embeddings colored by species, with 'Unknown' plotted first
plt.scatter(
    df[df["common_name"] == "Unknown"]["umap_x"],
    df[df["common_name"] == "Unknown"]["umap_y"],
    label="Unknown",
    alpha=0.1,
    color="#eaeaea",
    s=1,
)
for species, group in df[df["common_name"] != "Unknown"].groupby("common_name"):
    plt.scatter(group["umap_x"], group["umap_y"], label=species, alpha=0.5, s=2)
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
plt.show()


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
        s=2,
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
plt.show()

# Plot the embeddings colored by time
# Create a time palette based on spectral palette

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
        s=1,
        color=time_palette[i],
    )
plt.title("Embeddings colored by time")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
# add a colorbar legend based on time
sm = plt.cm.ScalarMappable(
    cmap=plt.cm.colors.ListedColormap(time_palette),
    norm=plt.Normalize(vmin=0, vmax=len(df["time"].unique())),
)
sm.set_array([])
cbarticks = [time.strftime("%H:%M") for time in sorted(df["time"].unique())][::12][::-1]
cb = plt.colorbar(sm, ax=plt.gca(), ticks=range(0, len(df["time"].unique()), 12))
cb.ax.set_yticklabels(cbarticks)
cb.ax.set_ylim(0, len(df["time"].unique()))  # Set the colorbar height
plt.xticks([])
plt.yticks([])
plt.show()


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
            s=1,
        )
    else:
        plt.scatter(
            group["umap_x"],
            group["umap_y"],
            label=cluster,
            alpha=0.5,
            s=2,
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
plt.show()


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
plt.show()

# Get a separate dataframe with the data for cluster == 0
df_cluster_0 = df[df["cluster"] == 0]


# ──── PLOT EMBEDDINGS FOR BIRDS ONLY ─────────────────────────────────────────

# filter rows with no detections
df_detections = df.dropna(subset=["common_name"])

# project embeddings to 2D using UMAP
X = np.array(df_detections["embeddings"].tolist())
umap_embedding = cuml.UMAP(n_neighbors=10, min_dist=0.07, n_components=3).fit_transform(
    X
)

df_detections.loc[:, "umap_x"] = umap_embedding[:, 0]
df_detections.loc[:, "umap_y"] = umap_embedding[:, 1]

# plot the embeddings colored by species
plt.figure(figsize=(10, 6))
for species, group in df_detections.groupby("common_name"):
    plt.scatter(group["umap_x"], group["umap_y"], label=species, alpha=0.5)
plt.title("Embeddings colored by species")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()

# Now color by site
plt.figure(figsize=(10, 6))
for site, group in df_detections.groupby("site"):
    plt.scatter(group["umap_x"], group["umap_y"], label=site, alpha=0.5)
plt.title("Embeddings colored by site")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend()
plt.show()


# ──── WREN ANALYSIS ──────────────────────────────────────────────────────────

X = np.array(
    df_detections[df_detections["common_name"] == "Eurasian Wren"][
        "embeddings"
    ].tolist()
)
umap_embedding = cuml.UMAP(
    metric="euclidean", n_neighbors=10, min_dist=0.07, n_components=3
).fit_transform(X)

wren_df = df_detections[df_detections["common_name"] == "Eurasian Wren"]
wren_df.loc[:, "umap_x"] = umap_embedding[:, 0]
wren_df.loc[:, "umap_y"] = umap_embedding[:, 1]


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
    fontsize=textsize * 0.7,
    title="Site",
    title_fontsize=textsize * 0.7,
)
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.title("Eurasian Wren embeddings by site", fontsize=textsize)
plt.show()

# ──── CLUSTER WREN DATA ──────────────────────────────────────────────────────

clusterer = cuml.HDBSCAN(min_samples=10, min_cluster_size=5)
wren_df.loc[:, "cluster"] = clusterer.fit_predict(umap_embedding)

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
legend = plt.legend(frameon=False, handletextpad=0.1, fontsize=textsize)
for text in legend.get_texts():
    text.set_color("white")
for handle in legend.legend_handles:
    handle.set_sizes([50.0])
    handle.set_alpha(1)
plt.xticks([])
plt.yticks([])
plt.title("Eurasian Wren embeddings by cluster", fontsize=textsize)
plt.show()


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
plt.tight_layout()
plt.show()


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

plt.title("Average distance between sites over time", fontsize=textsize)
plt.xlabel("Time", fontsize=textsize)
plt.ylabel("Dissimilarity", fontsize=textsize)
plt.xticks(rotation=0, fontsize=textsize)
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
plt.legend(
    fontsize=textsize,
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    title="Site",
    title_fontsize=textsize,
    frameon=False,
    handletextpad=0.1,
    markerscale=5,
)
plt.show()
