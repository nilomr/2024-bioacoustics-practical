import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

rootdir = Path(__file__).parent.parent
detections_file = Path(rootdir, "data", "derived", "detections.json")

# Read the detections from the json file and convert to a pandas DataFrame
with open(detections_file, encoding="utf-8") as file:
    detections = json.load(file)

df = pd.DataFrame(detections, columns=["file_name", "site", "detections"])
df = df.explode("detections").dropna(subset=["detections"])
df = pd.concat([df, df["detections"].apply(pd.Series)], axis=1)
df["start_time"] = pd.to_datetime(
    df["file_name"], format="%Y%m%d_%H%M%S"
) + pd.to_timedelta(df["start_time"], unit="s")
df["end_time"] = pd.to_datetime(
    df["file_name"], format="%Y%m%d_%H%M%S"
) + pd.to_timedelta(df["end_time"], unit="s")
df = df.sort_values(["file_name", "start_time"])


# Plot the number of detections by site
site_counts = df["site"].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(site_counts.index, site_counts.values)
plt.title("Number of detections by site")
plt.xlabel("Site")
plt.ylabel("Count")
plt.show()

# Plot the number of detections by species
species_counts = df["common_name"].value_counts()
plt.figure(figsize=(10, 6))
plt.barh(species_counts.index, species_counts.values)
plt.title("Number of detections by species")
plt.xlabel("Count")
plt.ylabel("Species")
plt.gca().invert_yaxis()
plt.show()

# Plot the number of detections by time in 30min bins throughout the recording
# period
df["time_bin"] = df["start_time"].dt.floor("30min")
time_counts = df.groupby(["time_bin", "site"]).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))

plt.plot(time_counts.index, time_counts)
plt.title("Number of detections by time")
plt.xlabel("Time")
plt.ylabel("Count")
plt.legend()
plt.show()


# ──── PLOT THE NUMBER OF DETECTIONS BY SITE AND SPECIES ──────────────────────

species_sites = df.groupby(["common_name", "site"]).size().unstack(fill_value=0)
species_sites = species_sites[sorted(species_sites.columns, key=lambda x: int(x[2:]))]
species_sites = species_sites.loc[
    species_sites.sum(axis=1).sort_values(ascending=False).index
]

plt.figure(figsize=(8, 8))
cmap = plt.get_cmap("viridis")
cmap.set_bad("black")
plt.imshow(np.log(species_sites), cmap=cmap, aspect="auto")

# Add counts to each cell in white text
for i in range(len(species_sites.index)):
    for j in range(len(species_sites.columns)):
        count = species_sites.iloc[i, j]
        if count > 0:
            plt.text(j, i, str(count), ha="center", va="center", color="white")

plt.title("Vocal activity by site")
plt.xlabel("Site")
plt.ylabel("Species")
plt.yticks(range(len(species_sites.index)), species_sites.index)
plt.xticks(range(len(species_sites.columns)), species_sites.columns)
plt.show()


# ──── PLOT VOCAL ACTIVITY OVER TIME BY SPECIES ───────────────────────────────

# (only include the top 5 species by number of detections)
top_species = species_counts.head(5).index
species_time = (
    df[df["common_name"].isin(top_species)]
    .groupby(["time_bin", "common_name"])
    .size()
    .unstack(fill_value=0)
)
species_time = species_time.loc[
    species_time.sum(axis=1).sort_values(ascending=False).index
]
# arrange by time_bin
species_time = species_time.sort_index()

# Add missing time bins with zero detections
all_time_bins = pd.date_range(
    start=df["time_bin"].min(), end=df["time_bin"].max(), freq="30min"
)
species_time = species_time.reindex(all_time_bins, fill_value=0)


plt.figure(figsize=(7, 10))
cmap = plt.get_cmap("viridis")
cmap.set_bad("black")
plt.imshow(np.log(species_time), cmap=cmap, aspect="auto")

# Add counts to each cell in white text
for i in range(len(species_time.index)):
    for j in range(len(species_time.columns)):
        count = species_time.iloc[i, j]
        if count > 0:
            plt.text(j, i, str(count), ha="center", va="center", color="white")

plt.title("Vocal activity by species", pad=20)
plt.xlabel("Species", labelpad=20)
plt.ylabel("Time (UTC)", labelpad=20)
plt.yticks(range(len(species_time.index)), species_time.index.strftime("%B %d %H:%M"))
plt.xticks(
    range(len(species_time.columns)),
    [t.replace(" ", "\n") for t in species_time.columns],
)


plt.show()
