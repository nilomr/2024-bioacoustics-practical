import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.manifold import MDS

rootdir = Path(__file__).parent.parent
detections_file = Path(rootdir, "data", "derived", "detections.json")
fig_dir = Path(rootdir, "figures", "detections")
fig_dir.mkdir(parents=True, exist_ok=True)


# ──── PLOT SETTINGS ──────────────────────────────────────────────────────────

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
        "axes.titlepad": 17,
        "font.size": 15,  # set all text size to 15
        "xtick.labelsize": 13,  # set x-axis tick font size to 13
        "ytick.labelsize": 13,  # set y-axis tick font size to 13
    }
)
site_palette = [plt.get_cmap("Spectral", 7)(i) for i in range(7)]
cluster_palette = ["#ed6a5a", "#f4f1bb", "#9bc1bc"]
cluster_palette_4 = ["#335c67", "#fff3b0", "#e09f3e", "#9e2a2b"]

figwidth = 8
textsize = 15

# ──── DATA INGEST ────────────────────────────────────────────────────────────

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
df["time_bin"] = df["start_time"].dt.floor("30min")

# ──── PLOTS ──────────────────────────────────────────────────────────────────

# Plot the number of detections by site
site_counts = df["site"].value_counts()
# arrange alphabetically
site_counts = site_counts.sort_index()
plt.figure(figsize=(10, 6))
plt.bar(site_counts.index, site_counts.values, color=site_palette)
plt.title("Number of detections by site")
plt.xlabel("Site")
plt.ylabel("Count")
plt.savefig(fig_dir / "detections_by_site.png", bbox_inches="tight")

# Plot the number of detections by species (linear)
species_counts = df["common_name"].value_counts()
plt.figure(figsize=(6, 6))
plt.barh(species_counts.index, species_counts.values, color="#acacac")
plt.title("Number of detections by species")
plt.xlabel("Count")
plt.ylabel("Species")
plt.gca().invert_yaxis()
plt.tick_params(axis="x", which="major", width=1, length=8)
plt.tick_params(axis="x", which="minor", width=1, length=5)
plt.savefig(fig_dir / "detections_by_species.png", bbox_inches="tight")

# Plot the number of detections by species (log)
species_counts = df["common_name"].value_counts()
plt.figure(figsize=(6, 6))
plt.barh(species_counts.index, species_counts.values, color="#acacac")
plt.xscale("log")  # Set x-axis scale to logarithmic
plt.xticks([1, 10, 100, 1000])  # Set x-axis tick positions
plt.title("Number of detections by species")
plt.xlabel("Count")
plt.ylabel("Species")
plt.gca().invert_yaxis()
plt.tick_params(axis="x", which="major", width=1, length=8)
plt.tick_params(axis="x", which="minor", width=1, length=5)
plt.savefig(fig_dir / "detections_by_species_log.png", bbox_inches="tight")


# Plot the number of detections by time in 30min bins throughout the recording
# period

time_counts = df.groupby(["time_bin", "site"]).size().unstack(fill_value=0)
plt.figure(figsize=(9, 4))

for i, site in enumerate(time_counts.columns):
    plt.plot(time_counts.index, time_counts[site], color=site_palette[i], label=site)

plt.title("Number of detections across time")
plt.xlabel("Time")
plt.ylabel("Count")
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
for handle in plt.gca().get_legend().legendHandles:
    handle.set_linewidth(7)
    handle.set_alpha(1)

# Format x-axis tick labels as 00:00
plt.xticks(ha="center")
plt.gca().xaxis.set_major_formatter(
    plt.FixedFormatter(time_counts.index.strftime("%H:%M"))
)
plt.savefig(fig_dir / "detections_by_time.png", bbox_inches="tight")


# ──── PLOT THE NUMBER OF DETECTIONS BY SITE AND SPECIES ──────────────────────

species_sites = df.groupby(["common_name", "site"]).size().unstack(fill_value=0)
species_sites = species_sites[sorted(species_sites.columns, key=lambda x: int(x[2:]))]
species_sites = species_sites.loc[
    species_sites.sum(axis=1).sort_values(ascending=False).index
]

plt.figure(figsize=(7, 7))
cmap = plt.get_cmap("viridis")
cmap.set_bad("black")
plt.imshow(np.log(species_sites), cmap=cmap, aspect="auto")

# Add counts to each cell in white text
for i in range(len(species_sites.index)):
    for j in range(len(species_sites.columns)):
        count = species_sites.iloc[i, j]
        if count > 0:
            plt.text(
                j, i, str(count), ha="center", va="center", color="white", fontsize=10
            )

plt.title("Vocal activity by site")
plt.xlabel("Site")
plt.ylabel("Species")
plt.yticks(range(len(species_sites.index)), species_sites.index)
plt.xticks(range(len(species_sites.columns)), species_sites.columns)
plt.savefig(fig_dir / "detections_by_site_species.png", bbox_inches="tight")


# ──── PLOT VOCAL ACTIVITY OVER TIME BY SPECIES ───────────────────────────────

# (only include the top 6 species by number of detections)
top_species = species_counts.head(6).index
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
for i in range(len(species_time.index)):
    for j in range(len(species_time.columns)):
        count = species_time.iloc[i, j]
        if count > 0:
            plt.text(
                j, i, str(count), ha="center", fontsize=10, va="center", color="white"
            )

plt.title("Vocal activity by species", pad=20)
plt.xlabel("Species", labelpad=20)
plt.ylabel("Time (UTC)", labelpad=20)

# Get the indices corresponding to each hour
hour_indices = [i for i, t in enumerate(species_time.index) if t.minute == 0]

# Generate yticks and labels
yticks_positions = hour_indices
yticks_labels = species_time.index[hour_indices].strftime("%H:%M")

plt.yticks(yticks_positions, yticks_labels)
plt.xticks(
    range(len(species_time.columns)),
    [t.replace(" ", "\n") for t in species_time.columns],
)
plt.savefig(fig_dir / "detections_by_species_time.png", bbox_inches="tight")


# ──── PLOT COMMUNITY COMPOSITION OVER TIME BY SITE ───────────────────────────


# Append any missing time bins
all_time_bins = pd.date_range(
    start=df["time_bin"].min(), end=df["time_bin"].max(), freq="30min"
)
#

# from all_time_bins, create a df with the same columns as the original df,
# values as NA
missing_time_bins = pd.DataFrame(index=all_time_bins, columns=df.columns, data=np.nan)
# all time bins to a time_bin column, and reset the index
missing_time_bins["time_bin"] = missing_time_bins.index
missing_time_bins = missing_time_bins.reset_index(drop=True)

# append the missing time bins to the original df as many times as sites,
# changing the site name to the corresponding site for each appended time bin
# dataframe
dfs = []
for site in df["site"].unique():
    site_df = missing_time_bins.copy()
    site_df["site"] = site
    dfs.append(site_df)

df = pd.concat([df] + dfs, ignore_index=True)


# Group by site and time bin, and count the number of different common_name
# values there are
species_presence = df.groupby(["site", "time_bin"])["common_name"].nunique().unstack()


# Change the dataframe from a wide to a long format
species_presence = species_presence.reset_index().melt(
    id_vars="site", var_name="time_bin", value_name="species_count"
)

# Plot the species count over time by site as histograms
# where the number of species is represented by the height of the bars
# and bar represents a time bin. One subplot per site in 2 rows and 4 columns.
# bar color by site
fig, axs = plt.subplots(1, 7, figsize=(20, 4), sharex=True, sharey=True)
for i, (site, data) in enumerate(species_presence.groupby("site")):
    ax = axs[i]
    ax.bar(data.index, data["species_count"], color=site_palette[i], width=10)

    ax.set_title(site, fontsize=textsize * 1.3, fontweight="bold")
    ax.set_xlabel("Time", fontsize=textsize, labelpad=10)
    if i == 0:
        ax.set_ylabel("Species count", fontsize=textsize, labelpad=10)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=textsize)
    # Add lines at y tick positions
    ax.grid(axis="y", linestyle="-", alpha=0.5)

plt.savefig(fig_dir / "species_count_by_site.png")


# Plot cumulative count of species over time by site
unique_df = df.groupby(["site", "time_bin"])["common_name"].unique().reset_index()
cumulative_species = {}

df.sort_values(by=["site", "time_bin"], inplace=True)

cumulative_species = {}

for _, row in df.iterrows():
    site = row["site"]
    time_bin = row["time_bin"]
    species = row["common_name"]
    if site not in cumulative_species:
        cumulative_species[site] = {time_bin: [species]}
    else:
        if time_bin not in cumulative_species[site]:
            cumulative_species[site][time_bin] = [species]
        else:
            cumulative_species[site][time_bin] = list(
                set(cumulative_species[site][time_bin] + [species])
            )

# for each site calculate the total number of unique species detected until
# that point in time by adding the lists together, eg time bin two would have
# the species in t1 and t2, time bin three would have the species in t1, t2 and
# t3, etc

df.sort_values(by=["site", "time_bin"], inplace=True)

cumdfs = []
for site in df.site.unique():
    # get all the species detected in each time bin
    species = df[df["site"] == site].groupby("time_bin")["common_name"].unique()
    # calculate the cumulative list of species detected eg if the
    # first list has dunnock and the second has dunnock and blackbird and robin,
    # the second list should be dunnock, blackbird, robin, eetc
    cumlists = []
    # remove any pd nan from the lists
    species = species.apply(lambda x: x[~pd.isnull(x)])
    for i in range(len(species)):
        cumlists.append(np.unique(np.concatenate(species[: i + 1])))
        # create a new series with the time bin as the first column and the
        # cumulative list as the second
    cumdfs.append(pd.Series(cumlists, index=species.index))

# create a df from the series, with site, time bin, and list as columns
cumdf = pd.concat(cumdfs, keys=df.site.unique(), names=["site"]).reset_index()

# change 0 column name to species_list
cumdf.columns = ["site", "time_bin", "species_list"]

# count the number of species in each time bin
cumdf["species_count"] = cumdf["species_list"].apply(len)
# order by time bin
cumdf = cumdf.sort_values(by=["site", "time_bin"])

# plot the cumulative species count over time by site
fig, axs = plt.subplots(1, 7, figsize=(20, 4), sharex=True, sharey=True)
for i, (site, data) in enumerate(cumdf.groupby("site")):
    ax = axs[i]
    ax.plot(data.time_bin, data["species_count"], color=site_palette[i], linewidth=5)
    ax.set_title(site, fontsize=textsize * 1.3, fontweight="bold")
    ax.set_xlabel("Time", fontsize=textsize, labelpad=10)
    if i == 0:
        ax.set_ylabel("Cumulative species count", fontsize=textsize, labelpad=10)
    ax.set_xticks([])
    ax.set_yticks(range(0, 14, 2))
    ax.tick_params(axis="y", labelsize=textsize)
    ax.grid(axis="y", linestyle="-", alpha=0.5)
plt.savefig(fig_dir / "cumulative_species_count_by_site.png")


# get the last cumulative list for each site
cumdf_last = cumdf.groupby("site").tail(1)

# calculate the jaccard distance between the species list for all sites and build a
# distance matrix


# Create a DataFrame where each row represents a site and each column represents a species
species_df = pd.DataFrame(index=cumdf_last["site"].unique())

for index, row in cumdf_last.iterrows():
    for species in row["species_list"]:
        if species not in species_df.columns:
            species_df[species] = 0
        species_df.loc[row["site"], species] = 1

# Calculate Jaccard distances
jaccard_distances = distance.pdist(species_df.values, metric="jaccard")

# Convert to square distance matrix
square_distance_matrix = distance.squareform(jaccard_distances)

# Convert to DataFrame
distance_df = pd.DataFrame(
    square_distance_matrix, index=species_df.index, columns=species_df.index
)

# Plot as a matrix, adding the site labels:
plt.figure(figsize=(10, 10))
cmap = plt.get_cmap("viridis")
cmap.set_bad("black")
plt.imshow(distance_df, cmap=cmap)
plt.colorbar(label="Jaccard distance")
plt.title("Jaccard distance between sites")
plt.xticks(range(len(distance_df.columns)), distance_df.columns)
plt.yticks(range(len(distance_df.index)), distance_df.index)

# Plot a MDS of the matrix
mds = MDS(n_components=2, dissimilarity="precomputed")
embedding = mds.fit_transform(square_distance_matrix)

plt.figure(figsize=(8, 8))
for i, site in enumerate(species_df.index):
    plt.scatter(
        embedding[i, 0], embedding[i, 1], s=140, color=site_palette[i], label=site
    )
    plt.text(
        embedding[i, 0],
        embedding[i, 1] + 0.02,
        site,
        fontsize=13,
        ha="center",
        va="bottom",
    )

plt.title("MDS of Jaccard $d$ between site communities")
plt.xlabel("MDS1")
plt.ylabel("MDS2")
plt.xticks([])
plt.yticks([])
plt.savefig(fig_dir / "community_jaccard_d.png")


# ──── SAVE ALL DERIVED DATA TO DISK ──────────────────────────────────────────
species_counts.to_csv(Path(rootdir, "data", "derived", "species_counts.csv"))
time_counts.to_csv(Path(rootdir, "data", "derived", "time_counts.csv"))
species_sites.to_csv(Path(rootdir, "data", "derived", "species_sites.csv"))
species_time.to_csv(Path(rootdir, "data", "derived", "species_time.csv"))
species_presence.to_csv(Path(rootdir, "data", "derived", "species_presence.csv"))
cumdf.to_csv(Path(rootdir, "data", "derived", "cumulative_species.csv"))
cumdf_last.to_csv(Path(rootdir, "data", "derived", "cumulative_species_last.csv"))
distance_df.to_csv(Path(rootdir, "data", "derived", "distance_matrix.csv"))

# create a dataframe with the number of detections by site, time and species
species_site_time = (
    df.groupby(["site", "start_time", "common_name"]).size().unstack(fill_value=0)
).reset_index()

species_site_time.to_csv(Path(rootdir, "data", "derived", "species_site_time.csv"))

# Save a readme
with open(Path(rootdir, "data", "derived", "README.txt"), "w") as file:
    file.write(
        """This folder contains the following files:
- species_counts.csv: Number of detections by species
- time_counts.csv: Number of detections by time
- species_sites.csv: Number of detections by species and site
- species_time.csv: Number of detections by species and time
- species_presence.csv: Number of species detected by site and time
- cumulative_species.csv: Cumulative species count by site and time
- cumulative_species_last.csv: Cumulative species count for the last time bin
- distance_matrix.csv: Jaccard distance between sites
- species_site_time.csv: Number of detections by site, time and species
"""
    )


# get all rows where common_name includes goldcrest
df_nona = df.dropna(subset=["common_name"])
goldcrest_df = df_nona[df_nona["common_name"].str.contains("goldfinch", case=False)]
