import datetime
import multiprocessing as mp
from pathlib import Path

import maad
import matplotlib.pyplot as plt
import pandas as pd
from maad import features, sound
from maad.util import date_parser, false_Color_Spectro
from tqdm import tqdm

rootdir = Path(__file__).parent.parent
file_paths = list(Path(rootdir, "data").rglob("*.WAV"))
# Create the output directory for the figs
Path(rootdir, "figures", "longspecs").mkdir(parents=True, exist_ok=True)

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

# ──── FUNCTION DEFINITIONS ───────────────────────────────────────────────────


def get_sites(rootdir):
    return [
        folder.name
        for folder in Path(rootdir, "data", "raw").iterdir()
        if folder.is_dir()
    ]


def filter_dates(df):
    df["Date"] = df.index
    df = df[(df["Date"] > "2024-05-01 17:00:00") & (df["Date"] < "2024-05-02 11:00:00")]
    df.drop("Date", inplace=True, axis=1)
    return df


def process_row(row):
    fullfilename = row["file"]
    wave, fs = sound.load(
        filename=fullfilename, channel="left", detrend=True, verbose=False
    )
    # Compute the Power Spectrogram Density (PSD) : Sxx_power
    Sxx_power, tn, fn, ext = sound.spectrogram(
        x=wave,
        fs=fs,
        window="hann",
        nperseg=1024,
        noverlap=1024 // 2,
        verbose=False,
        display=False,
        savefig=None,
    )

    # Individual metrics
    # ACI
    _, ACI_per_bin, _ = features.acoustic_complexity_index(
        Sxx=Sxx_power,
    )

    # Spectral Event Count
    Sxx_noNoise = maad.sound.median_equalizer(Sxx_power)
    Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    _, _, EVNspCount_per_bin, _ = maad.features.spectral_events(
        Sxx_dB_noNoise,
        dt=tn[1] - tn[0],
        dB_threshold=6,
        rejectDuration=0.1,
        display=False,
        extent=ext,
    )
    # Spectral entropy
    _, Ht_per_bin = maad.features.frequency_entropy(Sxx_power)
    return [fn, ACI_per_bin, Ht_per_bin, EVNspCount_per_bin]


def process_site(rootdir, site):
    df = date_parser(
        Path(rootdir, "data", "raw", site),
        extension=".WAV",
        dateformat="%Y%m%d_%H%M%S",
        verbose=False,
    )

    if df.empty:
        print(f"\033[91mSkipping site {site}: No data found\033[0m")
        return None

    df = filter_dates(df)
    pool = mp.Pool(20)
    dfdata = pool.map(process_row, [row for _, row in df.iterrows()])
    idc_per_bin = pd.DataFrame(
        dfdata,
        columns=["frequencies", "ACI_per_bin", "Ht_per_bin", "EVNspCount_per_bin"],
    )
    idc_per_bin.index = df.index
    return idc_per_bin


def plot_spectrogram(idc_per_bin, site_name, savefig=True):
    fcarray = false_Color_Spectro(
        idc_per_bin, display=False, unit="hours", figsize=[30, 20]
    )
    channel_mapping = {"red": 0, "green": 1, "blue": 2}

    # prepare ticks
    xticks = idc_per_bin.index
    start_time = idc_per_bin.index[0]
    end_time = idc_per_bin.index[-1]
    start_hour = datetime.datetime(
        start_time.year, start_time.month, start_time.day, start_time.hour
    )
    end_hour = datetime.datetime(
        end_time.year, end_time.month, end_time.day, end_time.hour
    )
    xticks_hour = pd.date_range(
        start=start_hour + datetime.timedelta(hours=1), end=end_hour, freq="2h"
    )

    yticks = idc_per_bin.iloc[0]["frequencies"]

    plt.figure(figsize=(14, 4))
    plt.imshow(
        fcarray[0][
            :,
            :,
            (channel_mapping["red"], channel_mapping["green"], channel_mapping["blue"]),
        ],
        origin="lower",
        aspect="auto",
        extent=[xticks[0], xticks[-1], yticks[0], yticks[-1]],
        interpolation=None,
    )

    plt.text(
        0.98,
        0.95,
        site_name,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        fontweight="bold",
        fontsize=textsize,
    )
    ytick_positions = [0, 5000, 10000, 15000, 20000]
    ytick_labels = [f"{i//1000}k Hz" for i in ytick_positions]
    plt.yticks(ytick_positions, ytick_labels)
    plt.xticks(xticks_hour, xticks_hour.strftime("%H:%M"))

    if savefig:
        filename = rootdir / "figures" / "longspecs" / f"{site_name}_spectrogram.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_channels(idc_per_bin, site_name, savefig=True):
    fcarray = false_Color_Spectro(
        idc_per_bin, display=False, unit="hours", figsize=[30, 20]
    )
    channel_mapping = {"red": 0, "green": 1, "blue": 2}

    fig, ax = plt.subplots(3, 1, figsize=(14, 9))
    for i, (channel, variable) in enumerate(
        zip(
            ["red", "green", "blue"],
            ["Acoustic\nComplexity", "Spectral\nEntropy", "Event\nCount"],
        )
    ):
        color_cmap = (
            "Reds" if channel == "red" else "Greens" if channel == "green" else "Blues"
        )
        ax[i].imshow(
            fcarray[0][:, :, channel_mapping[channel]],
            origin="lower",
            aspect="auto",
            cmap=color_cmap,
        )
        ax[i].text(
            -0.03,
            0.5,
            variable,
            transform=ax[i].transAxes,
            rotation=90,
            va="center",
            ha="right",
            ma="center",
            fontsize=textsize,
            fontweight="bold",
        )
        ax[i].text(
            0.98,
            0.95,
            site_name,
            transform=ax[i].transAxes,
            ha="right",
            va="top",
            fontweight="bold",
            fontsize=textsize,
        )
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()

    if savefig:
        filename = rootdir / "figures" / "longspecs" / f"{site_name}_channels.png"
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


# ──── MAIN ───────────────────────────────────────────────────────────────────

# Extract long term spectrograms
sites = sorted(get_sites(rootdir))
for site in tqdm(sites, desc="Processing sites"):

    idc_per_bin = process_site(rootdir, site)
    if idc_per_bin is None:
        continue
    plot_spectrogram(idc_per_bin, site, savefig=True)
    plot_channels(idc_per_bin, site, savefig=True)
