import contextlib
import json
import multiprocessing
from datetime import datetime
from multiprocessing import Manager
from pathlib import Path

import pandas as pd
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from tqdm import tqdm

# Get list of all .mp3 files in the specified directory
# find repo rootdir
rootdir = Path(__file__).parent.parent
file_paths = list(Path(rootdir, "data").rglob("*.WAV"))

# Remove files outside the recording period
file_paths = [
    f for f in file_paths if f.stem > "20240501_170000" and f.stem < "20240502_110000"
]
print(f"Found {len(file_paths)} files")


# Load and initialize the BirdNET-Analyzer model
analyzer = Analyzer(version="2.4")

# Create a shared queue to store the detections
manager = Manager()
detections_queue = manager.Queue()


def process_file(file_path):
    date = datetime.strptime(file_path.stem.split("_")[0], "%Y%m%d")
    dir_name = file_path.parent.name

    recording = Recording(
        analyzer,
        str(file_path),
        lat=51.775036,
        lon=-1.336488,
        date=date,
        min_conf=0.75,
    )
    log_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    log_file = Path(rootdir, "logs", f"{log_date}.log")
    with open(log_file, "a") as f:
        with contextlib.redirect_stdout(f):
            recording.analyze()

    detections_queue.put([file_path.stem, dir_name, recording.detections])


# Main loop

# Create a pool of worker processes
pool = multiprocessing.Pool(processes=20)

# Map the process_file function to each file path in parallel
detections = []
with tqdm(total=len(file_paths), desc="Processing files") as pbar:
    for _ in pool.imap_unordered(process_file, file_paths):
        pbar.update(1)
        # Retrieve the detections from the queue
        while not detections_queue.empty():
            detections.append(detections_queue.get())

# Close the pool and wait for all processes to finish
pool.close()
pool.join()

# Save the reults to a json file
detections_file = Path(rootdir, "data", "derived")
detections_file.mkdir(parents=True, exist_ok=True)
detections_file = Path(detections_file, "detections.json")
with open(detections_file, "w") as f:
    json.dump(detections, f)
