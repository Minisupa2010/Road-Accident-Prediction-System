import requests
import argparse
from pathlib import Path

# Candidate URLs (DfT official CSVs)
URLS = [
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2022.csv",
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2021.csv",
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2020.csv",
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2019.csv",
]

def download(url, out_file):
    print(f"Trying: {url}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✅ Downloaded {url} -> {out_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV file path")
    args = parser.parse_args()

    success = False
    for url in URLS:
        try:
            success = download(url, args.out)
            break
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue

    if not success:
        print("All downloads failed. Please check DfT portal: https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-accidents-safety-data")
