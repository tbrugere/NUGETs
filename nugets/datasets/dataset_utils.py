import requests
from pathlib import Path
import zipfile

def download_from_url(url: str, savepath: Path, chunk_size: int=1024*1024):
    """
    Downloads raw data from a url into the directory specified in savepath
    
    Returns: path where the dataset has been saved
    """
    if savepath.exists():
        print(f'File exists at {savepath}')
        return savepath 
    with requests.get(url, stream=True) as r: 
        r.raise_for_status()

        total = r.headers.get("Content-Length")
        total = int(total)
        downloaded = 0

        with open(savepath, 'wb') as f: 
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                pct = 100.0 * downloaded/total
                print(f"\r Downloading dataset: {downloaded/1e9:5.2f}/{total/1e9:5.2f} GB ({pct:5.2f}%)", end="", flush=True)
    print()
    print(f"Saved in {savepath}")
    return savepath

def extract_data_from_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        if not names:
            print(f"[warn] empty zip: {zip_path}")
            return
        first = out_dir / names[0]
        if first.exists():
            print(f"[skip extract] {zip_path.name} -> {out_dir} (already extracted)")
            return
        z.extractall(out_dir)

def extract_nested_zips(raw_dir: Path, out_dir: Path): 
    out_dir.mkdir(parents=True, exist_ok=True)
    nested_zips = sorted(raw_dir.rglob("*.zip"))
    if not nested_zips:
        print("[info] no nested .zip files found.")
        return
    print(f"[info] found {len(nested_zips)} nested .zip tile archives; extracting to {out_dir}")
    for zp in nested_zips:
        # each nested zip is extracted into a subfolder named after it
        sub = out_dir / zp.stem
        extract_data_from_zip(zp, sub)
