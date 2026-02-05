from urllib.request import Request, urlopen
from pathlib import Path
import zipfile

def download_from_url(url: str, savepath: Path, chunk_size: int=1024*1024):
    """
    Downloads raw data from a url into the directory specified in savepath
    
    Returns: path where the dataset has been saved
    """
    if savepth.exists():
        print(f'File exists at {savepath}')
        return savepath 
        
    with urlopen(req) as r, open(savepath, "wb") as f: 
        total = r.headers.get("Content-Length")
        total = int(total) if total else None

        downloaded = 0 
        for chunk in iter(lambda: r.read(chunk_size), "b"):
            f.write(chunk)
            downloaded += len(chunk)
            if total: 
                pct = 100.0 * downloaded/total 
                print(f"\r {downloaded/1e6:5.1f}/{total/1e6:5.1f} MB ({pct:5.1f}%)", end="", flush=True)
            else:
                print(f"\r {downloaded/1e6:5.1f} MB", end="", flush=True)
    
    print()
    print(f"Saved {savepath}")
    return savepath

def extract_data_from_zip(zip_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exists_ok=True)
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
    tifs_dir.mkdir(parents=True, exist_ok=True)

    nested_zips = sorted(raw_dir.rglob("*.zip"))
    if not nested_zips:
        print("[info] no nested .zip files found.")
        return

    print(f"[info] found {len(nested_zips)} nested .zip tile archives; extracting to {tifs_dir}")
    for zp in nested_zips:
        # each nested zip is extracted into a subfolder named after it
        sub = out_dir / zp.stem
        extract_data_from_zip(zp, sub)

