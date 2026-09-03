from typing import Literal
from pathlib import Path

from torch_geometric.datasets import ShapeNet as Pyg_ShapeNet
from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from .dataset_utils import download_from_url, extract_data_from_zip, extract_nested_zips
from nugets.datasets.datapoint_types import Set_datapoint
from nugets.datasets.register import register as dataset_register
import numpy as np
import torch


@dataset_register 
class NZDEM(Dataset[Set_datapoint]):
    """

    3D point clouds representing terrain surfaces of New Zealand
    From OpenTopoData: https://www.opentopodata.org/

    """

    datatype: Set_datapoint

    url: str = 'https://storage.googleapis.com/www-ajnisbet-com/nzdem-may-2020.zip'

    seed: int = 42  # random seed set for sampling from point cloud
    point_clouds_per_terrain: int = 100
    sampling: str = "random" # This is an additional parameter that can change
    split_seed: int = 42 # split for test/train split

    def __init__(self, n_points: int = 100, which="train", **kwargs):
        super().__init__(**kwargs)
        self.dim = 3  # dimension of NZDem file. 
        self.n_points = n_points
        # Locations for all relevant parts of the dataset
        root_dir = Path("workdir/datasets/raw/nz_dem")
        if not root_dir.exists():
            print("Downloading and saving raw terrain data in ", root_dir)
            root_dir.mkdir(exist_ok=True, parents=True)
        raw_dataset_pth = root_dir / 'nz_dem_pointcloud' / ('per_sample=' + str(n_points) + '.npz') # Where the cleaned data will live
        raw_dataset_pth.parent.mkdir(exist_ok=True, parents=True)
        zip_file_pth = root_dir / 'nzdem-may-2020.zip' # where zip file will be saved
        tif_file_pth = root_dir / 'tifs' # where tif files are extracted to

        if raw_dataset_pth.exists():
            raw_data = np.load(raw_dataset_pth)
            inner = raw_data['pointsets']
        else: 
            download_from_url(self.url, zip_file_pth) # Download original zip from url
            extract_data_from_zip(zip_file_pth, root_dir / 'nzdem-zip') # Extract zip files from the dataset
            extract_nested_zips(root_dir / 'nzdem-zip', tif_file_pth) # Per tile dataset extraction

            # prepare data from rasterio 
            dataset, labels = self.prepare(tif_file_pth)
            np.savez(raw_dataset_pth, pointsets=dataset, labels=labels)
            print("cached dataset in:", raw_dataset_pth)
            inner = np.array(dataset)
        
        if which == "ood": 
            which = "val"
        is_train_or_val = which in ("train", "val") 
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=self.split_seed,
                    splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)
        inner = inner - inner.mean(dim=0, keepdim=True) # center dataset
        self.inner = torch.tensor(inner, dtype=torch.float32)
        

    def prepare(self, tif_file_path):
        """
        Download data and extract zip files
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        import rasterio
        from tqdm import tqdm

        files_list = sorted(Path(tif_file_path).iterdir())
        dataset = []
        labels = []

        def prepare_file(args):
            terrain_dir, seed = args
            rng = np.random.default_rng(seed)
            tif_file = next(terrain_dir.glob("*.tif"))
            label = tif_file.name[:2]
            pointsets = []

            with rasterio.open(tif_file) as ds:
                z = ds.read(1)
                res = ds.res
                nodata = ds.nodata
                mask = ds.read_masks(1) == 0 if nodata is None else (z == nodata)
                valid_rows, valid_cols = np.where(~mask)

                min_x, max_x = np.min(valid_cols), np.max(valid_cols)
                min_y, max_y = np.min(valid_rows), np.max(valid_rows)
                for _ in range(self.point_clouds_per_terrain):
                    valid_indices = np.empty(0, dtype=np.intp)
                    while valid_indices.size == 0:
                        x0, x1 = np.sort(rng.integers(min_x, max_x + 1, size=2))
                        y0, y1 = np.sort(rng.integers(min_y, max_y + 1, size=2))
                        window_mask = mask[y0:y1 + 1, x0:x1 + 1]
                        valid_indices = np.flatnonzero(~window_mask)
                    chosen = rng.choice(valid_indices,
                                        size=self.n_points,
                                        replace=True)
                    local_rows, local_cols = np.unravel_index(chosen, window_mask.shape)
                    sampled_rows = local_rows + y0
                    sampled_cols = local_cols + x0
                    subsample = np.column_stack((sampled_cols * res[0],
                                                 sampled_rows * res[1],
                                                 z[sampled_rows, sampled_cols])).astype(np.float32)
                    pointsets.append(subsample)

            return pointsets, [label] * len(pointsets)

        seeds = np.random.SeedSequence(self.seed).spawn(len(files_list))
        workers = max(1, min(len(files_list), os.cpu_count() or 1, 8))
        print("formatting datasets....")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = executor.map(prepare_file, zip(files_list, seeds))
            for pointsets, terrain_labels in tqdm(results, total=len(files_list)):
                dataset.extend(pointsets)
                labels.extend(terrain_labels)
        return dataset, labels
    
    def dataset_parameters(self):
        return {'dim': self.dim, 'n_points': self.n_points}
