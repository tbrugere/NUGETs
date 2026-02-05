from typing import Literal
import numpy as np
import torch

from ml_lib.datasets import Dataset
from ml_lib.datasets.splitting import SplitTransform

from nugets.datasets.register import register as dataset_register
from .datapoint_types import Set_datapoint
from .dataset_utils import download_from_url
import rasterio

@dataset_register 
class NZDEM(Dataset[Set_datapoint]):
    datatype: Set_datapoint
    url: str = 'https://storage.googleapis.com/www-ajnisbet-com/nzdem-may-2020.zip'
    seed: int = 42  # random seed set for sampling from point cloud
    point_clouds_per_terrain: int = 100

    def __init__(self, n_points: int =100, which="train", **kwargs):
        super().__init__(**kwargs)
        # Locations for all relevant parts of the dataset
        root_dir = Path("workdir/datasets/raw/nz_dem")
        root_dir.mkdir(exists_ok=True, parents=True)
        raw_dataset_pth = root_dir / 'nz_dem_pointcloud' / str(n_points) + '.npz' # Where the cleaned data will live
        zip_file_pth = root_dir / 'nzdem-may-2020.zip' # where zip file will be saved
        tif_file_pth = root_dir / 'tifs' # where tif files are extracted to

        if raw_dataset_pth.exists():
            raw_data = np.load(raw_dataset)
            inner = raw_data['pointsets']
        else: 
            download_from_url(zip_file_pth)
            extract_nested_zips(zip_file_pth, tif_file_pth)
            dataset, labels = self.prepare(tif_file_pth)
            np.savez(raw_dataset_pth, pointsets=dataset, labels=labels)
            print("cached dataset in:", raw_dataset_pth)
            inner = np.array(dataset)
        
        if which == "ood": 
            which = "val"
        is_train_or_val = which in ("train", "val") 
        if is_train_or_val:
            split_transform: SplitTransform = SplitTransform(
                    which=which, seed=split_seed, 
                    splits=["train", "val"], percents=[.9, .1])
            inner = split_transform(inner)
        self.inner = torch.tensor(inner, dtype=torch.float32)

    def prepare(self, tif_file_pth):
        """
        Download data and extract zip files
        """
        rng = np.random.default_rng(self.seed)
        files_list = [p for p in Path(directory_path).iterdir()]
        dataset = []
        labels = []
        for f in files_list:
            # get the tif file
            inner_files = [file_path for file_path in directory_path.glob(".tif")]
            with rasterio.open(f) as ds:
                z = ds.read(1)
                res = ds.res
                nodata = ds.nodata
                mask = (z == nodata)
                rows, cols = np.where(~mask)
                zs = z[rows, cols].astype(np.float32)
                all_pts = np.column_stack([cols * res[0], rows * res[1], zs])

                min_x, max_x = np.min(cols), np.max(cols)
                min_y, max_y = np.min(rows), np.max(rows)
                for _ in range(self.point_clouds_per_terrain):
                    num_candidates = 0
                    while num_candidates == 0:
                        x0, x1 = np.sort(rng.integers(min_x, max_x, size=2, replace=False))
                        y0, y1 = np.sort(rng.integers(min_y, max_y, size=2, replace=False))
                        in_chunk = (
                            (rows >= y0) & (rows < r0 + chunk) &
                            (cols >= c0) & (cols < c0 + chunk)
                        )
                        np.candidates = np.sum(in_chunk)
                    
                    subsample = rng.choice(all_pts[in_chunk], size=self.n_points)
                    dataset.append(subsample)
        return dataset, labels
    
    def dataset_parameters(self):
        return {'dim': self.dim, 'n_points': self.n_points}
