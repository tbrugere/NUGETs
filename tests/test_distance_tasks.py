from nugets.datasets.geometric_shapes import GrowingCircles
from nugets.tasks.distance_tasks import HausdorffDistanceTask
from nugets.tasks.distance_tasks import FrechetDistanceTask

from scipy.spatial.distance import directed_hausdorff
from shapely import LineString
from shapely import frechet_distance
import numpy as np


def test_hausdorf_distance_task():
    hausdorf = HausdorffDistanceTask(
        "GrowingCircles", 
            dict(
                dim=2,
                min_points=10,
                max_points=20,
                radius="linear", 
                length= 64, 
                )
        )

    dataset = hausdorf.get_dataset("train")[0]
    set1, set2 = dataset.set1, dataset.set2
    result = hausdorf.distance(set1, set2)

    assert isinstance(result, (int, float)) and result >= 0


def test_frechet_distance_task():
    frechet = FrechetDistanceTask(
        "GrowingCircles", 
            dict(
                dim=2,
                min_points=10,
                max_points=20,
                radius="linear", 
                length= 64, 
                )
        )

    dataset = frechet.get_dataset("train")[0]
    set1, set2 = dataset.set1, dataset.set2
    result = frechet.distance(set1, set2)

    assert isinstance(result, (int, float)) and result >= 0