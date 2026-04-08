import yaml
from pathlib import Path
from nugets.models import BackBone, Model
import torch
import numpy as np
import matplotlib.pyplot as plt
from nugets.datasets.datapoint_types import *
from tqdm import tqdm, trange
import csv
import argparse
from nugets.pipeline.configs import Config

def get_range_search_accuracy(model):
    return 

def get_query_accuracy(model):
    return

def get_set_summary_accuracy(model):

    return 

def get_mean_relative_error(model):
    """
    Get average relative error for distances
    """
    test = model.test_dataloader()
    for batch in test:
        pred = model(batch).detach().numpy()
        distances = batch.distance.numpy()
        avg_relative_error = np.abs(pred - distances)/(distances + 1e-5)
    return {'avg_re': np.mean(avg_relative_error),'std': np.std(avg_relative_error)}, pred

parser = argparse.ArgumentParser()
parser.add_argument("--yaml-dir", type=str)
parser.add_argument("--task-type", type=str)
parser.add_argument("--output-file", type=str)
args = parser.parse_args()

print("Loading all models in:", args.yaml_dir)
yaml_files = [f for f in Path(args.yaml_dir).iterdir() if f.is_file()]
all_models = []
modelnames = []
for i in trange(len(yaml_files)):
    cfg_file = yaml_files[i]
    
    model = Model.from_config_file(Path(cfg_file))
    dirname = model.get_dirname()
    ckpt_dir = f'workdir/server-workdir-0406/models/{dirname}'
    if not Path(ckpt_dir).exists() or not any(Path(ckpt_dir).glob('*.ckpt')):
        print("Experiment not finished:", cfg_file)
        continue
    latest_ckpt = max(Path(ckpt_dir).glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
    ckpt = torch.load(f'{latest_ckpt}')
    model.load_state_dict(ckpt['state_dict'])
    model.batch_size = 100000
    modelnames.append(cfg_file.name[:-5])
    all_models.append(model)

match args.task_type:
    case 'distances':
        test_metrics = get_mean_relative_error
    case 'queries':
        test_metrics = get_query_accuracy
    case 'summaries':
        test_metrics = get_set_summary_accuracy
    case other:
        raise NotImplementedError("Testing for task type has not been implemented yet")

cfg_pth = Path('config.yaml')
Config.load(cfg_pth)
config = Config.get()
with open(f'workdir/results/{args.output_file}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in trange(len(all_models)):
        name = yaml_files[i].name[:-5]
        model = all_models[i]
        name = modelnames[i]
        metrics, pred = test_metrics(model)
        writer.writerow([name, pred])