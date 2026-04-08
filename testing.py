import yaml
from pathlib import Path
from nugets.models import BackBone, Model
import torch
import numpy as np
import matplotlib.pyplot as plt
from nugets.datasets.datapoint_types import *
from tqdm import tqdm
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
    for batch in tqdm(test):
        pred = model(batch).detach().numpy()
        distances = batch.distance.numpy()
        avg_relative_error = np.abs(pred - distances)/(distances + 1e-5)
    return np.mean(avg_relative_error), np.std(avg_relative_error), avg_relative_error


parser = argparse.ArgumentParser()
parser.add_argument("--yaml-dir", type=str)
parser.add_argument("--task-type", type=str)
parser.add_argument("--output-file", type=str)
args = parser.parse_args()

print("Loading all models in:", args.yaml_dir)
yaml_files = [f for f in Path(args.yaml_dir).iterdir() if f.is_file()]
all_models = []
for i in range(len(yaml_files)):
    cfg_file = yaml_files[i]
    
    model = Model.from_config_file(Path(cfg_file))
    dirname = model.get_dirname()
    ckpt_dir = f'workdir/models/{dirname}'
    if not Path(ckpt_dir).exists() or not any(Path(ckpt_dir).glob('*.ckpt')) :
        print("Experiment not finished:", cfg_file)
        continue
    print(ckpt_dir, '\n')
    print('checkpoint in', dirname)
    latest_ckpt = max(Path(ckpt_dir).glob("*.ckpt"), key=lambda f: f.stat().st_mtime)
    ckpt = torch.load(f'{latest_ckpt}')
    model.load_state_dict(ckpt['state_dict'])
    model.batch_size = 100000
    all_models.append(model)

cfg_pth = Path('config.yaml')
Config.load(cfg_pth)
config = Config.get()

with open(f'workdir/results/{args.output_file}.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    for i in range(len(yaml_files)):
        name = yaml_files[i].name[:-5]
        model = all_models[i]
        
        avg, std, all_err = get_mean_relative_error(model)
        print(name, avg, std)
        writer.writerow([name, all_err])
