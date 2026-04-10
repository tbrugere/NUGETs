import yaml
from pathlib import Path
from nugets.models import BackBone, Model
import torch
import numpy as np
from nugets.datasets.datapoint_types import *
from tqdm import tqdm, trange
import argparse
from nugets.pipeline.configs import Config
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import re
from torch_geometric.utils import softmax, unbatch
import json 

def last_version(path: Path) -> int:
    m = re.fullmatch(r"last(?:-v(\d+))?\.ckpt", path.name)
    if not m:
        return -1
    return int(m.group(1) or 0)

def get_range_search_accuracy(model, t=0.5, **kwargs):
    """
    Get prediction metrics for range search queries
    Namely, F1 score, accuracy, precision, and accuracy for each 
    pair. 
    """
    test = model.test_dataloader()
    for batch in test:
        logits = model(batch)
        out = torch.sigmoid(logits)
        probs = out.detach().numpy()
        labels = batch.label
        predictions = (probs >= t).int()
    metrics = {'f1': f1_score(labels, predictions, zero_division=1.0), 
               'accuracy': accuracy_score(labels, predictions), 
               'precision': precision_score(labels, predictions), 
               'recall': recall_score(labels, predictions)}
    return metrics, probs

def _get_nearest_neighbor_ranking(query, pointset):
    distances = np.linalg.norm(pointset - query, axis=1)
    ground_truth_ranking = np.argsort(distances)[::-1]
    return ground_truth_ranking

def _get_extremal_point_ranking(query, pointset):
    projections = pointset @ query
    ground_truth_ranking = np.argsort(projections)[::-1]
    return ground_truth_ranking

def hit_rates_up_to_k(ranking, target_id, k):
    """
    Returns the hit rate at each cutoff from 1 to k.

    Example:
    ranking = [7, 3, 9, 2], target_id = 9, k = 4
    returns [0.0, 0.0, 1.0, 1.0]
    """
    out = np.zeros(k)
    hit = 0.0
    for i in range(k):
        if ranking[i] == target_id:
            hit = 1.0
        out[i] = hit
    return out

def get_query_accuracy(model,ranking_func, k=50, **kwargs):
    """
    Return hit rates up to k 
    """
    test = model.test_dataloader()
    predictions = []
    pointsets = []
    queries = []
    for batch in test:
        logits = model(batch)
        out = softmax(src=logits, index = batch.pointset.batch)
        unbatched_model_output = unbatch(out, batch.pointset.batch)
        predictions.extend([out.detach().numpy() for out in unbatched_model_output])
        queries.extend([q.numpy() for q in batch.queryset])
        unbatch_pointset = unbatch(batch.pointset.data, batch.pointset.batch)
        pointsets.extend([p.numpy() for p in unbatch_pointset])
    hit_rates = []
    for i in range(len(queries)):
        q = queries[i]
        pset = pointsets[i]
        predictions = predictions[i]
        predicted_target_id = np.argmax(predictions[i])
        ranking = ranking_func(q, pset)
        hit_rate = hit_rates_up_to_k(ranking, predicted_target_id, k=k)
        hit_rates.append(hit_rate)
    metrics = {'recall@5':hit_rates[4], 'recall@10': hit_rates[10]}
    return metrics, hit_rates

def get_set_summary_accuracy(model, t=0.5, **kwargs):
    """
    Test metrics for set summary tasks such as Convex Hulls and 
    Alpha Shapes
    """
    test = model.test_dataloader()

    for batch in test:
        logits = model(batch).data.detach().squeeze()
        expits = torch.sigmoid(logits).numpy()
        predicted = (expits >= t)
        start = 0
        f1 = []
        acc = []
        pre = []
        recall = []
        for num in batch.labelset.n_nodes:
            end = start + num
            pred_ = predicted[start:end]
            label_ = batch.labelset.data[start:end]
            f1.append(f1_score(label_,pred_, zero_division=1.0))
            acc.append(accuracy_score(label_, pred_))
            pre.append( precision_score(label_, pred_))
            recall.append( recall_score(label_, pred_))
    metrics = {'f1': (np.mean(f1), np.std(f1)), 
               'accuracy':(np.mean(acc), np.std(acc)), 
               'precision':(np.mean(pre), np.std(pre)),
               'recall': (np.mean(recall), np.std(recall))}
    return metrics, expits

def get_mean_relative_error(model, **kwargs):
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
    ckpt_dir = f'workdir/models/{dirname}'
    if not Path(ckpt_dir).exists() or not any(Path(ckpt_dir).glob('*.ckpt')) :
        print("Experiment not finished:", cfg_file)
        continue
    print(ckpt_dir, '\n')
    print('checkpoint in', dirname)
    latest_ckpt = max(Path(ckpt_dir).glob("*.ckpt"), key=last_version)
    ckpt = torch.load(f'{latest_ckpt}')
    model.load_state_dict(ckpt['state_dict'])
    model.batch_size = 100000
    modelnames.append(cfg_file.name[:-5])
    all_models.append(model)
ranking_func = None
match args.task_type:
    case 'distances':
        test_metrics = get_mean_relative_error
    case 'nearest-neighbor':
        test_metrics = get_query_accuracy
        ranking_func =_get_nearest_neighbor_ranking
    case 'extremal-point':
        test_metrics = get_query_accuracy
        ranking_func = _get_extremal_point_ranking
    case 'range-search':
        test_metrics = get_range_search_accuracy
    case 'summaries':
        test_metrics = get_set_summary_accuracy
    case other:
        raise NotImplementedError("Testing for task type has not been implemented yet")

cfg_pth = Path('config.yaml')
Config.load(cfg_pth)
config = Config.get()
results = {}
for i in trange(len(all_models)):
        name = yaml_files[i].name[:-5]
        model = all_models[i]
        
        name = modelnames[i]
        metrics, pred = test_metrics(model, ranking_func=ranking_func)
        print(name, metrics)
        results[name]{'metrics': pred, 'output': pred}

with open(f'workdir/results/{args.output_file}', 'w') as f:
    json.dump(results, f)
        
    