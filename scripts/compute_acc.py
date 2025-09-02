import json
from collections import defaultdict
import numpy as np

def compute_vanilla_acc(data):
    accs = []
    for elem in data:
        mcqa_gt = elem['mcqa_gt']
        mcqa_pred = int(elem['mcqa_pred'])
        accs.append(int(mcqa_pred == mcqa_gt))
    return np.mean(accs)

def compute_macro_acc(data):
    rec_dict = defaultdict(list)
    for elem in data:
        mcqa_gt = elem['mcqa_gt']
        mcqa_pred = int(elem['mcqa_pred'])
        gt_word = elem['losses'][mcqa_gt]['prompt'].split('] ')[-1].strip()
        rec_dict[gt_word].append(int(mcqa_pred == mcqa_gt))
    
    macro_avg_acc = []
    for k in rec_dict:
        # print(f'{k} - {np.mean(rec_dict[k])}')
        macro_avg_acc.append(np.mean(rec_dict[k]))
    
    # print(f'Macro Average Accuracy: {np.mean(macro_avg_acc)}')
    return np.mean(macro_avg_acc)

if __name__ == "__main__":
    fp = '<path to the prediction file>'
    with open(fp, 'r') as f:
        data = json.load(f)
    print("Vanilla Accuracy: ", compute_vanilla_acc(data))
    print("Macro Average Accuracy: ", compute_macro_acc(data))
