import os
import glob
import numpy as np
import csv
import pandas as pd

def _process_anotation(file_name):
    file_name = file_name[:-5]+'cpos.txt'
    with open(file_name, 'r') as f:
        label = list(map( 
            lambda coordinate: float(coordinate), f.read().strip().split()))
    return label

def _create_list_paths(path2data):
    folders = range(1, 11)
    folders = ['0' + str(i) if i < 10 else '10' for i in folders] 
    list_paths = []
    for folder in folders:
        for name in glob.glob(os.path.join(path2data, folder, 'pcd' + folder + '*r.png')):
            ab_path = os.path.abspath(name)
            list_paths.append(ab_path)
    print(len(list_paths))
    print(folders)
    return list_paths

def make_dataset():
    list_paths = _create_list_paths("../../data/interim/grasp/")
    labels = []
    for path in list_paths:
        label = _process_anotation(path)
        label_np = np.array(label)
        label_reshaped = label_np.reshape(-1, 8)
        label_clean = []
        for box in label_reshaped:
            if not np.isnan(box).any():
                label_clean.append(box.tolist())
        label_clean = np.array(label_clean).reshape(-1).tolist()
        labels.append(label_clean)

    # create csv file that has 2 colum : path2image and label_clean.
    columns = {}
    columns["path2image"] = list_paths
    columns["label"] = labels
    data = list(zip(columns["path2image"], columns["label"]))
    df = pd.DataFrame(data = data)
    df.to_csv("../../data/processed/grasp.csv", index=False, header=["path", "label"])
    
    
if __name__ == "__main__":
    make_dataset()
        
    
