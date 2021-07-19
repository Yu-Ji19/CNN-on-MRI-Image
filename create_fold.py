import os
import shutil
import glob
import random
import re
import numpy as np
import pandas as pd

paths =["../data/fold" + str(i+1) for i in range(5) ]
data_path = "../data/all"
label_path = "../label.txt"

sessions = []
map = {}

def move(session, i):
    for f in map[session]:
        shutil.copy(f, paths[i-1])

for path in paths:
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

files = sorted(glob.glob(os.path.join(data_path, '*.vtk'))) 



for f in files:
    session = re.findall(r'ses-([0-9]+?)_', f)[0]
    if session not in sessions:
        sessions.append(session)
        map[session] = [f]
    else:
        map[session].append(f)

raw_data = pd.read_csv(label_path, sep="\t", header=None).drop(index=0)
label_data = pd.DataFrame()

label_data["session"] = raw_data[1]
label_data["y"] = raw_data[6]

labels = [[] for i in range(17)]

for session in sessions:
    row = label_data.loc[label_data['session'] == session]
    label = float(row["y"].to_numpy()[0])
    labels[int(label)-29].append(session)

fold = 1

for i in range(len(labels)):
    bin = labels[i]
    while len(bin)>0:
        session = random.choice(bin)
        move(session, fold)
        fold = (fold + 1) % 5 + 1
        bin.remove(session)


    


