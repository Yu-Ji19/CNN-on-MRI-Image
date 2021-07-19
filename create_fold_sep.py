import os
import shutil
import glob
import random
import re
import pandas as pd

half = ["lh", "rh"]

paths =["../data/"+ h + "_fold" + str(i+1) for i in range(5) for h in half]
data_path = "../data/all"
label_path = "../label.txt"

for path in paths:
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

files = sorted(glob.glob(os.path.join(data_path, '*.vtk'))) 
ids = []
for f in files:
    id = re.findall(r'sub-([\x00-\x7F]+?_ses-[0-9A-Za-z_]+?\.)', f)[0]
    if id not in ids:
        ids.append(id)

raw_data = pd.read_csv(label_path, sep="\t", header=None).drop(index=0)
label_data = pd.DataFrame()
label_data["id"] = raw_data[0]
label_data["y"] = raw_data[6]


count = int(len(ids)/5)
for i in range(5):
    for j in range(count):
        id_choice = random.choice(ids)
        lh_file = "".join(["../data/all/sub-", id_choice, "lh.RegSphereSurf.WithROI.Resample.ico5.vtk"])
        rh_file = "".join(["../data/all/sub-", id_choice, "rh.RegSphereSurf.WithROI.Resample.ico5.vtk"])
        shutil.copy(lh_file, "../data/lh_fold" + str(i+1))
        shutil.copy(rh_file, "../data/rh_fold" + str(i+1))
        ids.remove(id_choice)

for id in ids:
    lh_file = "".join(["../data/all/sub-", id, "lh.RegSphereSurf.WithROI.Resample.ico5.vtk"])
    rh_file = "".join(["../data/all/sub-", id, "rh.RegSphereSurf.WithROI.Resample.ico5.vtk"])
    shutil.copy(lh_file, "../data/lh_fold5")
    shutil.copy(rh_file, "../data/rh_fold5")