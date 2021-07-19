import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#####################
half = ["lh", "rh"]
plot_type = "histogram"
#####################

paths =["../data/fold" + str(i+1) for i in range(5)]
paths.append("../data/all")

raw_data = pd.read_csv('../label.txt', sep="\t", header=None).drop(index=0)
label_data = pd.DataFrame()
label_data["session"] = raw_data[1]
label_data["y"] = raw_data[6]

distributions = []

for path in paths:
    files = sorted(glob.glob(os.path.join(path, '*.vtk')))  
    distribution = []
    for f in files: 
        session = re.findall(r'ses-([0-9]+?)_', f)[0]
        row = label_data.loc[label_data['session'] == session]
        label = np.asarray(row["y"].to_numpy()[0])
        label = label.astype(np.float32)
        distribution.append(label)
    distribution = np.asarray(distribution)
    distributions.append(distribution)

if plot_type == "boxplot":
    fig1, ax1 = plt.subplots()
    ax1.boxplot(distributions)
    plt.show()

if plot_type == "histogram":
    x = [i for i in range(29, 46)]
    y = [[0 for i in range(29, 46)] for j in range(len(distributions))]
    for index, distribution in enumerate(distributions):
        for age in distribution:
            y[index][int(age-29)] += 1
        plt.subplot(6, 2, index+1)
        plt.bar(x, y[index])
        plt.ylim(0, 2)
        plt.title(paths[index], {"fontsize":10}, pad=10)

    plt.show()
        
    