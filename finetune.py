import re
import matplotlib.pyplot as plt
import numpy as np

f = open("./output/output2", "r").read()

names = re.findall(r'Model_name:  ([A-Za-z_\.0-9]+?)\n', f)
factors = [float(re.findall(r'factor([\.0-9]+?)_', name)[0]) for name in names]
weight_decays = [float(re.findall(r'weight_decay_([\.0-9]+)', name)[0]) for name in names]
testerrors = [float(error) for error in re.findall(r'Test Error:  ([\.0-9]+?)\n', f)]
print(factors)
print(weight_decays)
print(testerrors)
mean = np.mean(testerrors)
std = np.std(testerrors)

errors = [(error - mean)/std*10 for error in testerrors]

errors = [error - min(errors) for error in errors]

plt.scatter(factors, weight_decays, s=errors)
plt.show()