import pandas as pd
import torch
import torch.nn as nn
import torchvision
import scipy.io as sio 

import numpy as np
import glob
import os
import re

from scipy.stats.stats import pearsonr   
from vtk_io import read_vtk
from model import *
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

# ******************************************************
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
use_cuda = True
half = "all"

if use_cuda:
    device = torch.device('cuda:0')

batch_size = 1
in_channels = 9
out_channels = 36
learning_rate = 0.001
momentum = 0.99
weight_decay = 0.0001
fold = 1
# ******************************************************


def gender2bin(x):
    return 1 if x.lower() == "male" else 0

def single2bin(x):
    return 1 if x.lower() == "single" else 0

raw_data = pd.read_csv('../label.txt', sep="\t", header=None).drop(index=0)
label_data = pd.DataFrame()
label_data["session"] = raw_data[1]
label_data["gender"] = [gender2bin(x) for x in raw_data[2]]
label_data["y"] = raw_data[6]



feature_names = ["sulc", "thickness", "curv", "par_fs", "vertexArea", "EucDepthExTh", "LGI60ExTh"]


class BrainSphere(torch.utils.data.Dataset):

    def __init__(self, folds):
        self.files = []
        self.sessions = []
        self.map = {}
        for fold in folds:
            self.files += sorted(glob.glob(os.path.join(fold, '*.vtk')))
        for f in self.files:
            session = re.findall(r'ses-([0-9]+?)_', f)[0]
            if session not in self.sessions:
                self.sessions.append(session)
                self.map[session] = [f]
            else:
                self.map[session].append(f)
        self.files = [session for _, session in self.map.items()]
            


    def __getitem__(self, index):
        l_r = self.files[index]
        
        session1 = re.findall(r'ses-([0-9]+?)_', l_r[0])[0]
        session2 = re.findall(r'ses-([0-9]+?)_', l_r[1])[0]
        assert session1 == session2

        row = label_data.loc[label_data['session'] == session1]
        
        feats_lr = []
        for file in l_r:
            data = read_vtk(file)
            feats = [data[feat] for feat in feature_names]
            feats = np.asarray(feats).T
            feat_max = np.max(feats,0)
            for i in range(np.shape(feats)[1]):
                if feat_max[i] != 0:
                    feats[:,i] = feats[:, i]/feat_max[i]
            feats_lr.append(feats)
        label = np.asarray([row["y"].to_numpy()[0], row["gender"].to_numpy()[0]])
        # label = np.asarray(row["y"].to_numpy()[0])

        # gender = np.asarray([row["gender"].to_numpy()[0]] * num_points).reshape(num_points, 1)
        # singleton = np.asarray([row["singleton"].to_numpy()[0]] * num_points).reshape(num_points, 1)

        # difference = feats_lr[0]-feats_lr[1]
        feats = np.concatenate((feats_lr[0], feats_lr[1]), axis = 1)
        return feats.astype(np.float32), label.astype(np.float32)
            
    def __len__(self):
        return len(self.files)

class BrainSphereHalf(torch.utils.data.Dataset):
    def __init__(self, folds):  
        pass

    def __getitem__(self, index):
        file = self.files[index]
        id = re.findall(r'sub-([\x00-\x7F]+?)_ses', file)[0]
        row = label_data.loc[label_data['id'] == id]
        data = read_vtk(file)
        feats = [data[feat] for feat in feature_names]
        num_points = len(feats[0])
        feats.append([row["gender"].to_numpy()[0]] * num_points)
        feats.append([row["singleton"].to_numpy()[0]] * num_points)
        feats = np.asarray(feats).T
        feat_max = np.max(feats,0)
        for i in range(np.shape(feats)[1]):
            if feat_max[i] != 0:
                feats[:,i] = feats[:, i]/feat_max[i]
    
        label = np.asarray(row["y"].to_numpy()[0])
        return feats.astype(np.float32), label.astype(np.float32)


    def __len__(self):
        return len(self.files)


def compute_abs_error(prediction, target):
    return torch.abs(prediction-target)


def multitask_train_step(data, target):
    model.train()
    if use_cuda:
        data, target = data.cuda(device), target.cuda(device)

    prediction = model(data)
    loss1 = age_criterion(prediction[0], target[0])
    
    loss2 = gender_criterion(prediction[1:3].unsqueeze(0), target[1].unsqueeze(0).long())

    optimizer.zero_grad()
    loss1.backward(retain_graph=True)
    loss2.backward()
    optimizer.step()

    return [loss1.item(), loss2.item()]

def train_step(data, target):
    model.train()
    if use_cuda:
        data, target = data.cuda(device), target.cuda(device)

    prediction = model(data)
    loss = age_criterion(prediction, target)
    

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()



def multitask_val_during_training(dataloader, train=True):
    abs_error_age = np.zeros(len(dataloader))
    if not train:
        targets = np.zeros(len(dataloader))
        ages = np.zeros(len(dataloader))
        
    gender_accuracy = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze()
        target = target.squeeze()
        
        data = data.squeeze()
        if use_cuda:
            data, target = data.cuda(device), target.cuda(device)

        with torch.no_grad():
            prediction = model(data)
        abs_error_age[batch_idx] = compute_abs_error(prediction[0], target[0])
        if torch.argmax(prediction[1:3]) == target[1]:
            gender_accuracy += 1
        if not train:
            targets[batch_idx] = target[0]
            ages[batch_idx] = prediction[0]
    gender_accuracy = float(gender_accuracy)/float(len(dataloader))

    if not train:
        print("**********************************")
        print("**********************************")
        print("Correlation: ", pearsonr(targets, ages)[0])
        print("**********************************")
        print("**********************************")
        with open("correlation", "w+") as f:
            for target in targets:
                f.write('%s ' % target)
            f.write("\n")
            for age in ages:
                f.write('%s ' % age)

    return [abs_error_age, gender_accuracy]


def val_during_training(dataloader, train = True):
    abs_error = np.zeros(len(dataloader)) if not train else np.zeros(10) 
    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.squeeze()
        target = target.squeeze()
        if train and batch_idx == 10:
            break
        data = data.squeeze()
        if use_cuda:
            data, target = data.cuda(device), target.cuda(device)

        with torch.no_grad():
            prediction = model(data)
        abs_error[batch_idx] = compute_abs_error(prediction, target)

    return abs_error

# fold1_l = "../data/lh_fold1"
# fold2_l = "../data/lh_fold2"
# fold3_l = "../data/lh_fold3"
# fold4_l = "../data/lh_fold4"
# fold5_l = "../data/lh_fold5"

# fold1_r = "../data/rh_fold1"
# fold2_r = "../data/rh_fold2"
# fold3_r = "../data/rh_fold3"
# fold4_r = "../data/rh_fold4"
# fold5_r = "../data/rh_fold5"

fold1 = "../data/fold1"
fold2 = "../data/fold2"
fold3 = "../data/fold3"
fold4 = "../data/fold4"
fold5 = "../data/fold5"

print("Setting up data loader...")


# if half == "lh":
#     train_fold = [fold2_l, fold3_l, fold4_l]
#     val_fold = [fold1_l]
#     test_fold = [fold5_l]
# elif half == "rh":
#     train_fold = [fold2_r, fold3_r, fold4_r]
#     val_fold = [fold1_r]
#     test_fold = [fold5_r]
# else:
train_fold = [fold2, fold3, fold4]
val_fold = [fold1]
test_fold = [fold5, fold5]


train_dataset = BrainSphere(train_fold)
val_dataset = BrainSphere(val_fold)
test_dataset = BrainSphere(test_fold)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)



print("Finish loading data")

in_channels = 7 if half == "all" else 9



# weight_decays_range = [np.random.uniform(0.01, 0.0001) for i in range(4)]
# factor_range = [np.random.uniform(0, 1) for i in range(4)]
# dropout_range = [np.random.uniform(0, 1) for i in range(3)]


for model_count in range(1):

        ########## vgg fine tuning ############
        # factor = np.random.uniform(0.475, 0.6)
        # weight_decay = np.random.uniform(0.01, 0.006)

        ########## Resnet fine tuning ###########
        factor = np.random.uniform(0, 1)
        weight_decay = np.random.uniform(0, 0.01)


        # model = ResNet(in_channels, out_channels)
        # model_name ="rh_ResNet_factor_" + str(factor)[0:5] + "_weight_decay_" + str(weight_decay)[0:5]
        model = Multitask_vgg16_Final_Regression(in_channels)
        model_name = "Multitask_vgg16_Final_Regression_factor_" + str(factor)[0:5] + "_weight_decay_" + str(weight_decay)[0:5]
        # model = DenseNet(in_channels)
        # model_name = "DenseNet_default"

        print("Model_name: ", model_name)


        if use_cuda:
            # print("Memory used: ", torch.cuda.memory_allocated(device))
            # print("Max Memory allowed: ", torch.cuda.max_memory_allocated(device))
            model.cuda(device)
            
        age_criterion = nn.MSELoss()
        gender_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=factor, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', min_lr=1e-6)


        print("Start training")

        train_loss = [0,0,0,0,0]
        train_history = []
        val_history = []
        for epoch in range(100):
            
            train_error = multitask_val_during_training(train_dataloader)
            scheduler.step(np.mean(train_error[0]))

            train_history.append(np.sum(np.square(train_error[0])))
            # print("Training_error: ", train_error)

            # val_error = multitask_val_during_training(val_dataloader)    
            # val_history.append(np.sum(np.square(val_error[0])))
            # print("Val_error: ", val_error)


            for batch_idx, (data, target) in enumerate(train_dataloader):
                # print("epoche: {}, batch: {}", (epoch, batch_idx))
                data = data.squeeze()
                target = target.squeeze()
                loss = multitask_train_step(data, target)
                # writer.add_scalar('Train/Loss', loss, epoch*len(train_dataloader) + batch_idx)  


            train_loss[epoch % 5] = np.mean(train_error[0])
            # print("last five train absolute error: ",train_loss)
            if np.std(np.array(train_loss)) <= 0.00001:
                torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+"_final.pkl"))
                break

            torch.save(model.state_dict(), os.path.join('trained_models', model_name+'_'+str(fold)+".pkl"))

        print("Final training error: ", np.mean(train_error[0]))
        print("Final training accuracy: ", train_error[1])
        # print("Final validation error: ", np.mean(val_error[0]))
        # print("Final validation accuracy: ", val_error[1])
        print("Finish Training. ")

        test_error = multitask_val_during_training(test_dataloader, train=False)
        print("**********************************")
        print("**********************************")
        print("Test Error: ", np.mean(test_error[0]))
        print("Test accuracy: ", test_error[1])
        print("**********************************")
        print("**********************************")

        # print("Save files...")

        # train_history_path_name = "history/train_history_" + model_name
        # val_history_path_name = "history/val_history_" + model_name

        # if os.path.exists(train_history_path_name):
        #     os.remove(train_history_path_name)
        # with open(train_history_path_name, "w+") as f:
        #     for hist in train_history:
        #         f.write('%s\n' % hist)

        # if os.path.exists(val_history_path_name):
        #     os.remove(val_history_path_name)
        # with open(val_history_path_name, "w+") as f:
        #     for hist in val_history:
        #         f.write('%s\n' % hist)

        # print("Finish saving files")