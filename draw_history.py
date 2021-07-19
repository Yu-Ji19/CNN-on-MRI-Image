import matplotlib.pyplot as plt

# One.png: names = ["original", "linear_decay_no_reg", "linear_decay_0.0001reg", "linear_decay_0.001reg"]
# Two.png: names = ["vgg16_original", "vgg_0.001threshold", "ResNet", "vgg16_factor_0.5", "ResNet_factor_0.2", "ResNet_factor_0.5_threshold_0.001", "vgg16_factor_0.5_threshold_0.001"]
# Three.png: names = ["vgg16_factor_0.5", "vgg16_factor_0.5_min_1e-7", "vgg16_factor_0.8"]
# Four.png: names = ["DenseNet_default", "ResNet_factor_0.2_weight_decay_0.001", "ResNet_factor_0.5_min_5e-6_learning_decay_0.001", "ResNet_factor_0.5_weight_decay_0.001"]
# Five.png: names = ["ResNet_factor_0.2_learning_decay_0.001", "ResNet_factor_0.2_min_5e-6", "ResNet_factor_0.5", "ResNet_factor_0.5_min_5e-6", "ResNet_factor_0.5_min_5e-6_learning_decay_0.05", "vgg16_factor_0.5_min_5e-6"]
# Six.png: names = ["lh_ResNet_factor_0.2_weight_decay_0.001", "rh_ResNet_factor_0.2_weight_decay_0.001", "lh_vgg16_factor_0.2", "lh_vgg16_factor_0.5", "vgg16_factor_0.5", "ResNet_factor_0.2_weight_decay_0.001_Two_halves", "vgg16_factor_0.5_Two_halves"]
# Seven.png: names = ["ResNet_factor_0.2_weight_decay_0.001_Independent_10sample", 
#             "ResNet_factor_0.2_weight_decay_0.001_Two_halves", 
#             "vgg16_factor_0.5_Independent", 
#             "vgg16_factor_0.5_Two_halves",
#             "vgg16_factor_0.5_Two_halves_0.001_reg",
#             "vgg16_Dropout_0.5_factor_0.5_Two_halves_0.001_reg",
#             "vgg16_Dropout_0.2_factor_0.5_Two_halves_0.001_reg",
#             "vgg16_Dropout_0.1_factor_0.5_Two_halves_0.001_reg"
#             ]
# Eight.png: names = ["vgg16_factor_0.2_lh",
#         "vgg16_factor_0.5_lh",
#         "lh_ResNet_factor_0.2_weight_decay_0.001",
#         "vgg16_factor0.909_rh",
#         "vgg16_factor0.432_rh",
#         "vgg16_factor0.829_rh",
#         "vgg16_factor0.112_rh",
#         "vgg16_factor0.166_lh",
#         "vgg16_factor0.008_lh",
#         "vgg16_factor0.806_lh",
#         "vgg16_factor0.536_lh"
#         ]

names = []

# Best: 
#   VGG - factor 0.5
#   ResNet: ResNet_factor_0.2_weight_decay_0.001
#   DenseNet: 


i = 0
for name in names:
    i += 1
    train_path = "history/train_history_"+name
    val_path = "history/val_history_"+name

    train_history = []
    val_history = []

    with open(train_path, "r") as f:
        for line in f:
            line = line[:-1]
            train_history.append(float(line))

    with open(val_path, "r") as f:
        for line in f:
            line = line[:-1]
            val_history.append(float(line))

    train_history = train_history[1:]
    val_history = val_history[1:]

    x = range(1,100)
    plt.subplot(3, 4, i)
    plt.ylim(0,25)
    plt.plot(x, train_history)
    plt.plot(x, val_history)
    plt.legend(["train", "val"])
    plt.title(names[i-1], {"fontsize":10}, pad=10)

    

plt.show()
