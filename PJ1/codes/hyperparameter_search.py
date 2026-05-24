# you may do your own hyperparameter search job here.
import mynn as nn
from draw_tools.plot import plot
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
import os

np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)

with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

with open('idx.pickle', 'rb') as f:
    idx = pickle.load(f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000] / 255.0
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:] / 255.0
train_labs = train_labs[10000:]

lr_list = [0.1, 0.01, 0.001]
results = []

print(f"{'Learning Rate':<15} | {'Best Dev Accuracy':<15}")
print("-" * 35)

for lr in lr_list:
    linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU')
    
    optimizer = nn.optimizer.SGD(init_lr=lr, model=linear_model)
    loss_fn = nn.op.MultiCrossEntropyLoss(model=linear_model, max_classes=train_labs.max()+1)
    
    runner = nn.runner.RunnerM(linear_model, optimizer, nn.metric.accuracy, loss_fn)
    
    print(f"\n[Training]MLP_{lr}")
    runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=f'./saved_models/MLP_Lr_{lr}')
    
    _, axes = plt.subplots(1, 2)
    plot(runner, axes)
    plt.show()

    results.append((lr, runner.best_score))
    print(f"[Done] LR: {lr}, Best Acc: {runner.best_score:.4f}")

    
print("\n" + "="*35)
print(f"{'Learning Rate':<15} | {'Best Dev Acc':<15}")
print("-" * 35)
for lr, acc in results:
    print(f"{lr:<15} | {acc:<15.4f}")
print("="*35)