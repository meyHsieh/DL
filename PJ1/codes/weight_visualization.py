# weight_visualization.py

import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle


model = nn.models.Model_MLP()
model.load_model(r'.\saved_models\MLP_baseline.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
preds = np.argmax(logits, axis=-1)

print(f"Test Accuracy: {nn.metric.accuracy(logits, test_labs):.4f}")

#confusion matrix 
def plot_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')
    
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max()/2 else "black",
                          fontsize=10)
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix (MLP Baseline)', fontsize=14)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels([str(i) for i in range(num_classes)])
    ax.set_yticklabels([str(i) for i in range(num_classes)])
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(r'.\Figs\MLP_Figure\confusion_matrix.png', dpi=150)
    plt.show()
    
    return cm

print("\nConfusion Matrix")
cm = plot_confusion_matrix(test_labs, preds)

for i in range(10):
    total = cm[i].sum()
    correct = cm[i, i]
    error_rate = (total - correct) / total * 100
    print(f"Digit {i}: {correct}/{total} correct, Error rate: {error_rate:.2f}%")

#Misclassified Examples
def plot_misclassified_examples(images, labels, preds, num_examples=20):
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    
    worst = max(range(10), key=lambda i: 1 - cm[i,i]/cm[i].sum() if cm[i].sum() > 0 else 0)
    
    cm[worst, worst] = 0
    confused = np.argmax(cm[worst])
    
    idx = np.where((labels == worst) & (preds == confused))[0]
    selected = np.random.choice(idx, min(num_examples, len(idx)), replace=False)
    
    rows = (len(selected) + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(12, 2.5*rows))
    for i, ax in enumerate(axes.flat):
        if i < len(selected):
            ax.imshow(images[selected[i]].reshape(28,28), cmap='gray')
            ax.set_title(f'{worst}->{confused}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(r'.\Figs\MLP_Figure\misclassified.png', dpi=150)
    plt.show()

print("\nMisclassified Examples")
plot_misclassified_examples(test_imgs, test_labs, preds, num_examples=20)