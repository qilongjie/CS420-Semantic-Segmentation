import os.path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import Unet
from Dataset import ISBIDataset
import torch.nn.functional

ROOT = os.path.dirname(os.path.abspath('__file__'))

TRAIN_IMGS_DIR = os.path.join(ROOT, 'dataset/train_img')
TRAIN_LABELS_DIR = os.path.join(ROOT, 'dataset/train_label')
TEST_IMGS_DIR = os.path.join(ROOT, 'dataset/test_img')
TEST_LABELS_DIR = os.path.join(ROOT, 'dataset/test_label')
OUTPUT_DIR = os.path.join(ROOT,'output_lr=0.002')
#parameters
batch_size = 2
learn_rate = 0.002
num_epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = ISBIDataset(imgs_dir=TRAIN_IMGS_DIR, labels_dir=TRAIN_LABELS_DIR, flip=True)
test_dataset = ISBIDataset(imgs_dir=TEST_IMGS_DIR, labels_dir=TEST_LABELS_DIR, flip=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

train_num = len(train_dataset)
test_num = len(test_dataset)
train_batch = int(np.ceil(train_num / batch_size))
test_batch = int(np.ceil(test_num / batch_size))

model = Unet().to(device)
loss = nn.BCEWithLogitsLoss().to(device)
optim = torch.optim.Adam(params=model.parameters(), lr=learn_rate)

# train
for epoch in range(1,num_epochs+1):
    model.train()

    for batch_idx, data in enumerate(train_loader, 1):
        img = data['img'].to(device)
        label = data['label'].to(device)
        output = model(img)

        optim.zero_grad()
        err = loss(output, label)
        err.backward()
        optim.step()

        print('[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'.format(epoch, num_epochs,
                                                                                                    batch_idx,
                                                                                                    train_batch,
                                                                                                    err.item()))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
#save the final model    
torch.save(
    {'net': model.state_dict(),'optim': optim.state_dict()},
    os.path.join(OUTPUT_DIR, 'model.pth'),
)
# test
with torch.no_grad():
    model.eval()

    for batch_idx, data in enumerate(test_loader, 1):
        img = data['img'].to(device)
        label = data['label'].to(device)
        output = model(img)

        err = loss(output,label)
        print('[Test] | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}'.format(batch_idx, test_batch, err.item()))

        img = img.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
        label = label.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # greater than 0.5 is classified to 1 otherwise 0
        def classify(x):
            return 1.0*(torch.sigmoid(x)>0.5)

        output = classify(output).to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        for j in range(label.shape[0]):
            current = batch_size*(batch_idx-1)+j
            plt.imsave(os.path.join(OUTPUT_DIR, f'label_{current:04}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(OUTPUT_DIR, f'output_{current:04}.png'), output[j].squeeze(), cmap='gray')
