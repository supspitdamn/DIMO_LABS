import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import zipfile
import os

import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, Dataset

import shutil
from tqdm import tqdm


device = "cuda"

def show_input(input_tensor, title = ''):

    image = input_tensor.permute(1, 2, 0).numpy()
    plt.imshow(image.clip((0, 1)))
    plt.title(title)
    plt.show()
    plt.pause(0.001)

import torch
from tqdm import tqdm

def train(model, loss, optimizer, scheduler, num_epochs, device, train_dataloader, val_dataloader):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}:", flush=True)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_acc = 0.0

            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(inputs)
                    loss_value = loss(preds, labels)
                    preds_class = preds.argmax(dim=1)

                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()

                running_loss += loss_value.item()
                running_acc += (preds_class == labels.data).float().mean().item()

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        scheduler.step()
    
    return model

zip_path = "./LAB2/plates.zip"
extract_path = "./LAB2/working"
data_root = './LAB2/data'    

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_path)

class_names = ['cleaned', 'dirty']
for stage in ['train', 'val']:
    for cl in class_names:
        os.makedirs(os.path.join(data_root, stage, cl), exist_ok=True)

for class_name in class_names:

    source_dir = os.path.join(extract_path, 'plates', 'train', class_name)
    
    files = os.listdir(source_dir)
    for i, file_name in enumerate(tqdm(files, desc=f"Processing {class_name}")):

        if i % 6 == 0:
            dest_stage = 'val'
        else:
            dest_stage = 'train'
        
        shutil.copy(
            os.path.join(source_dir, file_name), 
            os.path.join(data_root, dest_stage, class_name, file_name)
        )

print("Данные успешно распределены по папкам train и val")

train_dir = os.path.join(data_root, 'train') # './LAB2/data/train'
val_dir = os.path.join(data_root, 'val')     # './LAB2/data/val'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(train_dir, transform)
val_dataset = datasets.ImageFolder(val_dir, transform)

b_size = 2

train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=b_size, shuffle=True)

X_batch, y_batch = next(iter(train_dataloader))
plt.imshow(X_batch[0].permute(1,2,0).numpy())

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Количество обучаемых параметров: {num_params:,}')

model.to(device)
train(model, loss, optimizer, scheduler, 50, device, train_dataloader, val_dataloader)


model.eval()

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

test_path = os.path.join(extract_path, 'plates', 'test') 
test_dataset = ImageFolderWithPaths(test_path, transform)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=b_size, shuffle=False, num_workers=0)

test_predictions = []
test_img_paths = []

model.to(device)
for inputs, labels, paths in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)
    with torch.set_grad_enabled(False):
        preds = model(inputs)
    test_predictions.append(
        torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
    test_img_paths.extend(paths)

test_predictions = np.concatenate(test_predictions)

submission_df = pd.DataFrame.from_dict({'id': test_img_paths, 'label': test_predictions})

submission_df['id'] = submission_df['id'].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
submission_df['label'] = submission_df['label'].map(lambda pred: 'dirty' if pred > 0.5 else 'cleaned')

submission_df.set_index('id', inplace=True)
print(submission_df.head(n=6))