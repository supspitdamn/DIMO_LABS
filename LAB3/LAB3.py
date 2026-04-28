import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from datetime import datetime

new_dir = f"LAB3//new_trial_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"
data_root = r"LAB3\img_align_celeba"
os.makedirs(new_dir, exist_ok=True)

# Гиперпараметры
if __name__ == '__main__':

    workers = 2
    batch_size = 128 
    image_size = 64
    nc = 3 # Каналы изображения (RGB)
    nz = 100 # Длина векторов скрытого пространства
    ngf = 64 # Параметр глубины карт признаков генератора
    ndf = 64 # Параметр глубины карт признаков дескриминатора
    num_epochs = 5 
    lr = 0.0002
    beta1 = 0.5
    ngpu = 1

    dataset = dset.ImageFolder(root=data_root,
                            transform=transforms.Compose([
                                transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset,
                                            shuffle = True,
                                                batch_size=batch_size,
                                                num_workers=workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")
    print(f"Обучение на: {device}")

    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Тренировочные изображения")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))

    def init_weights(m):
        classname = m.__class__.__name__
        # Для сверточных слоев
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):

        def __init__(self, ngpu):

            super().__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),

                nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
            )

        def forward(self, input):

            return self.main(input)
    
    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(

                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)
        
    netG = Generator(ngpu).to(device)
    netD = Discriminator(ngpu).to(device)

    if device.type == "cuda" and ngpu > 1:
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    netG.apply(init_weights)
    netD.apply(init_weights)

    print(netG)
    print(netD)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    def training(netD, netG, optimizerD, optimizerG, dataloader, device, epochs, nz, criterion):
        img_list = []
        G_losses = []
        D_losses = []
        real_label = 1.
        fake_label = 0.
        
        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        for epoch in tqdm(range(epochs), desc="Процесс обучения"):
            for idx, (images, _) in enumerate(dataloader):
                
                # --- 1. Обновление Дискриминатора ---
                netD.zero_grad()
                real_cpu = images.to(device)
                b_size = real_cpu.size(0)
                # Исправлено dtyp -> dtype
                label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                
                output = netD(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()

                # Генерация фейков
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)
                label.fill_(fake_label)
                
                output = netD(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()
                
                errD = errD_real + errD_fake
                optimizerD.step()

                # --- 2. Обновление Генератора ---
                netG.zero_grad()
                label.fill_(real_label) # Генератор хочет, чтобы дискриминатор выдал "1"
                
                output = netD(fake).view(-1)
                errG = criterion(output, label)
                errG.backward()
                optimizerG.step()

                # Сохраняем значения Loss
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            # Сохраняем картинки после каждой эпохи
            with torch.no_grad():
                fake_imgs = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_imgs, padding=2, normalize=True))

        return G_losses, D_losses, img_list

    G_losses, D_losses, img_list = training(netD, netG, optimizerD, optimizerG, dataloader, device, 10, nz, criterion)

    plt.figure(figsize=(10,5))
    plt.title("График функции потерь генератора и дискриминатора")
    plt.plot(G_losses, label = "Генератор")
    plt.plot(D_losses, label = "Дискриминатор")
    plt.xlabel("Итерации")
    plt.ylabel("Лосс")
    plt.legend()
    plt.savefig(f"{new_dir}//GD_loss_func_graph.png")
    plt.show()
    plt.pause(3)
    plt.close()

    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save(f"{new_dir}//training_evolution.gif", writer='pillow', fps=2)

    real_batch = next(iter(dataloader))

    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Реальные картинки")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Подделанные картинки")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig(f"{new_dir}//final_comparison.png", dpi = 300)
    plt.show()
    plt.pause(3)
    plt.close()


