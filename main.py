import torch
import argparse
import os
import random
import torchvision.utils
from Generator import Generator
from Discriminator import Discriminator
from torchvision import transforms
import csv
from tqdm import tqdm  # 新增
import time  # 新增



def main():
    # 固定参数
    batch_size = 128
    image_size = 64
    nc = 3
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 10
    lr = 0.0002
    beta1 = 0.5

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='faces', help='faces')
    parser.add_argument('--dataroot', default='./faces/faces', help='path to the root of dataset')
    parser.add_argument('--workers', type=int, default=2, help='number of worker threads for loading the data with Dataloader')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size used in training')
    parser.add_argument('--img_size', type=int, default=image_size, help='the height / width of the input images used for training')
    parser.add_argument('--nz', type=int, default=nz, help='size of the latent vector z')
    parser.add_argument('--ngf', type=int, default=ngf, help='size of feature maps in G')
    parser.add_argument('--ndf', type=int, default=ndf, help='size of feature maps in D')
    parser.add_argument('--nepoch', type=int, default=num_epochs, help='number of epochs to run')
    parser.add_argument('--lr', type=float, default=lr, help='learning rate for training')
    parser.add_argument('--beta1', type=float, default=beta1, help='beta1 hyperparameter for Adam optimizers')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs available')
    parser.add_argument('--dev', type=int, default=0, help='which CUDA device to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--results', default='./results', help='folder to store images and model checkpoints')
    opt = parser.parse_args()
    print(opt)

    """
    Create a folder to store images and model checkpoints
    """
    if not os.path.exists(opt.results):
        os.mkdir(opt.results)

    # 固定随机种子
    manualSeed = 42
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    """
    Use GPUs if available.
    """
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    torch.cuda.set_device(opt.dev)

    """
    Create the dataset.
    """
    # 只支持 faces 数据集
    # 数据增强操作，参考 MindSpore 代码
    transform = transforms.Compose([
        transforms.Resize(image_size),           # Resize
        transforms.CenterCrop(image_size),       # CenterCrop
        transforms.ToTensor(),                   # HWC2CHW + [0,1]
        # 去掉 Normalize，仅归一化到 [0,1]
    ])

    # 创建数据集
    dataset = torchvision.datasets.ImageFolder(
        root=opt.dataroot,
        transform=transform
    )
    nc = 3

    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers)
    )

    """
    Custom weights initialization called on netG and netD.
    All model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02. 
    Note We set bias=False in both Conv2d and ConvTranspose2d.
    """
    def weights_init(input):
        classname = input.__class__.__name__
        if classname.find('Conv') != -1:
            input.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            input.weight.data.normal_(1.0, 0.02)
            input.bias.data.fill_(0)

    '''
    Create a generator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    '''
    netG = Generator(nz=opt.nz, ngf=opt.ngf, nc=nc, ngpu=opt.ngpu).to(device)
    netG.apply(weights_init)
    '''
    Load the trained netG to continue training if it exists.
    '''
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    '''
    Create a discriminator and apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    '''
    netD = Discriminator(nc=nc, ndf=opt.ndf, ngpu=opt.ngpu).to(device)
    netD.apply(weights_init)
    '''
    Load the trained netD to continue training if it exists.
    '''
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    """
    A Binary Cross Entropy loss is used and two Adam optimizers are responsible for updating 
     netG and netD, respectively. 
    """
    loss = torch.nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    #Create batch of latent vectors that we will use to visualize
     #the progression of the generator.
    fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)
    #Establish convention for real and fake labels during training.
    real_label = 1
    fake_label = 0

    """
    Training.
    """
    #This line may increase the training speed a bit.
    torch.backends.cudnn.benchmark = True

    # 新增：用于收集损失的列表
    G_losses = []
    D_losses = []

    # 新增：定义csv文件路径
    loss_csv_path = os.path.join(opt.results, "losses.csv")

    # 选取一张真实图片（如第一张），并生成一个固定噪声
    single_real_img, _ = dataset.dataset[0]  # 注意这里 dataset.dataset 是 ImageFolder
    single_real_img = single_real_img.unsqueeze(0).to(device)  # 加 batch 维度
    single_noise = torch.randn(1, opt.nz, 1, 1, device=device)

    print("Starting Training Loop...")
    for epoch in range(opt.nepoch):
        print(f"Epoch [{epoch+1}/{opt.nepoch}]")
        total_batches = len(dataset)
        start_time = time.time()  # 记录epoch开始时间
        for i, data in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}", ncols=80), 0):
            """
            (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            """
            '''
            Train with real batch.
            '''
            netD.zero_grad()
            #Format batch
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float)
            #Forward propagate real batch through D.
            output = netD(real_cpu)
            #Calculate loss on real batch.
            errD_real = loss(output, label)
            #Calculate gradients for D in the backward propagation.
            errD_real.backward()
            D_x = output.mean().item()

            '''
            Train with fake batch.
            '''
            #Sample batch of latent vectors.
            noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
            #Generate fake image batch with G.
            fake = netG(noise)
            label.fill_(fake_label)

            #Classify all fake batch with D.
            #.detach() is a safer way for the exclusion of subgraphs from gradient computation.
            output = netD(fake.detach())
            #Calculate D's loss on fake batch.
            errD_fake = loss(output, label)
            #Calculate the gradients for fake batch.
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            #Get D's total loss by adding the gradients from the real and fake batches.
            errD = errD_real + errD_fake
            #Update D.
            optimizerD.step()

            """
            (2) Update G network: maximize log(D(G(z)))
            """
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for Generator cost
            #Since we just updated D, perform another forward propagation of fake batch through D
            output = netD(fake)
            #Calculate G's loss based on this output.
            errG = loss(output, label)
            #Calculate the gradients for G.
            errG.backward()
            D_G_z2 = output.mean().item()
            #Update G.
            optimizerG.step()

            # tqdm会自动显示进度，无需手动打印batch进度
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.nepoch, i, total_batches,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # 新增：每50次迭代收集一次损失
            if i % 50 == 0:
                G_losses.append(errG.item())
                D_losses.append(errD.item())

            if i % 100 == 0:
                '''
                #torchvision.utils.save_image(tensor, filename, nrow=8, padding=2, normalize=False,\
                 range=None, scale_each=False, pad_value=0)
                 #Save a given Tensor into an image file.
                '''
                torchvision.utils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.results,
                        normalize=True)
                fake = netG(fixed_noise)
                torchvision.utils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.results, epoch),
                        normalize=True)

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}] 用时: {epoch_time:.2f} 秒")
        print()  # 每个epoch结束换行
        # 每个epoch结束后保存损失到csv
        with open(loss_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # 如果是第一个epoch且第一个batch，写表头
            if epoch == 0 and len(G_losses) == 1:
                writer.writerow(["epoch", "iter", "G_loss", "D_loss"])
            for idx, (g, d) in enumerate(zip(G_losses, D_losses)):
                writer.writerow([epoch, idx * 50, g, d])
            # 清空本epoch的损失列表
            G_losses.clear()
            D_losses.clear()

        """
        Save the trained model.
        """ 
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.results, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.results, epoch))

        # 新增：每个epoch结束后生成一张特定噪声的图片
        with torch.no_grad():
            single_fake_img = netG(single_noise)
            fake_images = netG(fixed_noise)
        torchvision.utils.save_image(
            single_fake_img.detach(),
            f"{opt.results}/single_sample_epoch_{epoch:03d}.png",
            normalize=True
        )
        torchvision.utils.save_image(
            fake_images.detach(),
            f"{opt.results}/fake_samples_epoch_end_{epoch:03d}.png",
            normalize=True
        )
        # 可选：也保存真实图片用于对比（只保存一次）
        if epoch == 0:
            torchvision.utils.save_image(
                single_real_img,
                f"{opt.results}/single_real_sample.png",
                normalize=True
            )

if __name__ == "__main__":
    main()