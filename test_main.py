import torch
import os
import random
import torchvision.utils
from Generator import Generator
from Discriminator import Discriminator
from torchvision import transforms, datasets

# 测试参数
batch_size = 16
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 1
lr = 0.0002
beta1 = 0.5
ngpu = 1

results_dir = './results_test'
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

manualSeed = 42
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
])

# 只取前100张图片
dataset_full = datasets.ImageFolder(
    root='./faces/faces',
    transform=transform
)
indices = list(range(min(100, len(dataset_full))))
dataset = torch.utils.data.Subset(dataset_full, indices)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

def weights_init(input):
    classname = input.__class__.__name__
    if classname.find('Conv') != -1:
        input.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        input.weight.data.normal_(1.0, 0.02)
        input.bias.data.fill_(0)

netG = Generator(nz=nz, ngf=ngf, nc=nc, ngpu=ngpu).to(device)
netG.apply(weights_init)
netD = Discriminator(nc=nc, ndf=ndf, ngpu=ngpu).to(device)
netD.apply(weights_init)

loss = torch.nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

torch.backends.cudnn.benchmark = True

print("Starting Test Training Loop...")
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size_cur = real_cpu.size(0)
        label = torch.full((batch_size_cur,), real_label, device=device, dtype=torch.float)
        output = netD(real_cpu)
        errD_real = loss(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size_cur, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = loss(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = loss(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, num_epochs, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i == 0:
            torchvision.utils.save_image(real_cpu,
                    '%s/real_samples.png' % results_dir,
                    normalize=True)
            fake = netG(fixed_noise)
            torchvision.utils.save_image(fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (results_dir, epoch),
                    normalize=True)

    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (results_dir, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (results_dir, epoch))
    with torch.no_grad():
        fake_images = netG(fixed_noise)
    torchvision.utils.save_image(
        fake_images.detach(),
        '%s/fake_samples_epoch_end_%03d.png' % (results_dir, epoch),
        normalize=True
    )
print("Test run finished.")
