import torch
import argparse
import os
from Generator import Generator
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./netG_epoch_59.pth', help='生成器模型路径')
    parser.add_argument('--output', type=str, default='./sample_from_59.png', help='输出图片路径')
    parser.add_argument('--nz', type=int, default=100, help='噪声向量长度')
    parser.add_argument('--ngf', type=int, default=64, help='生成器特征图数')
    parser.add_argument('--nc', type=int, default=3, help='输出通道数')
    parser.add_argument('--ngpu', type=int, default=1, help='GPU数量')
    parser.add_argument('--num', type=int, default=16, help='生成图片数量')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 构建生成器并加载权重
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc, ngpu=args.ngpu).to(device)
    netG.load_state_dict(torch.load(args.model_path, map_location=device))
    netG.eval()

    # 固定随机种子保证复现
    torch.manual_seed(42)

    # 生成噪声
    noise = torch.randn(args.num, args.nz, 1, 1, device=device)
    with torch.no_grad():
        fake_imgs = netG(noise)
    # 保存图片
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    vutils.save_image(fake_imgs, args.output, nrow=4, normalize=True)
    print(f"已保存生成图片到 {args.output}")

if __name__ == '__main__':
    main()
