# py paper_comparison.py ./crop ./paper/comparison.png 0 4 9 19 39 59
import os
from PIL import Image
import torchvision.utils
import torch

def comparison_grid(epoch_list, src_dir, out_path):
    imgs = []
    for epoch in epoch_list:
        fname = f"crop_fake_samples_epoch_end_{epoch:03d}.png"
        img_path = os.path.join(src_dir, fname)
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            imgs.append(torch.from_numpy(np.array(img)).permute(2,0,1).float()/255)
        else:
            print(f"警告: 未找到 {img_path}，已跳过")
    if not imgs:
        print("错误: 没有找到任何有效图片，未生成对比图")
        return
    # 横向拼接
    grid = torchvision.utils.make_grid(imgs, nrow=len(imgs), padding=2, normalize=True)
    if not os.path.splitext(out_path)[1]:
        out_path += ".png"
    torchvision.utils.save_image(grid, out_path)
    print(f"已保存对比图到 {out_path}")

if __name__ == "__main__":
    import numpy as np
    import sys
    # 用法: python paper_comparison.py <src_dir> <输出路径> <epoch1> <epoch2> ...
    if len(sys.argv) < 4:
        print("用法: python paper_comparison.py <src_dir> <输出路径> <epoch1> <epoch2> ...")
        # 示例：自动对比 0, 5, 9
        # src_dir = "./results"
        # out_path = "./paper/comparison.png"
        # epochs = [0, 5, 9]
        # comparison_grid(epochs, src_dir, out_path)
    else:
        src_dir = sys.argv[1]
        out_path = sys.argv[2]
        epochs = [int(e) for e in sys.argv[3:]]
        comparison_grid(epochs, src_dir, out_path)
        comparison_grid(epochs, src_dir, out_path)
