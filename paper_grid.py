#  python paper_grid.py <图片文件夹> <输出路径> [每行图片数]
import os
from PIL import Image
import torchvision.utils
import torch

def make_grid_from_folder(img_folder, out_path, nrow=10):
    imgs = []
    for fname in sorted(os.listdir(img_folder)):
        if fname.endswith(".png"):
            img = Image.open(os.path.join(img_folder, fname)).convert("RGB")
            imgs.append(torch.from_numpy(np.array(img)).permute(2,0,1).float()/255)
            if len(imgs) == 60:
                break
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    torchvision.utils.save_image(grid, out_path)
    print(f"已保存网格图到 {out_path}")

if __name__ == "__main__":
    import numpy as np
    import sys
    # 用法: python paper_grid.py <图片文件夹> <输出路径> [nrow]
    if len(sys.argv) < 3:
        print("用法: python paper_grid.py <图片文件夹> <输出路径> [每行图片数]")
    else:
        nrow = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        make_grid_from_folder(sys.argv[1], sys.argv[2], nrow)
