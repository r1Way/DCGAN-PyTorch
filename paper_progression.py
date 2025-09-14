import os
from PIL import Image
import torchvision.utils
import torch

def progression_grid(src_dir, out_path, nrow=10):
    imgs = []
    for fname in sorted(os.listdir(src_dir)):
        if fname.startswith("single_sample_epoch_") and fname.endswith(".png"):
            img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
            imgs.append(torch.from_numpy(np.array(img)).permute(2,0,1).float()/255)
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=2, normalize=True)
    torchvision.utils.save_image(grid, out_path)
    print(f"已保存演化图到 {out_path}")

if __name__ == "__main__":
    import numpy as np
    import sys
    # 用法: python paper_progression.py <src_dir> <输出路径> [nrow]
    if len(sys.argv) < 3:
        print("用法: python paper_progression.py <src_dir> <输出路径> [每行图片数]")
    else:
        nrow = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        progression_grid(sys.argv[1], sys.argv[2], nrow)
