import os
from PIL import Image

def crop_from_grid(img_path, row, col):
    img = Image.open(img_path)
    x = 2 + col * (64 + 2)
    y = 2 + row * (64 + 2)
    box = (x, y, x + 64, y + 64)
    return img.crop(box)

def batch_crop(src_dir, row, col, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for fname in os.listdir(src_dir):
        if fname.startswith("fake_samples_epoch_end_") and fname.endswith(".png"):
            img_path = os.path.join(src_dir, fname)
            cropped = crop_from_grid(img_path, row, col)
            out_path = os.path.join(out_dir, f"crop_{fname}")
            cropped.save(out_path)
            print(f"已保存 {out_path}")

if __name__ == "__main__":
    # 用法: python batch_crop_from_grid.py <src_dir> <row> <col> <out_dir>
    import sys
    if len(sys.argv) != 5:
        print("用法: python batch_crop_from_grid.py <src_dir> <row> <col> <out_dir>")
    else:
        batch_crop(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
