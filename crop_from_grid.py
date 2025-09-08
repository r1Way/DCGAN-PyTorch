from PIL import Image
import os

def crop_from_grid(big_img_path, row, col, out_path):
    # 参数说明：row/col均从0开始
    img = Image.open(big_img_path)
    # 计算左上角坐标
    x = 2 + col * (64 + 2)
    y = 2 + row * (64 + 2)
    box = (x, y, x + 64, y + 64)
    small_img = img.crop(box)
    # 自动补全扩展名
    if not os.path.splitext(out_path)[1]:
        out_path += ".png"
    small_img.save(out_path)
    print(f"已保存第{row}行第{col}列的小图到 {out_path}")

if __name__ == "__main__":
    # 示例用法：python crop_from_grid.py fake_images.png 1 2 out.png
    import sys
    if len(sys.argv) != 5:
        print("用法: python crop_from_grid.py <大图路径> <行号> <列号> <输出路径>")# 行号从上往下，列号从左往右，均从0开始。
    else:
        crop_from_grid(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
