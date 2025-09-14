# 用法: python paper_gif.py <图片文件夹> <输出路径> [间隔毫秒]
import os
from PIL import Image

def make_gif_from_folder(img_folder, out_path, duration=300):
    imgs = []
    for fname in sorted(os.listdir(img_folder)):
        if fname.endswith(".png"):
            img = Image.open(os.path.join(img_folder, fname)).convert("RGB")
            imgs.append(img)
    if not imgs:
        print("没有找到任何图片，未生成GIF")
        return
    imgs[0].save(
        out_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0
    )
    print(f"已保存GIF到 {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python paper_gif.py <图片文件夹> <输出路径> [间隔毫秒]")
    else:
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else 300
        make_gif_from_folder(sys.argv[1], sys.argv[2], duration)
