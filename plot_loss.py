import matplotlib.pyplot as plt
import csv

csv_path = "./results/losses.csv"

iters = []
G_losses = []
D_losses = []

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    # 自动检测 G_loss 和 D_loss 的列索引
    try:
        g_idx = header.index("G_loss")
        d_idx = header.index("D_loss")
    except ValueError:
        # 如果没有表头，假定G_loss在第2列，D_loss在第3列
        g_idx = 2
        d_idx = 3
        # 如果没有表头，第一行就是数据
        reader = csv.reader(open(csv_path, "r"))
    for idx, row in enumerate(reader):
        if len(row) < max(g_idx, d_idx) + 1:
            continue
        iters.append(idx)
        G_losses.append(float(row[g_idx]))
        D_losses.append(float(row[d_idx]))

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(iters, G_losses, label="G")
plt.plot(iters, D_losses, label="D")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("./results/loss_curve.png")
plt.show()
