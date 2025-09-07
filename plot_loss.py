import matplotlib.pyplot as plt
import csv

csv_path = "./results/losses.csv"

iters = []
G_losses = []
D_losses = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row["iter"]) + int(row["epoch"]) * 1000)  # 估算迭代数
        G_losses.append(float(row["G_loss"]))
        D_losses.append(float(row["D_loss"]))

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
