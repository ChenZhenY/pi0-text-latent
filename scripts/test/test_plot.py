import matplotlib.pyplot as plt

x = list(range(12))
y = [5, 5, 5, 5, 5, 3, 4, 2, 1, 3, 0, 0]

plt.figure(figsize=(8, 4))
plt.plot(x, y, color='teal', marker='o', linewidth=2)
for xi, yi in zip(x, y):
    plt.text(xi, yi, str(yi), ha='center', va='bottom', fontsize=10)
plt.title('Sample Line Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.tight_layout()
plt.savefig('plot.png')
plt.close()
