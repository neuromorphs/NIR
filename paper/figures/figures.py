import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Define Software (SW) and Hardware (HW) labels
sw_labels = [f"SW {i}" for i in range(1, 7)]
hw_labels = [f"HW {i}" for i in range(1, 6)]

# Draw the Software (SW) labels on the left
for i, label in enumerate(sw_labels):
    ax.text(0.1, 0.9 - i * 0.15, label, fontsize=12, ha="center", va="center")

# Draw the Hardware (HW) labels on the right
for i, label in enumerate(hw_labels):
    ax.text(0.7, 0.9 - i * 0.15 - 0.075, label, fontsize=12, ha="center", va="center")

# Draw connections between SW and HW without NIR
for i in range(len(sw_labels)):
    for j in range(len(hw_labels)):
        spacing = 0.05
        ax.plot(
            [0.1 + spacing, 0.7 - spacing],
            [0.9 - i * 0.15, 0.9 - j * 0.15 - 0.075],
            "k-",
        )

# # Draw 'Without NIR' label at the top
# ax.text(0.4, 1, 'Without NIR', fontsize=14, ha='center', va='center')

# # Draw '30 HW->SW' label at the bottom
# rect = patches.Rectangle((0.15, 0.05-0.1), 0.5, 0.15, linewidth=1, edgecolor='k', facecolor='lightgray')
# ax.add_patch(rect)
# ax.text(0.4, 0.125-0.1, '30 HW->SW', fontsize=12, ha='center', va='center')

# Set aspect of the plot and hide axes
ax.set_aspect("equal", "box")
ax.axis("off")
plt.tight_layout()
plt.savefig("compiler_withoutnir.pdf", dpi=300)
plt.show()

#########################################################################################

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Define Software (SW) and Hardware (HW) labels
sw_labels = [f"SW{i}" for i in range(1, 7)]
hw_labels = [f"HW{i}" for i in range(1, 6)]

# Draw the Software (SW) labels on the left
for i, label in enumerate(sw_labels):
    ax.text(0.1, 0.9 - i * 0.15, label, fontsize=12, ha="center", va="center")

# Draw the Hardware (HW) labels on the right
for i, label in enumerate(hw_labels):
    ax.text(0.7, 0.9 - i * 0.15 - 0.075, label, fontsize=12, ha="center", va="center")

# Draw the centered NIR box
rect_centery = (
    0.9 - (len(sw_labels) - 1) * 0.15 / 2
)  # Vertically center with respect to the labels
hrect = 0.15
xoff = -0.1
rect = patches.Rectangle(
    (0.4 + xoff, rect_centery - hrect / 2),
    0.2,
    hrect,
    linewidth=1,
    edgecolor="w",
    facecolor="lightgray",
)
ax.add_patch(rect)
ax.text(0.5 + xoff, rect_centery, "NIR", fontsize=12, ha="center", va="center")

# Draw connections between SW and NIR
spacing = 0.05
for i in range(len(sw_labels)):
    ax.plot([0.1 + spacing, 0.4 + xoff], [0.9 - i * 0.15, rect_centery], "k-")

# Draw connections between HW and NIR
for i in range(len(hw_labels)):
    ax.plot([0.75 + xoff, 0.6 + xoff], [0.9 - i * 0.15 - 0.075, rect_centery], "k-")

# Set aspect of the plot and hide axes
ax.set_aspect("equal", "box")
ax.axis("off")
plt.tight_layout()
plt.savefig("compiler_withnir.pdf", dpi=300)
plt.show()
