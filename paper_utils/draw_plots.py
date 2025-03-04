import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Data points
NFE = [10, 20, 30, 40, 50, 80, 100]
FFS = [52.69, 43.38, 49.55, 53.32, 53.03, 52.86, 57.81]
DMLab = [76.44, 45.84, 48.15, 48.77, 46.01, 45.84, 50.05]

# NFE = [20, 50, 100, 250, 400, 500]
# FFS = [121.72, 71.56, 57.81, 48.98, 45.99, 43.08]
# DMLab = [54.87, 49.76, 50.85, 49.62, 51.92, 50.975]

# Define colors to match the original plot
ffs_color = "#2f4f4f"  # Dark greenish color
dmlab_color = "#f4a700"  # Golden yellow

# Create the figure
plt.figure(figsize=(9, 7), dpi=300)

# Plot lines with square markers and dashed style
plt.plot(NFE, FFS, linestyle="--", marker="s", color=ffs_color, label="FaceForensics", linewidth=2)
plt.plot(NFE, DMLab, linestyle="--", marker="s", color=dmlab_color, label="DMLab", linewidth=2)

# Labels and formatting
plt.xlabel("NFE", fontsize=14, fontweight="bold")
plt.ylabel("FVD", fontsize=14, fontweight="bold")

# Ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add the legend
legend_handles = [
    mlines.Line2D([], [], color=color, linestyle="--", marker='s', markersize=10, label=method)
    for method, color in [("DMLab", dmlab_color), ("FaceForensics", ffs_color)]
]
plt.legend(handles=legend_handles, fontsize=18, loc="upper right", frameon=False, handlelength=2)

# Save the figure as high-resolution PDF
pdf_path = "fvd_vs_nfe_mg.pdf"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")

print("plot saved!")

# Show the plot
plt.show()
