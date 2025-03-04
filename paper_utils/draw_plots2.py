import matplotlib.pyplot as plt
import matplotlib.lines as mlines


nfe_data = {
    "Rolling Diffusion": [788, 1652, 3092],
    "Diffusion Forcing": [798, 1596, 3192],
    "MaskFlow (full sequence)": [60, 120, 240],  
    "MaskFlow (autoregressive)": [340, 1300, 2900]
}

fvd_data = {
    "Rolling Diffusion": [72.49, 248.13, 451.38],
    "Diffusion Forcing": [144.43, 272.14, 306.31],
    "MaskFlow (full sequence)": [59.93, 108.74, 214.39],  
    "MaskFlow (autoregressive)": [30.43, 104.69, 165.0]
}

extrapolation_factors = {"Rolling Diffusion": ["2x", "5x", "10x"],
                          "Diffusion Forcing": ["2x", "5x", "10x"],
                          "MaskFlow (full sequence)": ["2x", "5x", "10x"],
                          "MaskFlow (autoregressive)": ["2x", "5x", "10x"]} 

marker_size_map = {"2x": 200, "5x": 500, "10x": 1000}  

color_scheme = {
    "Rolling Diffusion": "#A0A0A0",  # Light grey
    "Diffusion Forcing": "#C0C0C0",  # Lighter grey
    "MaskFlow (full sequence)": "#2F4F4F",  # Lighter green 
    "MaskFlow (autoregressive)": "#8FBFBF",  # Darker green
}

plt.figure(figsize=(9, 7), dpi=300)


for method, color in color_scheme.items():
    if method in nfe_data:
        linewidth = 3 if "MaskFlow" in method else 2  
        
        plt.plot(
            nfe_data[method],
            fvd_data[method],
            linestyle="--", 
            color=color, 
            linewidth=linewidth, 
            alpha=0.8,
            label=method  
        )
        
        for i, (x, y) in enumerate(zip(nfe_data[method], fvd_data[method])):
            factor = extrapolation_factors[method][i] if i < len(extrapolation_factors[method]) else "5x"  
            marker_size = marker_size_map.get(factor, 50)  
            plt.scatter(
                [x], [y],
                s=marker_size,  
                color=color,
                marker='s'  
            )
            plt.text(x, y, factor, fontsize=12, ha='center', va='center', color='white', fontweight='bold')

plt.xlabel("NFE", fontsize=18, fontweight="bold")
plt.ylabel("FVD", fontsize=18, fontweight="bold")
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

legend_handles = [mlines.Line2D([], [], color=color, linestyle="--", marker='s', markersize=10, label=method) for method, color in color_scheme.items()]
plt.legend(handles=legend_handles, fontsize=18, loc="upper left", frameon=False, handlelength=2)

pdf_path = "nfe_vs_fvd_vs_ep_ffs.pdf"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()
