import matplotlib.pyplot as plt
import matplotlib.lines as mlines

nfe_data = {
    "RDM (s = k - m)": [500, 986, 2084],
    "DF (s = k - m)": [286, 858, 2002],
    "MaskFlow (s = k - m)": [20, 140, 240],
    "MaskFlow (s = 1)": [740, 2900]
}

fvd_data = {
    "RDM (s = k - m)": [52.43, 201.7, 338.34],
    "DF (s = k - m)": [60.3, 175.01, 232.89],
    "MaskFlow (s = k - m)": [53.17, 188.22, 334.15],
    "MaskFlow (s = 1)": [50.87, 181.11]
}


extrapolation_factors = {
    "RDM (s = k - m)": ["1x", "2x", "5x"],
    "DF (s = k - m)": ["1x", "2x", "5x"],
    "MaskFlow (s = k - m)": ["1x", "2x", "5x"],
    "MaskFlow (s = 1)": ["2x", "5x"]  
}

marker_size_map = {"1x": 100, "2x": 200, "5x": 500}

color_scheme = {
    "RDM (s = k - m)": "#A0A0A0",  
    "DF (s = k - m)": "#C0C0C0",  
    "MaskFlow (s = k - m)": "#2F4F4F", 
    "MaskFlow (s = 1)": "#8FBFBF", 
}

all_nfe = [x for vals in nfe_data.values() for x in vals]
legend_loc = "upper left"

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

pdf_path = "nfe_vs_fvd_vs_ep_dmlab.pdf"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.show()
