#!/usr/bin/env python3
"""Redraw the official per-task deltas bar chart (neater standalone version).
Writes eval/plots/official_deltas.{pdf,png}. Numbers from eval/official_results.md."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

INK="#2b2b2b"; BLUE="#4477AA"; RED="#CC6677"; GREEN="#228833"; LIGHT="#DDDDDD"

rows = [("COMPS",+1.30,9),("GLUE",+1.08,9),("EWoK",+0.35,5),
        ("BLiMP",+0.30,6),("Entity Tracking",-0.21,5),("Supplement",-0.69,3)]
rows.sort(key=lambda r: r[1])
names=[r[0] for r in rows]; means=[r[1] for r in rows]; poss=[r[2] for r in rows]

plt.rcParams.update({
    "font.size":10.5,"axes.labelsize":10.5,"axes.edgecolor":INK,"axes.labelcolor":INK,
    "text.color":INK,"xtick.color":INK,"ytick.color":INK,
    "axes.spines.top":False,"axes.spines.right":False,"axes.spines.left":False,
    "axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":200,
    "savefig.bbox":"tight","font.family":"DejaVu Sans"})

fig,ax=plt.subplots(figsize=(6.6,3.2)); y=list(range(len(names)))
colors=[GREEN if p==9 else (BLUE if m>0 else RED) for m,p in zip(means,poss)]
ax.barh(y,means,color=colors,height=0.62,zorder=3)
ax.axvline(0,color=INK,lw=1.0,zorder=2)
ax.grid(axis="x",color=LIGHT,linewidth=0.7,zorder=0); ax.grid(axis="y",visible=False)
for yi,m,p in zip(y,means,poss):
    off=0.04 if m>=0 else -0.04; ha="left" if m>=0 else "right"
    ax.text(m+off,yi,f"{m:+.2f}  ({p}/9)",va="center",ha=ha,fontsize=9.2,
            color=INK,fontweight=("bold" if p==9 else "normal"),zorder=4)
ax.set_yticks(y); ax.set_yticklabels(names); ax.tick_params(axis="y",length=0)
ax.tick_params(axis="x",length=3,color=INK)
ax.set_xlabel("mean $\\Delta$  (vision-init $-$ baseline, points)",fontsize=10)
ax.set_xlim(-1.4,2.15)
ax.legend(handles=[Patch(color=GREEN,label="positive in all 9 configs"),
                   Patch(color=BLUE,label="positive (not all 9)"),
                   Patch(color=RED,label="negative")],
          loc="lower right",fontsize=8.6,handlelength=1.1,handleheight=1.1,
          borderpad=0.5,labelspacing=0.35)
fig.tight_layout()
out=Path("eval/plots"); out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/"official_deltas.pdf"); fig.savefig(out/"official_deltas.png",dpi=200)
print("wrote eval/plots/official_deltas.{pdf,png}")
