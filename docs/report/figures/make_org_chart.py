"""Draw the department organisation chart for the report (fig-org-chart.png).

    python docs/report/figures/make_org_chart.py

Boxes whose text is in [brackets] are placeholders the author still has to confirm; they are
drawn with a yellow fill so they stand out. Edit BOXES and rerun.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).with_name("fig-org-chart.png")

# name: (x centre, y centre, title, subtitle[, width])
BOXES = {
    "plant": (5.6, 9.0, "Plant leader", "GE Aerospace Pune facility (MMF)"),
    "fcoe": (3.0, 7.0, "Fabrication Center of Excellence (FCOE)", "[title of the FCOE head]"),
    "mcoe": (6.95, 7.0, "Machining COE", "Center of Excellence", 2.5),
    "acoe": (9.45, 7.0, "Assembly COE", "Center of Excellence", 2.2),
    "me": (3.0, 5.0, "Manufacturing Engineering (ME)", "Pankesh Pisal, General Manager"),
    "other": (7.4, 5.0, "[Other FCOE sections]", "[e.g. production, quality]"),
    "pa": (3.0, 3.0, "Process Automation team", "Aslam Mulla, Senior Manager (trainer)"),
    "medp": (7.4, 3.0, "ME Development Program", "Soham Bhat Dubere (second supervisor)"),
    "intern": (3.0, 1.0, "Intern (author)", "Vision Guard project"),
    "support": (7.4, 1.0, "Site functions", "EHS · Maintenance · IT · Lean management"),
}
REPORTS = [("plant", "fcoe"), ("plant", "mcoe"), ("plant", "acoe"), ("fcoe", "me"),
           ("fcoe", "other"), ("me", "pa"), ("pa", "intern")]
DOTTED = [("me", "medp"), ("medp", "intern"), ("support", "intern")]
W, H = 4.0, 1.15


def width(name):
    box = BOXES[name]
    return box[4] if len(box) > 4 else W


def edge(a, b, style):
    (xa, ya, *_), (xb, yb, *_) = BOXES[a], BOXES[b]
    if abs(ya - yb) < 0.1:  # same level: side to side
        wa, wb = width(a) / 2, width(b) / 2
        x0, x1 = (xa + wa, xb - wb) if xa < xb else (xa - wa, xb + wb)
        plt.plot([x0, x1], [ya, yb], style, color="#455a64", lw=1.4)
        return
    mid = (ya - H / 2 + yb + H / 2) / 2  # elbow: down, across, down
    plt.plot([xa, xa, xb, xb], [ya - H / 2, mid, mid, yb + H / 2], style, color="#455a64", lw=1.4)


fig = plt.figure(figsize=(11, 7.4), dpi=200)
ax = plt.gca()
for a, b in REPORTS:
    edge(a, b, "-")
for a, b in DOTTED:
    edge(a, b, ":")
for name, (x, y, title, sub, *_) in BOXES.items():
    w = width(name)
    placeholder = title.startswith("[") or sub.startswith("[")
    face = "#fff3b0" if placeholder else ("#e3eef7" if title != "Intern (author)" else "#d8efe0")
    box = FancyBboxPatch((x - w / 2, y - H / 2), w, H,
                         boxstyle="round,pad=0.02,rounding_size=0.12",
                         fc=face, ec="#1f4e79", lw=1.3)
    ax.add_patch(box)
    plt.text(x, y + 0.18, title, ha="center", va="center", fontsize=11.5, weight="bold",
             color="#10243a")
    plt.text(x, y - 0.24, sub, ha="center", va="center", fontsize=9.5, color="#33475b")
plt.plot([], [], "-", color="#455a64", label="reports to")
plt.plot([], [], ":", color="#455a64", label="supervision or support")
plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, frameon=False, fontsize=10)
plt.xlim(0.6, 10.7)
plt.ylim(0.1, 9.8)
plt.axis("off")
plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
