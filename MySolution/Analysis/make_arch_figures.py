import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
FIG_DIR = os.path.join(REPO_ROOT, "Paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "font.family": "sans-serif",
})

C_BLUE        = "#4c72b0"
C_ORANGE      = "#dd8452"
C_GREEN       = "#55a467"
C_RED         = "#c44e52"
C_PURPLE      = "#8172b3"
C_GREY        = "#cccccc"
C_DARK        = "#333333"
C_LIGHT_BLUE  = "#d8e3f0"
C_LIGHT_GREEN = "#d9ebe1"
C_LIGHT_RED   = "#f1d6d6"


def add_box(ax, x, y, w, h, label, *, fc="white", ec=C_DARK, lw=1.2,
            fontsize=10, fontweight="normal", rounding=0.02):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, zorder=3)


def add_arrow(ax, x1, y1, x2, y2, *, color=C_DARK, lw=1.5, style="-|>",
              connectionstyle="arc3,rad=0"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
        connectionstyle=connectionstyle, zorder=1,
    )
    ax.add_patch(arr)


def setup(ax, xlim, ylim, title=None):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, loc="left")


def make_fig01_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5.5))
    setup(ax, (0, 14), (0, 6), "Fig. 1  Pipeline of the Echoes of Equity study")

    add_box(ax, 0.2, 2.4, 1.8, 1.4, "TIS Corpus\n1,152 WAV\n96 speakers",
            fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5, fontweight="bold")
    ax.text(1.1, 2.2, "White 40  |  Black 28  |  South Asian 28",
            ha="center", fontsize=8, color=C_DARK)

    add_box(ax, 2.6, 2.6, 1.8, 1.0, "VoiceLab\n(Praat backend)",
            fc="white", ec=C_DARK)
    ax.text(3.5, 2.3, "60 acoustic features", ha="center", fontsize=8, color=C_DARK)

    add_box(ax, 5.0, 2.4, 1.9, 1.4, "Joint-stratified\n80/20 split\n(seed=42)",
            fc="white", ec=C_DARK)

    models = [
        ("RF",          C_BLUE,   False),
        ("LR",          C_ORANGE, False),
        ("ANN",         C_GREEN,  False),
        ("CNN",         C_RED,    False),
        ("DANN-Trust",  C_PURPLE, True),
    ]
    model_x = 7.6
    y0 = 4.6
    dy = 0.7
    for i, (name, color, headline) in enumerate(models):
        y = y0 - i * dy
        lw = 2.4 if headline else 1.2
        fc = color if headline else "white"
        fontcolor = "white" if headline else "black"
        box = FancyBboxPatch(
            (model_x, y), 1.5, 0.55,
            boxstyle="round,pad=0.005,rounding_size=0.04",
            linewidth=lw, edgecolor=color, facecolor=fc, zorder=2,
        )
        ax.add_patch(box)
        ax.text(model_x + 0.75, y + 0.275, name,
                ha="center", va="center", fontsize=10,
                fontweight="bold", color=fontcolor)
        add_arrow(ax, 6.9, 3.1, model_x, y + 0.275, color=C_GREY, lw=1)
    ax.text(model_x + 0.75, y0 + 0.7, "headline contribution",
            ha="center", fontsize=8, fontweight="bold", color=C_PURPLE, style="italic")

    add_box(ax, 9.8, 2.4, 1.8, 1.4,
            "Per-demographic\neval\n(5 seeds)",
            fc=C_LIGHT_GREEN, ec=C_GREEN, lw=1.5)
    for i in range(len(models)):
        y = y0 - i * dy + 0.275
        add_arrow(ax, model_x + 1.5, y, 9.8, 3.1, color=C_GREY, lw=0.8)

    add_box(ax, 12.0, 2.4, 1.8, 1.4,
            "results.json\nFigs 9, 13\nTables 3-10",
            fc="white", ec=C_DARK)

    add_arrow(ax, 2.0, 3.1, 2.6, 3.1, lw=2)
    add_arrow(ax, 4.4, 3.1, 5.0, 3.1, lw=2)
    add_arrow(ax, 11.6, 3.1, 12.0, 3.1, lw=2)

    ax.text(5.95, 2.2, "921 train  |  231 test",
            ha="center", fontsize=8, color=C_DARK)

    out = os.path.join(FIG_DIR, "fig1_pipeline.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def draw_stack(ax, blocks, *, y_centre=2.5, gap=0.4, height=1.6,
               arrow_color=C_DARK):
    x = 0.4
    centres = []
    for b in blocks:
        w = b["w"]
        add_box(ax, x, y_centre - height / 2, w, height, b["label"],
                fc=b.get("fc", "white"), ec=b.get("ec", C_DARK),
                lw=b.get("lw", 1.2), fontweight=b.get("fontweight", "normal"),
                fontsize=b.get("fontsize", 10))
        if b.get("sublabel"):
            ax.text(x + w / 2, y_centre - height / 2 - 0.25, b["sublabel"],
                    ha="center", va="top", fontsize=8, color=C_DARK,
                    style=b.get("substyle", "normal"))
        if b.get("toplabel"):
            ax.text(x + w / 2, y_centre + height / 2 + 0.2, b["toplabel"],
                    ha="center", va="bottom", fontsize=8, color=b.get("topcolor", C_DARK),
                    fontweight=b.get("topweight", "normal"), style="italic")
        centres.append((x, x + w))
        x += w + gap

    for i in range(len(blocks) - 1):
        x1 = centres[i][1]
        x2 = centres[i + 1][0]
        add_arrow(ax, x1, y_centre, x2, y_centre,
                  color=arrow_color, lw=1.5)
    return centres, x


def make_fig3_ann_arch():
    fig, ax = plt.subplots(figsize=(12, 4.5))

    blocks = [
        {"label": "x in R^60\n60 features", "w": 1.4,
         "fc": C_LIGHT_BLUE, "ec": C_BLUE, "fontweight": "bold"},
        {"label": "Dense 128\nReLU + BN\nDropout 0.3", "w": 1.7,
         "fc": "white", "sublabel": "7.8k params"},
        {"label": "Dense 64\nReLU + BN\nDropout 0.3", "w": 1.6,
         "fc": "white", "sublabel": "8.3k params"},
        {"label": "Dense 32\nReLU\n(bottleneck)", "w": 1.7,
         "fc": C_LIGHT_GREEN, "ec": C_GREEN, "lw": 1.8,
         "sublabel": "2.1k params",
         "toplabel": "re-used in DANN encoder (Fig. 6)",
         "topcolor": C_GREEN},
        {"label": "Dense 1\nSigmoid", "w": 1.3,
         "fc": "white", "sublabel": "33 params"},
        {"label": "y_hat in [0, 1]\nNeutral / Trust", "w": 1.5,
         "fc": C_LIGHT_BLUE, "ec": C_BLUE, "fontweight": "bold"},
    ]
    _, x_end = draw_stack(ax, blocks, y_centre=2.5, height=1.7)

    ax.set_xlim(0, x_end + 0.4)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fig. 3  Architecture of the ANN trust classifier",
                 fontsize=12, fontweight="bold", pad=10, loc="left")

    ax.text(x_end / 2, 0.4,
            "Total: 18.2k parameters  *  Adam lr=1e-3  *  batch=32  *  50 epochs  *  BCE loss",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    out = os.path.join(FIG_DIR, "fig3_ann_arch.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def make_fig4_cnn_arch():
    fig, ax = plt.subplots(figsize=(13, 4.5))

    blocks = [
        {"label": "X in R^{60x1}\nlength-60\nsignal", "w": 1.5,
         "fc": C_LIGHT_BLUE, "ec": C_BLUE, "fontweight": "bold"},
        {"label": "Conv1D 64, k=3\nReLU + BN\nDropout 0.2", "w": 2.0,
         "fc": "white", "sublabel": "-> (58, 64)",
         "toplabel": "kernel slides over ARBITRARY features  (see §IV-D caveat)",
         "topcolor": C_RED},
        {"label": "Conv1D 32, k=3\nReLU", "w": 1.7,
         "fc": "white", "sublabel": "-> (56, 32)"},
        {"label": "Flatten", "w": 1.2,
         "fc": "white", "sublabel": "1,792 units"},
        {"label": "Dense 64\nReLU", "w": 1.4,
         "fc": "white", "sublabel": "~115k params"},
        {"label": "Dense 1\nSigmoid", "w": 1.3,
         "fc": "white", "sublabel": "65 params"},
        {"label": "y_hat in [0,1]", "w": 1.3,
         "fc": C_LIGHT_BLUE, "ec": C_BLUE, "fontweight": "bold"},
    ]
    _, x_end = draw_stack(ax, blocks, y_centre=2.5, height=1.7)

    ax.set_xlim(0, x_end + 0.4)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fig. 4  Architecture of the 1D-CNN trust classifier",
                 fontsize=12, fontweight="bold", pad=10, loc="left")

    ax.text(x_end / 2, 0.4,
            "Total: ~119k parameters  *  Adam lr=1e-3  *  batch=32  *  50 epochs  *  BCE loss",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    out = os.path.join(FIG_DIR, "fig4_cnn_arch.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")



def make_fig5_dann_arch():
    fig, ax = plt.subplots(figsize=(14.5, 8.0))

    enc_y = 4.0
    add_box(ax, 0.4, enc_y - 0.7, 1.5, 1.4, "x in R^60\n60 features",
            fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5, fontweight="bold")
    add_box(ax, 2.4, enc_y - 0.7, 1.7, 1.4, "Dense 128\nReLU + BN\nDrop 0.3",
            fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5)
    add_box(ax, 4.6, enc_y - 0.7, 1.7, 1.4, "Dense 32\nReLU + BN",
            fc=C_LIGHT_BLUE, ec=C_BLUE, lw=1.5)

    phi_x, phi_y = 7.2, enc_y
    circle = plt.Circle((phi_x, phi_y), 0.6, facecolor="#fff5d0",
                        edgecolor=C_DARK, linewidth=2.2, zorder=3)
    ax.add_patch(circle)
    ax.text(phi_x, phi_y, "phi(x)\nin R^32", ha="center", va="center",
            fontsize=10, fontweight="bold")

    add_arrow(ax, 1.9, enc_y, 2.4, enc_y, lw=2)
    add_arrow(ax, 4.1, enc_y, 4.6, enc_y, lw=2)
    add_arrow(ax, 6.3, enc_y, phi_x - 0.6, enc_y, lw=2)
    ax.text(3.5, enc_y - 1.55, "Encoder  G_f  (shared, blue)",
            ha="center", fontsize=10, fontweight="bold", color=C_BLUE, style="italic")

    th_y = 6.0
    add_box(ax, 8.7, th_y - 0.55, 1.7, 1.1, "Dense 16\nReLU + Drop",
            fc=C_LIGHT_GREEN, ec=C_GREEN, lw=1.5)
    add_box(ax, 10.8, th_y - 0.55, 1.5, 1.1, "Dense 1\nSigmoid",
            fc=C_LIGHT_GREEN, ec=C_GREEN, lw=1.5)
    add_box(ax, 12.6, th_y - 0.55, 1.2, 1.1, "y_hat\ntrust prob",
            fc=C_GREEN, ec=C_GREEN, lw=1.5, fontweight="bold")
    add_arrow(ax, phi_x + 0.6, phi_y + 0.4, 8.7, th_y, color=C_GREEN, lw=2)
    add_arrow(ax, 10.4, th_y, 10.8, th_y, color=C_GREEN, lw=2)
    add_arrow(ax, 12.3, th_y, 12.6, th_y, color=C_GREEN, lw=2)
    ax.text(11.0, th_y + 1.0, "Trust head  G_y  (green)",
            ha="center", fontsize=10, fontweight="bold", color=C_GREEN, style="italic")
    ax.text(11.0, th_y - 1.05, "L_trust(y_hat, y) = binary cross-entropy",
            ha="center", fontsize=8, color=C_DARK, style="italic")

    dh_y = 2.0
    add_box(ax, 8.4, dh_y - 0.6, 2.0, 1.2, "GRL\nfwd: identity\nbwd: x(-lambda)",
            fc=C_LIGHT_RED, ec=C_RED, lw=2.4, fontweight="bold")
    add_box(ax, 10.7, dh_y - 0.55, 1.5, 1.1, "Dense 16\nReLU + Drop",
            fc=C_LIGHT_RED, ec=C_RED, lw=1.5)
    add_box(ax, 12.4, dh_y - 0.55, 1.4, 1.1, "Dense 3\nSoftmax",
            fc=C_LIGHT_RED, ec=C_RED, lw=1.5)
    ax.text(14.0, dh_y, "z_hat in\n{W, B, SA}",
            ha="left", va="center", fontsize=8, color=C_RED, fontweight="bold")

    add_arrow(ax, phi_x + 0.6, phi_y - 0.4, 8.4, dh_y, color=C_RED, lw=2)
    add_arrow(ax, 10.4, dh_y, 10.7, dh_y, color=C_RED, lw=2)
    add_arrow(ax, 12.2, dh_y, 12.4, dh_y, color=C_RED, lw=2)
    ax.text(11.4, dh_y - 0.85, "Domain head  G_d  (red, adversarial via GRL)",
            ha="center", fontsize=10, fontweight="bold", color=C_RED, style="italic")
    ax.text(11.4, dh_y - 1.15, "L_dom(z_hat, z) = categorical cross-entropy",
            ha="center", fontsize=8, color=C_DARK, style="italic")

    # LOSS LEGEND STRIP
    add_box(ax, 1.0, 0.0, 13.5, 0.65,
            r"$\mathcal{L}_{\rm total} = \mathcal{L}_{\rm trust}(\hat y, y)"
            r"\; - \; \lambda \cdot \mathcal{L}_{\rm dom}(\hat z, z)$"
            r"     $\quad\quad$  $\lambda(p) = \lambda_{\max} \cdot"
            r"\left( \frac{2}{1 + e^{-10 p}} - 1 \right)$,  $p = \rm epoch/80$,  $\lambda_{\max}{=}1$",
            fc="#fafafa", ec=C_DARK, fontsize=10)

    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Fig. 5  DANN-Trust architecture: shared encoder + trust head + GRL-protected domain head",
                 fontsize=12, fontweight="bold", pad=10, loc="left")

    out = os.path.join(FIG_DIR, "fig5_dann_arch.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print(f"Figure output directory: {FIG_DIR}")
    make_fig01_pipeline()
    make_fig3_ann_arch()
    make_fig4_cnn_arch()
    make_fig5_dann_arch()
    print("\nAll architecture figures written.")
