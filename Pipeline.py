
"""
================================================================================
 Pipeline  Dataset W/B  ->  Kruskal / Prim  ->  rVP (Reducción de Variables)
================================================================================
Aplicación de escritorio (Tkinter) que reproduce e ilustra, paso a paso, el
pipeline descrito en "Pipeline_WB_Kruskal_Prim_rVP.docx":

    Paso 1. Fusión de X7 y X8 en Y (I(X7;X8)=1.000 bits -> dependencia total)
    Paso 2. Separación por clase: Best (Y=1) / Worst (Y=0)
    Paso 3. Generación de d9_strong_b.csv / d9_strong_w.csv
    Paso 4. Matriz de Información Mutua (IM) + Árbol de Máximo Peso
            (Kruskal y Prim) por subconjunto, comparación ΣA-ΣB y
            contraste con la matriz de referencia ("pizarrón", X1-X5).

Requisitos:
    pip install pandas numpy matplotlib networkx
    (tkinter viene incluido en Python; en Linux a veces requiere:
     sudo apt-get install python3-tk)

Uso:
    python3 pipeline_gui.py
    (si no encuentra los CSV en la misma carpeta, los pedirá por diálogo)
================================================================================
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches
import networkx as nx


# ==============================================================================
#  CONFIGURACIÓN / DATOS DE REFERENCIA
# ==============================================================================

APP_TITLE = "Pipeline W/B -> Kruskal/Prim -> rVP  |  Reducción de Variables"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BEST_PATH = os.path.join(SCRIPT_DIR, "d9_strong_b.csv")
DEFAULT_WORST_PATH = os.path.join(SCRIPT_DIR, "d9_strong_w.csv")

FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_SUB = ("Segoe UI", 10, "bold")
FONT_TXT = ("Segoe UI", 10)

# Matriz de referencia del "informe anterior" (pizarrón, X1-X5), tal como
# se documenta en la sección 7 del pipeline.
PIZARRON_REF = pd.DataFrame({
    "Variable": ["X1", "X2", "X3", "X4", "X5"],
    "SumaA_Best": [24, 21, 42, 36, 33],
    "SumaB_Worst": [24, 21, 39, 36, 36],
})
PIZARRON_REF["Delta_A_B"] = PIZARRON_REF["SumaA_Best"] - PIZARRON_REF["SumaB_Worst"]
PIZARRON_REF["Discrimina"] = PIZARRON_REF["Delta_A_B"].apply(
    lambda d: "Sí (marcada) ★" if d != 0 else "No"
)


# ==============================================================================
#  LÓGICA:  Información Mutua  /  Kruskal  /  Prim  (máximo peso)
# ==============================================================================

def mutual_information(x: pd.Series, y: pd.Series) -> float:
    """I(X;Y) en bits para dos variables discretas (tabla de contingencia)."""
    ct = pd.crosstab(x, y)
    pxy = ct.values / ct.values.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px * py)
        terms = np.where(pxy > 0, pxy * np.log2(ratio), 0.0)
    return float(terms.sum())


def compute_mi_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Matriz simétrica de Información Mutua por pares de columnas."""
    cols = list(df.columns)
    n = len(cols)
    M = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    for i, j in combinations(range(n), 2):
        v = mutual_information(df.iloc[:, i], df.iloc[:, j])
        M.iloc[i, j] = v
        M.iloc[j, i] = v
    return M


class UnionFind:
    """Estructura Union-Find usada por Kruskal para detectar ciclos."""

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def kruskal_max_spanning_tree_steps(M: pd.DataFrame):
    """
    Genera la lista de pasos del algoritmo de KRUSKAL de máximo peso:
    ordena todas las aristas de mayor a menor peso (IM) y las acepta si
    no cierran un ciclo (Union-Find). Se detiene al completar el árbol
    (n-1 aristas).
    """
    cols = list(M.columns)
    edges = [(cols[i], cols[j], M.iloc[i, j]) for i, j in combinations(range(len(cols)), 2)]
    edges.sort(key=lambda e: -e[2])

    uf = UnionFind(cols)
    steps = []
    mst_edges = []
    total = 0.0
    for (u, v, w) in edges:
        cycle = uf.find(u) == uf.find(v)
        accepted = False
        if not cycle:
            uf.union(u, v)
            mst_edges.append((u, v, w))
            total += w
            accepted = True
        steps.append(dict(
            edge=(u, v), weight=w, accepted=accepted, is_cycle=cycle,
            mst_so_far=list(mst_edges), total_weight=total,
            done=(len(mst_edges) == len(cols) - 1),
        ))
        if len(mst_edges) == len(cols) - 1:
            break
    return steps


def prim_max_spanning_tree_steps(M: pd.DataFrame, start=None):
    """
    Genera la lista de pasos del algoritmo de PRIM de máximo peso:
    parte de `start`, y en cada paso agrega al árbol la arista de mayor
    peso que cruza el corte (nodos visitados / no visitados).
    """
    cols = list(M.columns)
    start = start if start in cols else cols[0]
    visited_set = {start}
    mst_edges = []
    total = 0.0
    steps = []
    while len(visited_set) < len(cols):
        candidates = []
        for u in visited_set:
            for v in cols:
                if v not in visited_set:
                    candidates.append((u, v, M.loc[u, v]))
        candidates.sort(key=lambda e: -e[2])
        chosen = candidates[0]
        u, v, w = chosen
        visited_set.add(v)
        mst_edges.append((u, v, w))
        total += w
        steps.append(dict(
            candidates=candidates, chosen=chosen, visited=set(visited_set),
            mst_so_far=list(mst_edges), total_weight=total,
            done=(len(visited_set) == len(cols)),
        ))
    return steps


# ==============================================================================
#  UTILIDADES DE CARGA DE DATOS
# ==============================================================================

REQUIRED_COLS = ["x1", "x2", "x3", "x4", "x5", "x6", "x9"]


def load_csv_safe(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"El archivo '{os.path.basename(path)}' no tiene las columnas "
            f"esperadas {REQUIRED_COLS}. Faltan: {missing}"
        )
    return df[REQUIRED_COLS].copy()


def compute_layout(M: pd.DataFrame, seed=42):
    """Layout fijo (spring layout) para dibujar el grafo completo ponderado."""
    G = nx.Graph()
    cols = list(M.columns)
    G.add_nodes_from(cols)
    for i, j in combinations(range(len(cols)), 2):
        G.add_edge(cols[i], cols[j], weight=float(M.iloc[i, j]) + 0.001)
    pos = nx.spring_layout(G, weight="weight", seed=seed, k=1.3 / np.sqrt(len(cols)))
    return G, pos


# ==============================================================================
#  WIDGET REUTILIZABLE: PANEL DE GRAFO (matplotlib embebido)
# ==============================================================================

class GraphPanel(ttk.Frame):
    """Panel con un canvas de matplotlib para dibujar el grafo ponderado y
    resaltar el MST parcial/actual durante la animación de Kruskal/Prim."""

    def __init__(self, master, figsize=(6.4, 5.6)):
        super().__init__(master)
        self.fig = Figure(figsize=figsize, dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def draw_graph(self, G, pos, mst_edges, highlight_edge=None,
                   highlight_state=None, visited=None, title=""):
        """
        G, pos            : grafo completo + posiciones fijas de nodos
        mst_edges         : lista [(u,v,w), ...] ya incorporadas al árbol
        highlight_edge    : (u,v) arista evaluada en el paso actual (o None)
        highlight_state   : "accepted" | "rejected" | "chosen" | None
        visited           : set de nodos visitados (solo para Prim); si es
                            None, todos los nodos se pintan igual (Kruskal)
        """
        ax = self.ax
        ax.clear()
        ax.set_axis_off()

        # 1) Fondo: todas las aristas del grafo completo, tenues
        for (u, v, data) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#c9d3dc", linewidth=1.0,
                    alpha=0.55, zorder=1)

        # 2) Aristas ya incorporadas al MST: gruesas, verdes, con etiqueta
        mst_set = {(u, v) for (u, v, _) in mst_edges} | {(v, u) for (u, v, _) in mst_edges}
        for (u, v, w) in mst_edges:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], color="#1f9d55", linewidth=3.6,
                    alpha=0.95, zorder=2, solid_capstyle="round")
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx, my, f"{w:.4f}", fontsize=8, color="#0b5e33",
                    ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#1f9d55", lw=0.6))

        # 3) Arista actualmente evaluada (resaltada)
        if highlight_edge is not None and highlight_edge not in mst_set:
            u, v = highlight_edge
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            color = {"accepted": "#1f9d55", "chosen": "#1f9d55",
                     "rejected": "#d64545"}.get(highlight_state, "#e08a1e")
            style = "--" if highlight_state == "rejected" else "-"
            w = G[u][v]["weight"] - 0.001
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=3.2,
                    linestyle=style, alpha=0.95, zorder=3)
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            tag = {"accepted": "ACEPTADA", "chosen": "ELEGIDA",
                   "rejected": "RECHAZADA\n(forma ciclo)"}.get(highlight_state, "EVALUANDO")
            ax.text(mx, my, f"{w:.4f}\n{tag}", fontsize=8, color=color,
                    ha="center", va="center", zorder=6, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.0))

        # 4) Nodos
        touched = set()
        for (u, v, _) in mst_edges:
            touched.add(u)
            touched.add(v)
        for node in G.nodes():
            x, y = pos[node]
            if visited is not None:
                if node in visited:
                    fc = "#2563eb" if node == list(visited)[0] and len(visited) == 1 else "#60a5fa"
                else:
                    fc = "#e5e7eb"
            else:
                fc = "#60a5fa" if node in touched else "#e5e7eb"
            ax.scatter([x], [y], s=1250, facecolor=fc, edgecolor="#1e293b",
                       linewidth=1.4, zorder=4)
            ax.text(x, y, node.upper(), ha="center", va="center", zorder=5,
                    fontsize=11, fontweight="bold", color="#0f172a")

        ax.set_title(title, fontsize=11, fontweight="bold", color="#0f172a")
        ax.margins(0.18)
        self.canvas.draw()


# ==============================================================================
#  TAB 1 - RESUMEN DEL PIPELINE
# ==============================================================================

class ResumenTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        pan.add(left, weight=1)
        pan.add(right, weight=1)

        ttk.Label(left, text="Pipeline: Dataset W/B -> Kruskal/Prim -> rVP",
                  font=FONT_TITLE).pack(anchor="w", pady=(0, 8))

        texto = (
            "Este programa reproduce e ilustra el pipeline aplicado sobre "
            "d9_strong.csv:\n\n"
            "PASO 1 - Fusión X7/X8 -> Y\n"
            "  I(X7;X8) ~ 1.000 bits (dependencia total): ambas columnas son "
            "redundantes y se fusionan en una sola etiqueta binaria Y (Y=X7=X8). "
            "El conjunto pasa de 9 a 8 columnas efectivas.\n\n"
            "PASO 2 - Separación por clase\n"
            "  Usando Y como criterio de partición: Best = {Y=1} (502 filas), "
            "Worst = {Y=0} (498 filas). Ambos subconjuntos conservan las 7 "
            "variables predictoras restantes: x1, x2, x3, x4, x5, x6, x9.\n\n"
            "PASO 3 - Generación de CSV\n"
            "  d9_strong_b.csv (Best) y d9_strong_w.csv (Worst), ya cargados "
            "en esta aplicación (pestaña 'Datos').\n\n"
            "PASO 4 - Matriz IM + MST (Kruskal / Prim) por subconjunto\n"
            "  Se calcula la matriz de Información Mutua de cada subconjunto "
            "y se construye el Árbol de Máximo Peso (Maximum Spanning Tree) "
            "con Kruskal y con Prim (pestañas 3, 4 y 5). Después se comparan "
            "las sumas ΣA-ΣB por variable (pestaña 6) y se contrastan con la "
            "matriz de referencia del informe anterior, el 'pizarrón' de "
            "X1-X5 (pestaña 7), para concluir cuántas variables se pueden "
            "reducir de forma segura (pestaña 8, rVP)."
        )
        txt = tk.Text(left, wrap="word", font=FONT_TXT, height=28, relief="flat",
                       bg="#f8fafc", padx=10, pady=10)
        txt.insert("1.0", texto)
        txt.configure(state="disabled")
        txt.pack(fill="both", expand=True)

        ttk.Label(right, text="Diagrama de flujo", font=FONT_SUB).pack(anchor="w")
        fig = Figure(figsize=(6, 6.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        stages = [
            "X1..X9\n(9 variables)",
            "Fusión X7≡X8 -> Y\n(8 columnas efectivas)",
            "Split por Y\nBest (Y=1) / Worst (Y=0)",
            "Matriz IM\n+ MST (Kruskal / Prim)\npor subconjunto",
            "Comparación ΣA-ΣB\nBest vs Worst\n+ vs Pizarrón (X1-X5)",
            "Conclusión rVP\n(variables a reducir)",
        ]
        n = len(stages)
        for i, s in enumerate(stages):
            y = 1 - i / (n - 1)
            box = mpatches.FancyBboxPatch((0.08, y - 0.055), 0.84, 0.09,
                                           boxstyle="round,pad=0.02",
                                           linewidth=1.4, edgecolor="#1e3a8a",
                                           facecolor="#dbeafe")
            ax.add_patch(box)
            ax.text(0.5, y - 0.01, s, ha="center", va="center", fontsize=9.5,
                     fontweight="bold", color="#0f172a")
            if i < n - 1:
                ax.annotate("", xy=(0.5, y - 0.075), xytext=(0.5, y - 0.055),
                            arrowprops=dict(arrowstyle="-|>", color="#1e3a8a", lw=1.6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        canvas = FigureCanvasTkAgg(fig, master=right)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()


# ==============================================================================
#  TAB 2 - DATOS Y FUSIÓN
# ==============================================================================

class DataTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Cargar Best CSV...", command=self.load_best).pack(side="left", padx=4)
        ttk.Button(top, text="Cargar Worst CSV...", command=self.load_worst).pack(side="left", padx=4)
        ttk.Button(top, text="Recalcular todo", command=self.app.recompute_all).pack(side="left", padx=12)

        self.info_lbl = ttk.Label(self, text="", font=FONT_SUB, foreground="#0f172a")
        self.info_lbl.pack(anchor="w", padx=8, pady=(4, 8))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        best_frame = ttk.LabelFrame(body, text="Best (Y=1) - vista previa")
        worst_frame = ttk.LabelFrame(body, text="Worst (Y=0) - vista previa")
        best_frame.pack(side="left", fill="both", expand=True, padx=4)
        worst_frame.pack(side="left", fill="both", expand=True, padx=4)

        self.tree_best = self._make_tree(best_frame)
        self.tree_worst = self._make_tree(worst_frame)

        note = (
            "Nota: estos CSV ya reflejan los pasos 1-3 del pipeline (fusión "
            "X7≡X8 -> Y, partición por clase y descarte de X7/X8 como columnas "
            "independientes). Columnas conservadas: x1, x2, x3, x4, x5, x6, x9."
        )
        ttk.Label(self, text=note, font=FONT_TXT, foreground="#475569",
                  wraplength=1150, justify="left").pack(anchor="w", padx=8, pady=(6, 8))

        self.redund_lbl = ttk.Label(self, text="", font=FONT_TXT, foreground="#7c2d12")
        self.redund_lbl.pack(anchor="w", padx=8, pady=(0, 8))

    def _make_tree(self, parent):
        cols = REQUIRED_COLS
        tree = ttk.Treeview(parent, columns=cols, show="headings", height=14)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=55, anchor="center")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return tree

    def load_best(self):
        path = filedialog.askopenfilename(title="Seleccionar d9_strong_b.csv",
                                           filetypes=[("CSV", "*.csv")])
        if path:
            self.app.best_path = path
            self.app.recompute_all()

    def load_worst(self):
        path = filedialog.askopenfilename(title="Seleccionar d9_strong_w.csv",
                                           filetypes=[("CSV", "*.csv")])
        if path:
            self.app.worst_path = path
            self.app.recompute_all()

    def refresh(self):
        b, w = self.app.df_best, self.app.df_worst
        self.info_lbl.config(
            text=(f"Best: {b.shape[0]} filas x {b.shape[1]} columnas   |   "
                  f"Worst: {w.shape[0]} filas x {w.shape[1]} columnas   |   "
                  f"Total: {b.shape[0] + w.shape[0]} filas")
        )
        for tree, df in [(self.tree_best, b), (self.tree_worst, w)]:
            tree.delete(*tree.get_children())
            for _, row in df.head(150).iterrows():
                tree.insert("", "end", values=list(row))

        match_b = (b["x9"] == 2 * b["x4"] + b["x6"]).mean() * 100
        match_w = (w["x9"] == 2 * w["x4"] + w["x6"]).mean() * 100
        self.redund_lbl.config(
            text=(f"Verificación de redundancia adicional (hallazgo del pipeline, paso 8): "
                  f"x9 = 2*x4 + x6  -> coincide en el {match_b:.1f}% de filas en Best "
                  f"y {match_w:.1f}% en Worst. Si el porcentaje es ~100%, x9 es una "
                  f"variable derivada de (x4, x6) y por tanto redundante.")
        )


# ==============================================================================
#  TAB 3 - MATRIZ DE INFORMACIÓN MUTUA (Best vs Worst)
# ==============================================================================

class MatrixTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="Matriz de Información Mutua I(Xi;Xj) en bits, por subconjunto",
                  font=FONT_SUB).pack(side="left")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        self.fig = Figure(figsize=(12.4, 5.6), dpi=100)
        self.ax_best = self.fig.add_subplot(121)
        self.ax_worst = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _heatmap(self, ax, M, title):
        ax.clear()
        cols = list(M.columns)
        data = M.values
        im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=max(0.05, data.max()))
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels([c.upper() for c in cols])
        ax.set_yticklabels([c.upper() for c in cols])
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i == j:
                    continue
                v = data[i, j]
                color = "white" if v > data.max() * 0.55 else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=7.6, color=color)
        sums = M.sum(axis=1)
        subtitle = "  ".join(f"Σ{c.upper()}={sums[c]:.3f}" for c in cols)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.3, fontweight="bold")
        return im

    def refresh(self):
        self._heatmap(self.ax_best, self.app.M_best, "BEST (Y=1)")
        self._heatmap(self.ax_worst, self.app.M_worst, "WORST (Y=0)")
        self.fig.tight_layout()
        self.canvas.draw()


# ==============================================================================
#  TAB 4/5 - STEPPER GENÉRICO PARA KRUSKAL Y PRIM (paso a paso, animado)
# ==============================================================================

class MSTStepperTab(ttk.Frame):
    def __init__(self, master, app, algorithm):
        super().__init__(master)
        self.app = app
        self.algorithm = algorithm          # "kruskal" | "prim"
        self.steps = []
        self.idx = -1                       # -1 = estado inicial
        self.playing = False
        self.G = None
        self.pos = None
        self._build()

    # ---------------------------------------------------------- construcción
    def _build(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Subconjunto:", font=FONT_TXT).pack(side="left")
        self.subset_var = tk.StringVar(value="BEST")
        cmb = ttk.Combobox(top, textvariable=self.subset_var, values=["BEST", "WORST"],
                            state="readonly", width=8)
        cmb.pack(side="left", padx=(4, 14))
        cmb.bind("<<ComboboxSelected>>", lambda e: self.reload_steps())

        if self.algorithm == "prim":
            ttk.Label(top, text="Nodo inicial:", font=FONT_TXT).pack(side="left")
            self.start_var = tk.StringVar(value="x1")
            self.start_cmb = ttk.Combobox(top, textvariable=self.start_var,
                                           values=REQUIRED_COLS, state="readonly", width=6)
            self.start_cmb.pack(side="left", padx=(4, 14))
            self.start_cmb.bind("<<ComboboxSelected>>", lambda e: self.reload_steps())

        btns = ttk.Frame(top)
        btns.pack(side="right")
        ttk.Button(btns, text="<< Reiniciar", command=self.reset).pack(side="left", padx=2)
        ttk.Button(btns, text="< Anterior", command=self.prev_step).pack(side="left", padx=2)
        self.play_btn = ttk.Button(btns, text="Reproducir >", command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)
        ttk.Button(btns, text="Siguiente >", command=self.next_step).pack(side="left", padx=2)
        ttk.Button(btns, text="Fin >>", command=self.go_end).pack(side="left", padx=2)

        ttk.Label(top, text="Velocidad:", font=FONT_TXT).pack(side="right", padx=(10, 4))
        self.speed_var = tk.IntVar(value=800)
        ttk.Scale(top, from_=250, to=1800, orient="horizontal", length=110,
                  variable=self.speed_var).pack(side="right")

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=400)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.graph_panel = GraphPanel(left)
        self.graph_panel.pack(fill="both", expand=True)

        ttk.Label(right, text="Estado del algoritmo", font=FONT_SUB).pack(anchor="w")
        self.status_lbl = ttk.Label(right, text="", font=FONT_TXT, wraplength=380,
                                     justify="left", foreground="#0f172a")
        self.status_lbl.pack(anchor="w", pady=(2, 10))

        list_title = ("Aristas ordenadas por peso (Kruskal)" if self.algorithm == "kruskal"
                      else "Candidatos del corte actual (frontera de Prim)")
        ttk.Label(right, text=list_title, font=FONT_SUB).pack(anchor="w")
        self.tree = ttk.Treeview(right, columns=("edge", "w", "action"),
                                   show="headings", height=13)
        self.tree.heading("edge", text="Arista")
        self.tree.heading("w", text="Peso (bits)")
        self.tree.heading("action", text="Estado")
        self.tree.column("edge", width=90, anchor="center")
        self.tree.column("w", width=90, anchor="center")
        self.tree.column("action", width=170, anchor="center")
        self.tree.pack(fill="both", expand=False, pady=(2, 8))
        self.tree.tag_configure("accepted", background="#dcfce7")
        self.tree.tag_configure("rejected", background="#fee2e2")
        self.tree.tag_configure("current", background="#fde68a")
        self.tree.tag_configure("chosen", background="#dcfce7")

        ttk.Label(right, text="Árbol construido hasta el momento (orden)", font=FONT_SUB).pack(anchor="w")
        self.hist = ttk.Treeview(right, columns=("n", "edge", "w", "acc"),
                                   show="headings", height=8)
        for c, t, wd in [("n", "#", 30), ("edge", "Arista", 80),
                          ("w", "Peso", 70), ("acc", "Acumulado", 80)]:
            self.hist.heading(c, text=t)
            self.hist.column(c, width=wd, anchor="center")
        self.hist.pack(fill="both", expand=False, pady=(2, 8))

        self.total_lbl = ttk.Label(right, text="Peso total del MST: 0.0000 bits",
                                    font=FONT_SUB, foreground="#0b5e33")
        self.total_lbl.pack(anchor="w")

    # ---------------------------------------------------------- datos / pasos
    def reload_steps(self):
        M = self.app.M_best if self.subset_var.get() == "BEST" else self.app.M_worst
        self.G, self.pos = compute_layout(M)
        if self.algorithm == "kruskal":
            self.steps = kruskal_max_spanning_tree_steps(M)
            self._populate_kruskal_tree()
        else:
            self.steps = prim_max_spanning_tree_steps(M, start=self.start_var.get())
        self.reset()

    def _populate_kruskal_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, s in enumerate(self.steps):
            u, v = s["edge"]
            iid = f"row{i}"
            self.tree.insert("", "end", iid=iid,
                              values=(f"{u.upper()}-{v.upper()}", f"{s['weight']:.4f}", "pendiente"))

    # ---------------------------------------------------------- navegación
    def reset(self):
        self.idx = -1
        self.playing = False
        self.play_btn.config(text="Reproducir >")
        self._render()

    def go_end(self):
        self.idx = len(self.steps) - 1
        self._render()

    def next_step(self):
        if self.idx < len(self.steps) - 1:
            self.idx += 1
            self._render()
        else:
            self.playing = False
            self.play_btn.config(text="Reproducir >")

    def prev_step(self):
        if self.idx > -1:
            self.idx -= 1
            self._render()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="Pausa ||" if self.playing else "Reproducir >")
        if self.playing:
            self._play_tick()

    def _play_tick(self):
        if not self.playing:
            return
        if self.idx < len(self.steps) - 1:
            self.next_step()
            self.after(int(self.speed_var.get()), self._play_tick)
        else:
            self.playing = False
            self.play_btn.config(text="Reproducir >")

    # ---------------------------------------------------------- render
    def _render(self):
        if not self.steps:
            return
        subset = self.subset_var.get()
        n_steps = len(self.steps)

        if self.idx == -1:
            mst_so_far = []
            highlight_edge = None
            highlight_state = None
            visited = {self.start_var.get()} if self.algorithm == "prim" else None
            if self.algorithm == "kruskal":
                status = (f"Estado inicial. Se evaluarán {n_steps} aristas ordenadas "
                          f"de mayor a menor peso (Información Mutua). Presione "
                          f"'Siguiente' para comenzar.")
            else:
                status = (f"Estado inicial. Prim parte del nodo {self.start_var.get().upper()} "
                          f"y en cada paso agrega la arista de mayor peso que conecta el "
                          f"árbol con un nodo aún no visitado.")
            total = 0.0
        else:
            s = self.steps[self.idx]
            mst_so_far = s["mst_so_far"]
            total = s["total_weight"]
            if self.algorithm == "kruskal":
                u, v = s["edge"]
                highlight_edge = (u, v)
                highlight_state = "accepted" if s["accepted"] else "rejected"
                if s["accepted"]:
                    status = (f"Paso {self.idx + 1}/{n_steps}: arista {u.upper()}-{v.upper()} "
                              f"(peso={s['weight']:.4f}). {u.upper()} y {v.upper()} están en "
                              f"componentes distintas -> NO cierra ciclo -> ACEPTADA. "
                              f"Árbol actual: {len(mst_so_far)} arista(s), "
                              f"peso acumulado={total:.4f} bits.")
                else:
                    status = (f"Paso {self.idx + 1}/{n_steps}: arista {u.upper()}-{v.upper()} "
                              f"(peso={s['weight']:.4f}). {u.upper()} y {v.upper()} ya están "
                              f"en la misma componente -> cerraría un CICLO -> RECHAZADA. "
                              f"Árbol actual sin cambios: {len(mst_so_far)} arista(s).")
                visited = None
            else:
                u, v, w = s["chosen"]
                highlight_edge = (u, v)
                highlight_state = "chosen"
                visited = s["visited"]
                status = (f"Paso {self.idx + 1}/{n_steps}: de todas las aristas que cruzan "
                          f"el corte (árbol <-> resto), la de mayor peso es "
                          f"{u.upper()}-{v.upper()} (peso={w:.4f}). Se agrega "
                          f"{v.upper()} al árbol. Nodos visitados: "
                          f"{', '.join(sorted(x.upper() for x in visited))}. "
                          f"Peso acumulado={total:.4f} bits.")

        title = f"{'Kruskal' if self.algorithm == 'kruskal' else 'Prim'} - {subset} - paso {max(self.idx + 1, 0)}/{n_steps}"
        self.graph_panel.draw_graph(self.G, self.pos, mst_so_far,
                                     highlight_edge=highlight_edge,
                                     highlight_state=highlight_state,
                                     visited=visited, title=title)
        self.status_lbl.config(text=status)
        self.total_lbl.config(text=f"Peso total del MST: {total:.4f} bits  "
                                    f"({len(mst_so_far)}/{len(REQUIRED_COLS) - 1} aristas)")

        if self.algorithm == "kruskal":
            for i, s in enumerate(self.steps):
                iid = f"row{i}"
                u, v = s["edge"]
                if i < self.idx or (i == self.idx and self.idx >= 0):
                    state = "Aceptada" if s["accepted"] else "Rechazada (ciclo)"
                    tag = "current" if i == self.idx else ("accepted" if s["accepted"] else "rejected")
                else:
                    state = "pendiente"
                    tag = ""
                self.tree.item(iid, values=(f"{u.upper()}-{v.upper()}", f"{s['weight']:.4f}", state),
                                tags=(tag,))
            if self.idx >= 0:
                self.tree.see(f"row{self.idx}")
        else:
            self.tree.delete(*self.tree.get_children())
            if self.idx >= 0:
                cands = self.steps[self.idx]["candidates"]
                chosen = self.steps[self.idx]["chosen"]
                for (u, v, w) in cands:
                    tag = "chosen" if (u, v, w) == chosen else ""
                    label = "ELEGIDA (máx. peso)" if (u, v, w) == chosen else "candidata"
                    self.tree.insert("", "end", values=(f"{u.upper()}-{v.upper()}", f"{w:.4f}", label),
                                      tags=(tag,))
            else:
                self.tree.insert("", "end", values=("-", "-", "presione Siguiente"))

        self.hist.delete(*self.hist.get_children())
        acc = 0.0
        for i, (u, v, w) in enumerate(mst_so_far, start=1):
            acc += w
            self.hist.insert("", "end", values=(i, f"{u.upper()}-{v.upper()}", f"{w:.4f}", f"{acc:.4f}"))


# ==============================================================================
#  TAB 6 - COMPARACIÓN BEST vs WORST (ΣA - ΣB)
# ==============================================================================

class CompareBWTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        pan.add(left, weight=1)
        pan.add(right, weight=1)

        ttk.Label(left, text="ΣA (Best) vs ΣB (Worst) por variable", font=FONT_SUB).pack(anchor="w")
        cols = ("var", "sa", "sb", "delta", "disc")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=10)
        headers = {"var": "Variable", "sa": "ΣA (Best)", "sb": "ΣB (Worst)",
                   "delta": "ΣA-ΣB", "disc": "¿Discrimina? (|Δ|>=0.02)"}
        for c in cols:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=140, anchor="center")
        self.tree.pack(fill="both", expand=True, pady=6)

        self.expl_lbl = ttk.Label(left, text="", font=FONT_TXT, wraplength=560,
                                   justify="left", foreground="#334155")
        self.expl_lbl.pack(anchor="w", pady=(6, 0))

        self.fig = Figure(figsize=(6, 5.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh(self):
        M_b, M_w = self.app.M_best, self.app.M_worst
        sa = M_b.sum(axis=1)
        sb = M_w.sum(axis=1)
        delta = sa - sb
        self.tree.delete(*self.tree.get_children())
        rows = []
        for c in REQUIRED_COLS:
            disc = "Sí (relevante)" if abs(delta[c]) >= 0.02 else "No (≈0)"
            rows.append((c.upper(), sa[c], sb[c], delta[c], disc))
        for (v, a, b, d, disc) in sorted(rows, key=lambda r: -abs(r[3])):
            self.tree.insert("", "end", values=(v, f"{a:.4f}", f"{b:.4f}", f"{d:+.4f}", disc))

        n_disc = sum(1 for r in rows if abs(r[3]) >= 0.02)
        self.expl_lbl.config(text=(
            f"Se marcan como discriminantes las variables con |ΣA-ΣB| >= 0.02 bits "
            f"(umbral orientativo, igual criterio que el informe de referencia). "
            f"En este dataset resultan discriminantes {n_disc} de {len(rows)} variables. "
            f"Las variables con Δ≈0 tienen un rol estructural casi idéntico en Best "
            f"y Worst, y son las primeras candidatas a reducción."
        ))

        ordered = sorted(rows, key=lambda r: r[3])
        vars_ = [r[0] for r in ordered]
        deltas = [r[3] for r in ordered]
        colors = ["#dc2626" if d < 0 else "#16a34a" for d in deltas]
        self.ax.clear()
        self.ax.barh(vars_, deltas, color=colors)
        self.ax.axvline(0, color="#0f172a", linewidth=1)
        self.ax.axvline(0.02, color="#94a3b8", linestyle="--", linewidth=1)
        self.ax.axvline(-0.02, color="#94a3b8", linestyle="--", linewidth=1)
        self.ax.set_title("ΣA - ΣB por variable\n(verde: mayor en Best · rojo: mayor en Worst)",
                           fontsize=10, fontweight="bold")
        self.ax.set_xlabel("ΣA - ΣB (bits)")
        self.fig.tight_layout()
        self.canvas.draw()


# ==============================================================================
#  TAB 7 - COMPARACIÓN CON LA MATRIZ DE REFERENCIA (PIZARRÓN, X1-X5)
# ==============================================================================

class ComparePizarronTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        ttk.Label(self, text="Contraste con la matriz de referencia (informe anterior, pizarrón, X1-X5)",
                  font=FONT_SUB).pack(anchor="w", padx=8, pady=(8, 4))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.LabelFrame(body, text="Referencia - Pizarrón (X1-X5)")
        left.pack(side="left", fill="both", expand=True, padx=4)
        cols = ("var", "sa", "sb", "delta", "disc")
        self.tree_ref = ttk.Treeview(left, columns=cols, show="headings", height=8)
        headers = {"var": "Variable", "sa": "ΣA (Best)", "sb": "ΣB (Worst)",
                   "delta": "ΣA-ΣB", "disc": "Marcada"}
        for c in cols:
            self.tree_ref.heading(c, text=headers[c])
            self.tree_ref.column(c, width=110, anchor="center")
        self.tree_ref.pack(fill="both", expand=True, pady=6, padx=6)
        for _, r in PIZARRON_REF.iterrows():
            self.tree_ref.insert("", "end", values=(r["Variable"], r["SumaA_Best"], r["SumaB_Worst"],
                                                      f"{r['Delta_A_B']:+d}", r["Discrimina"]))

        right = ttk.LabelFrame(body, text="Dataset actual - mismas variables (X1-X5) en bits")
        right.pack(side="left", fill="both", expand=True, padx=4)
        self.tree_now = ttk.Treeview(right, columns=cols, show="headings", height=8)
        for c in cols:
            self.tree_now.heading(c, text=headers[c])
            self.tree_now.column(c, width=110, anchor="center")
        self.tree_now.pack(fill="both", expand=True, pady=6, padx=6)

        note = (
            "Los valores numéricos no son comparables en escala directa (el pizarrón "
            "usa otra muestra/unidad), pero sí lo es el patrón cualitativo: qué "
            "variables tienen Δ≈0 (neutras / candidatas a reducción) frente a las "
            "que muestran una diferencia clara entre Best y Worst (discriminantes). "
            "En el informe de referencia, {X3, X5} resultaron discriminantes y "
            "{X1, X2, X4} neutras."
        )
        ttk.Label(self, text=note, font=FONT_TXT, wraplength=1150, justify="left",
                  foreground="#334155").pack(anchor="w", padx=8, pady=(4, 8))

    def refresh(self):
        M_b, M_w = self.app.M_best, self.app.M_worst
        sub_b = M_b.loc[["x1", "x2", "x3", "x4", "x5"], ["x1", "x2", "x3", "x4", "x5"]]
        sub_w = M_w.loc[["x1", "x2", "x3", "x4", "x5"], ["x1", "x2", "x3", "x4", "x5"]]
        sa = sub_b.sum(axis=1)
        sb = sub_w.sum(axis=1)
        delta = sa - sb
        self.tree_now.delete(*self.tree_now.get_children())
        for c in ["x1", "x2", "x3", "x4", "x5"]:
            disc = "Sí (relevante)" if abs(delta[c]) >= 0.02 else "No (≈0)"
            self.tree_now.insert("", "end", values=(c.upper(), f"{sa[c]:.4f}", f"{sb[c]:.4f}",
                                                       f"{delta[c]:+.4f}", disc))


# ==============================================================================
#  TAB 8 - CONCLUSIÓN rVP (Reducción de Variables)
# ==============================================================================

class ConclusionTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(pan)
        right = ttk.Frame(pan)
        pan.add(left, weight=1)
        pan.add(right, weight=1)

        ttk.Label(left, text="Conclusión - Reducción de Variables (rVP)", font=FONT_TITLE).pack(anchor="w")
        self.txt = tk.Text(left, wrap="word", font=FONT_TXT, height=30, relief="flat",
                            bg="#f8fafc", padx=10, pady=10)
        self.txt.pack(fill="both", expand=True, pady=(6, 0))

        ttk.Label(right, text="De 9 variables originales a variables mínimas no redundantes",
                  font=FONT_SUB).pack(anchor="w")
        self.fig = Figure(figsize=(6, 6.4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh(self):
        M_b, M_w = self.app.M_best, self.app.M_worst
        sa = M_b.sum(axis=1)
        sb = M_w.sum(axis=1)
        delta = (sa - sb).abs()
        low_disc = sorted([c for c in REQUIRED_COLS if delta[c] < 0.02 and c != "x9"])
        b, w = self.app.df_best, self.app.df_worst
        match = pd.concat([b, w])
        pct = (match["x9"] == 2 * match["x4"] + match["x6"]).mean() * 100

        texto = (
            "1) Fusión X7 ≡ X8 -> Y\n"
            "   I(X7;X8) ≈ 1.000 bits en el dataset original -> dependencia total. "
            "Se fusionan en una sola etiqueta Y. Reduce 1 variable "
            "(de 9 a 8 columnas efectivas).\n\n"
            "2) Redundancia determinista X9 = 2*X4 + X6\n"
            f"   Verificado sobre los datos cargados: coincide en el {pct:.1f}% de las "
            "filas (Best + Worst). X9 no aporta información nueva respecto de "
            "{X4, X6}: es una codificación determinista del par binario. "
            "Reduce 1 variable más (de 8 a 7, quitando X9 y conservando X4, X6).\n\n"
            "3) Backbone estructural idéntico en Best y Worst\n"
            "   El MST de máximo peso (Kruskal y Prim, pestañas 4 y 5) coincide en "
            "sus 4 aristas principales entre Best y Worst: X6-X9, X4-X9, X2-X5, "
            "X1-X5. Solo cambia la arista más débil (la que conecta a X3, peso "
            "≈0.002-0.006 bits, prácticamente ruido).\n\n"
            "4) Baja discriminancia (ΣA ≈ ΣB)\n"
            f"   Con el umbral |Δ|<0.02 bits sobre los datos cargados, resultan "
            f"casi invariantes entre clases: {', '.join(v.upper() for v in low_disc) if low_disc else '(ninguna con este umbral)'}. "
            "Estas variables aportan escasa capacidad discriminante entre Best y "
            "Worst y son candidatas a una reducción adicional (opcional).\n\n"
            "RESULTADO FINAL\n"
            "   De las 9 variables originales (X1-X9) se pueden reducir de forma "
            "segura 2 variables por redundancia total demostrada: X8 (fusionada "
            "en Y) y X9 (derivada de X4 y X6). Queda un conjunto mínimo no "
            "redundante de 6 variables predictoras {X1, X2, X3, X4, X5, X6} + la "
            "etiqueta Y.\n"
            "   Si además se aplica el criterio de baja discriminancia (ΣA≈ΣB), "
            "las variables listadas arriba podrían evaluarse para una reducción "
            "adicional opcional, dejando un núcleo mínimo de variables "
            "verdaderamente informativas para distinguir Best de Worst "
            "(típicamente X2, X4, X5 y X6/X9), sujeto a validación con más datos "
            "o un umbral formal de significancia."
        )
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", texto)
        self.txt.configure(state="disabled")

        ax = self.ax
        ax.clear()
        ax.set_axis_off()
        stages = [
            ("9 variables\n(X1..X9)", 1.0),
            ("Fusión X7≡X8 -> Y\n8 columnas efectivas", 0.85),
            ("X9 = f(X4,X6)\nredundante -> se elimina\n7 columnas", 0.7),
            ("Núcleo mínimo\nno redundante:\nX1,X2,X3,X4,X5,X6 + Y\n(6 variables)", 0.5),
            ("Reducción opcional\n(baja discriminancia ΣA≈ΣB):\n" +
             (", ".join(v.upper() for v in low_disc) if low_disc else "—"), 0.3),
        ]
        n = len(stages)
        for i, (s, _) in enumerate(stages):
            y = 1 - i / (n - 1) * 0.92
            fc = "#dbeafe" if i < 3 else ("#dcfce7" if i == 3 else "#fef3c7")
            box = mpatches.FancyBboxPatch((0.06, y - 0.075), 0.88, 0.13,
                                           boxstyle="round,pad=0.02",
                                           linewidth=1.4, edgecolor="#1e3a8a", facecolor=fc)
            ax.add_patch(box)
            ax.text(0.5, y - 0.01, s, ha="center", va="center", fontsize=9,
                     fontweight="bold", color="#0f172a")
            if i < n - 1:
                ax.annotate("", xy=(0.5, y - 0.16), xytext=(0.5, y - 0.075),
                            arrowprops=dict(arrowstyle="-|>", color="#1e3a8a", lw=1.6))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        self.canvas.draw()


# ==============================================================================
#  APLICACIÓN PRINCIPAL
# ==============================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1420x880")
        self.minsize(1150, 720)

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.best_path = DEFAULT_BEST_PATH
        self.worst_path = DEFAULT_WORST_PATH
        self.df_best = None
        self.df_worst = None
        self.M_best = None
        self.M_worst = None

        if not self._ask_paths_if_missing():
            self.destroy()
            return

        self._load_data()

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)
        self.nb = nb

        self.tab_resumen = ResumenTab(nb, self)
        self.tab_datos = DataTab(nb, self)
        self.tab_matrix = MatrixTab(nb, self)
        self.tab_kruskal = MSTStepperTab(nb, self, "kruskal")
        self.tab_prim = MSTStepperTab(nb, self, "prim")
        self.tab_compare = CompareBWTab(nb, self)
        self.tab_pizarron = ComparePizarronTab(nb, self)
        self.tab_conclusion = ConclusionTab(nb, self)

        nb.add(self.tab_resumen, text="1. Resumen del Pipeline")
        nb.add(self.tab_datos, text="2. Datos y Fusión Y")
        nb.add(self.tab_matrix, text="3. Matriz IM")
        nb.add(self.tab_kruskal, text="4. Kruskal (paso a paso)")
        nb.add(self.tab_prim, text="5. Prim (paso a paso)")
        nb.add(self.tab_compare, text="6. Best vs Worst (ΣA-ΣB)")
        nb.add(self.tab_pizarron, text="7. vs Pizarrón (X1-X5)")
        nb.add(self.tab_conclusion, text="8. Conclusión rVP")

        self.refresh_all()

    # ------------------------------------------------------------ utilidades
    def _ask_paths_if_missing(self):
        if not os.path.exists(self.best_path):
            messagebox.showinfo("Archivo Best no encontrado",
                                 "Seleccione el archivo d9_strong_b.csv (subconjunto Best, Y=1).")
            p = filedialog.askopenfilename(title="Seleccionar d9_strong_b.csv",
                                            filetypes=[("CSV", "*.csv")])
            if not p:
                return False
            self.best_path = p
        if not os.path.exists(self.worst_path):
            messagebox.showinfo("Archivo Worst no encontrado",
                                 "Seleccione el archivo d9_strong_w.csv (subconjunto Worst, Y=0).")
            p = filedialog.askopenfilename(title="Seleccionar d9_strong_w.csv",
                                            filetypes=[("CSV", "*.csv")])
            if not p:
                return False
            self.worst_path = p
        return True

    def _load_data(self):
        try:
            self.df_best = load_csv_safe(self.best_path)
            self.df_worst = load_csv_safe(self.worst_path)
        except Exception as exc:
            messagebox.showerror("Error al cargar CSV", str(exc))
            raise
        self.M_best = compute_mi_matrix(self.df_best)
        self.M_worst = compute_mi_matrix(self.df_worst)

    def recompute_all(self):
        try:
            self._load_data()
        except Exception:
            return
        self.refresh_all()

    def refresh_all(self):
        self.tab_datos.refresh()
        self.tab_matrix.refresh()
        self.tab_kruskal.reload_steps()
        self.tab_prim.reload_steps()
        self.tab_compare.refresh()
        self.tab_pizarron.refresh()
        self.tab_conclusion.refresh()


# ==============================================================================
#  PUNTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()