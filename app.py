# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
import skfuzzy as fuzz

st.set_page_config(page_title="Iris: K-means vs Fuzzy C-means", layout="wide")

# Paletas de cores fixas (GT diferente de clusters)
GT_COLORS = ['#9467bd', '#d62728', '#8c564b']         # roxo, vermelho, marrom
CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']    # azul, laranja, verde

# =========================
# Helpers
# =========================
def _palette(n=3):
    return [cm.tab10(i) for i in range(n)]

def plot_ground_truth(X2, y, names):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    colors = _palette(3)
    for c in range(3):
        pts = X2[y == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.9, label=names[c], c=[colors[c]])
    ax.set_title("Ground truth", fontsize=14, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)
    return fig

def plot_solution_scatter(X2, labels, centers2=None, title=""):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    uniq = np.unique(labels)
    for c in uniq:
        pts = X2[labels == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.9,
                   c=[CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)]],
                   label=f"c={int(c)}")
    if centers2 is not None:
        ax.scatter(centers2[:,0], centers2[:,1], s=120, marker="X",
                   edgecolor="k", linewidths=0.8)
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)
    return fig

def plot_silhouette(Xn, labels, title_prefix="Silhouette", bg=None, color_list=None):
    """
    bg: (vals, labels) para desenhar uma 'sombra' de referência
    """
    sil_vals = silhouette_samples(Xn, labels)
    sil_avg = silhouette_score(Xn, labels)

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    # fundo (sombra)
    if bg is not None:
        bg_vals, bg_labels = bg
        y_lower = 10
        for c in np.unique(bg_labels):
            vals_c = np.sort(bg_vals[bg_labels == c])
            size_c = len(vals_c)
            y_upper = y_lower + size_c
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                             alpha=0.12, color=cm.Greys(0.5))
            y_lower = y_upper + 10

    uniq = np.unique(labels)
    colors = (color_list if color_list is not None else [CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in uniq])
    y_lower = 10
    means_by_cluster = {}
    for idx, c in enumerate(uniq):
        vals_c = np.sort(sil_vals[labels == c])
        size_c = len(vals_c)
        if size_c == 0:
            continue
        y_upper = y_lower + size_c
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                         facecolor=colors[idx], edgecolor=colors[idx], alpha=0.9)
        means_by_cluster[int(c)] = vals_c.mean()
        ax.text(0.02, (y_lower + y_upper)/2, f"c={int(c)}", va="center", fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(sil_avg, linestyle="--", linewidth=1.3, color=cm.Purples(0.8))
    txt = f"{title_prefix}\nmean={sil_avg:.2f} | " + " | ".join(
        [f"c{c}={means_by_cluster[c]:.2f}" for c in sorted(means_by_cluster)]
    )
    ax.set_title(txt, fontsize=11, pad=8)
    ax.set_xlabel("Silhouette"); ax.set_yticks([])
    ax.set_xlim([-0.1, 1.0])
    return fig

def violin_grid(df_plot, label_col, feat_names, suptitle):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    for ax, var in zip(axes, feat_names):
        sns.violinplot(data=df_plot, x=label_col, y=var,
                       order=[0,1,2], inner="quartile", cut=0, ax=ax)
        ax.set_xlabel("cluster"); ax.set_title(var)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# -------------------------
# Helper: tabela descritiva no formato solicitado
# linhas = estatísticas (n, média, desvio padrão, min, mediana, max)
# colunas hierárquicas = variável -> (c0, c1, c2)
# -------------------------
import pandas as _pd

def make_desc_table(df: _pd.DataFrame, cluster_col: str, feature_names: list, cluster_order=(0,1,2)) -> _pd.DataFrame:
    # calcula agregados por cluster
    agg = (
        df.groupby(cluster_col)[feature_names]
          .agg(['count','mean','std','min','median','max'])
          .sort_index()
    )
    # mapeia nomes das linhas (estatísticas)
    stats_map = [
        ('count', 'n'),
        ('mean', 'média'),
        ('std', 'desvio padrão'),
        ('min', 'min'),
        ('median', 'mediana'),
        ('max', 'max'),
    ]
    # monta tabela com colunas multiíndice: (variável, cX)
    pieces = []
    cols = []
    for var in feature_names:
        for c in cluster_order:
            series_vals = []
            for stat_key, stat_label in stats_map:
                # valor do agregado para (cluster=c, variável=var, estatística=stat_key)
                try:
                    val = agg.loc[c, (var, stat_key)]
                except KeyError:
                    val = float('nan')
                series_vals.append(val)
            s = _pd.Series(series_vals, index=[lbl for _, lbl in stats_map])
            pieces.append(s)
            cols.append((var, f"c{c}"))
    table = _pd.concat(pieces, axis=1)
    table.columns = _pd.MultiIndex.from_tuples(cols, names=['variável', 'cluster'])
    return table

# =========================
# Data & Models (cached)
# =========================
@st.cache_data
def load_prepare():
    iris = datasets.load_iris(as_frame=True)
    X = iris.data.values
    y = iris.target.values
    names = iris.target_names
    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xn)

    km = KMeans(n_clusters=3, random_state=42)
    labels_km = km.fit_predict(Xn)
    centers2_km = pca.transform(km.cluster_centers_)

    # FCM (skfuzzy usa (features, samples))
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        Xn.T, c=3, m=2.0, error=1e-6, maxiter = 1000, seed=42)
    labels_fcm_argmax = np.argmax(u, axis=0)
    centers2_fcm = pca.transform(cntr)

    feat_names = iris.feature_names
    df_all = pd.DataFrame(X, columns=feat_names)
    df_all["km_cluster"] = labels_km
    df_all["fcm_cluster"] = labels_fcm_argmax
    df_all["fcm_max_u"] = u.max(axis=0)

    return (iris, X, Xn, X2, y, names,
            labels_km, centers2_km,
            u, labels_fcm_argmax, centers2_fcm,
            feat_names, df_all)

(iris, X, Xn, X2, y, names,
 labels_km, centers2_km,
 u, labels_fcm_argmax, centers2_fcm,
 feat_names, df_all) = load_prepare()

# Silhueta "dura" de referência para FCM (threshold=0)
sil_hard_vals = silhouette_samples(Xn, labels_fcm_argmax)
bg_ref = (sil_hard_vals, labels_fcm_argmax)

# =========================
# Sidebar (sempre visível)
# =========================
with st.sidebar:
    st.header("Parâmetros")
    thr = st.slider("FCM: threshold de pertinência dominante", 0.0, 1.0, 0.0, 0.01)
    st.caption("Pontos com max(pertinência) < threshold são ignorados nos gráficos/tabelas filtrados.")

st.title("Iris • K-means vs Fuzzy C-means")
st.caption("Normalização: MinMax • PCA 2D para visualização • k=c=3")

# =========================
# Tabs principais
# =========================
tab_graficos, tab_tabelas = st.tabs(["Gráficos", "Análise Descritiva por Cluster"])

# -------------------------
# TAB 1 — GRÁFICOS
# -------------------------
with tab_graficos:

    st.subheader("Painel geral comparativo")
    keep = (u.max(axis=0) >= thr)
    labels_thr = labels_fcm_argmax[keep] if keep.sum() > 0 else labels_fcm_argmax
    Xn_thr = Xn[keep] if keep.sum() > 0 else Xn

    fig_all, axs = plt.subplots(2, 3, figsize=(16, 8))

    # (0,0) Ground truth com cores GT_COLORS
    ax = axs[0,0]
    for idx, c in enumerate([0,1,2]):
        pts = X2[y == c]
        ax.scatter(pts[:,0], pts[:,1], s=18, alpha=0.9, c=[GT_COLORS[idx]], label=str(datasets.load_iris().target_names[c]))
    ax.set_title("Ground truth", fontsize=14, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)

    # (0,1) Scatter K-means com CLUSTER_COLORS + legenda
    ax = axs[0,1]
    for idx, c in enumerate(np.unique(labels_km)):
        pts = X2[labels_km == c]
        ax.scatter(pts[:,0], pts[:,1], s=18, alpha=0.9, c=[CLUSTER_COLORS[idx]], label=f"c={int(c)}")
    ax.scatter(centers2_km[:,0], centers2_km[:,1], s=120, marker='X', edgecolor='k', linewidths=0.8)
    ax.set_title("k-means (k=3)", fontsize=14, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)

    # (0,2) Silhouette K-means com as mesmas cores
    ax = axs[0,2]
    sil_vals_km = silhouette_samples(Xn, labels_km)
    sil_avg_km = silhouette_score(Xn, labels_km)
    y_lower = 10
    for idx, c in enumerate(np.unique(labels_km)):
        vals_c = np.sort(sil_vals_km[labels_km == c])
        y_upper = y_lower + len(vals_c)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                         facecolor=CLUSTER_COLORS[idx], edgecolor=CLUSTER_COLORS[idx], alpha=0.9)
        ax.text(0.02, (y_lower + y_upper)/2, f"c={int(c)}", va='center', fontsize=9)
        y_lower = y_upper + 10
    ax.axvline(sil_avg_km, linestyle='--', linewidth=1.3, color=cm.Purples(0.8))
    ax.set_title(f"Silhouette (k-means)\nmean={sil_avg_km:.2f}", fontsize=11, pad=8)
    ax.set_xlabel("Silhouette"); ax.set_yticks([]); ax.set_xlim([-0.1,1.0])

    # (1,0) vazio
    axs[1,0].axis('off')

    # (1,1) Scatter FCM (filtrado) com CLUSTER_COLORS + legenda
    ax = axs[1,1]
    # pontos ignorados em cinza
    ax.scatter(X2[~keep,0], X2[~keep,1], s=12, alpha=0.25, c=[cm.Greys(0.6)])
    for idx, c in enumerate(np.unique(labels_fcm_argmax)):
        m = keep & (labels_fcm_argmax == c)
        ax.scatter(X2[m,0], X2[m,1], s=18, alpha=0.95, c=[CLUSTER_COLORS[idx]], label=f"c={int(c)}")
    ax.scatter(centers2_fcm[:,0], centers2_fcm[:,1], s=120, marker='X', edgecolor='k', linewidths=0.8)
    ax.set_title(f"c-means (c=3) — threshold={thr:.2f}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)

    # (1,2) Silhouette FCM (filtrado) com sombra do threshold=0 e as mesmas cores
    ax = axs[1,2]
    if keep.sum() >= 3 and len(np.unique(labels_thr)) >= 2:
        # sombra da silhueta dura no subconjunto mantido
        bg_vals, bg_labels = bg_ref
        y_lower = 10
        # desenha sombra
        for c in np.unique(bg_labels[keep]):
            vals_c = np.sort(bg_vals[keep][bg_labels[keep] == c])
            y_upper = y_lower + len(vals_c)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c, alpha=0.12, color=cm.Greys(0.5))
            y_lower = y_upper + 10
        # barras coloridas
        sil_vals = silhouette_samples(Xn_thr, labels_thr)
        sil_avg = silhouette_score(Xn_thr, labels_thr)
        y_lower = 10
        uniq = np.unique(labels_thr)
        for idx, c in enumerate(uniq):
            vals_c = np.sort(sil_vals[labels_thr == c])
            y_upper = y_lower + len(vals_c)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                             facecolor=CLUSTER_COLORS[idx], edgecolor=CLUSTER_COLORS[idx], alpha=0.9)
            ax.text(0.02, (y_lower + y_upper)/2, f"c={int(c)}", va='center', fontsize=9)
            y_lower = y_upper + 10
        ax.axvline(sil_avg, linestyle='--', linewidth=1.3, color=cm.Purples(0.8))
        ax.set_title(f"Silhouette (c-means)\nmean={sil_avg:.2f}", fontsize=11, pad=8)
        ax.set_xlabel("Silhouette"); ax.set_yticks([]); ax.set_xlim([-0.1,1.0])
    else:
        ax.text(0.5, 0.5, 'Threshold alto — insuficiente para silhueta', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])

    fig_all.tight_layout()
    st.pyplot(fig_all, clear_figure=True)


    # Linha 3 — Violin plots 2×2 (K-means e FCM filtrado)
    st.subheader("Distribuições por variável (violin plots)")
    c6, c7 = st.columns([1,1], vertical_alignment="top")
    with c6:
        st.markdown("**K-means (k=3)**")
        st.pyplot(
            violin_grid(df_all, "km_cluster", feat_names,
                        "Variáveis do Iris por cluster (K-means)"),
            clear_figure=True
        )
    with c7:
        st.markdown(f"**Fuzzy C-means (c=3)** — filtrado por threshold = `{thr:.2f}`")
        df_plot = df_all[df_all["fcm_max_u"] >= thr].copy()
        if df_plot["fcm_cluster"].nunique() >= 2 and len(df_plot) >= 10:
            st.pyplot(
                violin_grid(df_plot, "fcm_cluster", feat_names,
                            "Variáveis do Iris por cluster (FCM filtrado)"),
                clear_figure=True
            )
        else:
            st.info("Threshold alto: pontos/clusters insuficientes para exibir os violinos.")

# -------------------------
# TAB 2 — TABELAS
# -------------------------
with tab_tabelas:
    st.subheader("Análise descritiva por cluster")
    colA, colB = st.columns([1,1], vertical_alignment="top")
    numeric_cols = list(feat_names)

    # K-means
    with colA:
        st.markdown("**K-means (k=3)**")
        desc_km = make_desc_table(df_all, 'km_cluster', numeric_cols, cluster_order=(0,1,2))
        st.dataframe(desc_km, use_container_width=True)

    # FCM (filtrado)
    with colB:
        st.markdown(f"**Fuzzy C-means (c=3)** — threshold atual: `{thr:.2f}`")
        df_fcm = df_all[df_all["fcm_max_u"] >= thr].copy()
        if df_fcm["fcm_cluster"].nunique() >= 1 and len(df_fcm) >= 3:
            desc_fcm = make_desc_table(df_fcm, 'fcm_cluster', numeric_cols, cluster_order=(0,1,2))
            st.dataframe(desc_fcm, use_container_width=True)
            st.caption("Tabela focada nas variáveis do Iris (sem incluir a pertinência).")
        else:
            st.info("Threshold alto: pontos/clusters insuficientes para a análise descritiva.")
