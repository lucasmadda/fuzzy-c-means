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
    colors = _palette(len(np.unique(labels)))
    for c in np.unique(labels):
        pts = X2[labels == c]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.9, c=[colors[c]])
    if centers2 is not None:
        ax.scatter(centers2[:,0], centers2[:,1], s=120, marker="X",
                   edgecolor="k", linewidths=0.8)
    ax.set_title(title, fontsize=14, pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    return fig

def plot_silhouette(Xn, labels, title_prefix="Silhouette", bg=None):
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

    colors = _palette(len(np.unique(labels)))
    y_lower = 10
    means_by_cluster = {}
    for c in np.unique(labels):
        vals_c = np.sort(sil_vals[labels == c])
        size_c = len(vals_c)
        if size_c == 0:
            continue
        y_upper = y_lower + size_c
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals_c,
                         facecolor=colors[c], edgecolor=colors[c], alpha=0.9)
        means_by_cluster[c] = vals_c.mean()
        ax.text(0.02, (y_lower + y_upper)/2, f"c={c}", va="center", fontsize=9)
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
    pca = PCA(n_components=2, random_state=5521)
    X2 = pca.fit_transform(Xn)

    km = KMeans(n_clusters=3, n_init='auto', random_state=5521)
    labels_km = km.fit_predict(Xn)
    centers2_km = pca.transform(km.cluster_centers_)

    # FCM (skfuzzy usa (features, samples))
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        Xn.T, c=3, m=2.0, error=1e-6, maxiter=500, init=None, seed=5521
    )
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
# UI
# =========================
st.title("Iris • K-means vs Fuzzy C-means")
st.caption("Normalização: MinMax • PCA 2D para visualização • k=c=3")

with st.sidebar:
    st.header("Parâmetros")
    thr = st.slider("FCM: threshold de pertinência dominante", 0.0, 1.0, 0.0, 0.01)
    st.write("Pontos com max(pertinência) < threshold são ignorados nos gráficos filtrados.")

# =========================
# Linha 1 — K-means: GT, scatter, silhouette
# =========================
st.subheader("K-means (k=3)")
col1, col2, col3 = st.columns(3)
with col1: st.pyplot(plot_ground_truth(X2, y, names), clear_figure=True)
with col2: st.pyplot(plot_solution_scatter(X2, labels_km, centers2_km, "k-means (k=3)"), clear_figure=True)
with col3: st.pyplot(plot_silhouette(Xn, labels_km, "Silhouette (k-means)"), clear_figure=True)

# =========================
# Linha 2 — FCM: scatter filtrado + silhouette com sombra de threshold=0
# =========================
st.subheader("Fuzzy C-means (c=3) com filtro por threshold")
# Filtragem
keep = (u.max(axis=0) >= thr)
labels_thr = labels_fcm_argmax[keep] if keep.sum() > 0 else labels_fcm_argmax
Xn_thr = Xn[keep] if keep.sum() > 0 else Xn

col4, col5 = st.columns(2)
with col4:
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = plt.gca()
    colors = _palette(3)
    # cinza para os ignorados
    ax.scatter(X2[~keep,0], X2[~keep,1], s=12, alpha=0.25, c=[cm.Greys(0.6)])
    # mantidos por cluster
    for c in np.unique(labels_fcm_argmax):
        m = keep & (labels_fcm_argmax == c)
        ax.scatter(X2[m,0], X2[m,1], s=18, alpha=0.95, c=[colors[c]], label=f"c={c}")
    ax.scatter(centers2_fcm[:,0], centers2_fcm[:,1], s=120, marker="X",
               edgecolor="k", linewidths=0.8)
    ax.set_title(f"c-means (c=3) — threshold={thr:.2f}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9, frameon=False)
    st.pyplot(fig, clear_figure=True)

with col5:
    if keep.sum() >= 3 and len(np.unique(labels_thr)) >= 2:
        st.pyplot(
            plot_silhouette(Xn_thr, labels_thr, "Silhouette (c-means)",
                            bg=(bg_ref[0][keep], bg_ref[1][keep])),
            clear_figure=True
        )
    else:
        st.info("Threshold alto: pontos/clusters insuficientes para calcular silhueta.")

# =========================
# Linha 3 — Violin plots 2×2 (K-means e FCM filtrado)
# =========================
st.subheader("Distribuições por variável (violin plots)")

col6, col7 = st.columns(2, vertical_alignment="top")

with col6:
    st.markdown("**K-means (k=3)**")
    st.pyplot(
        violin_grid(df_all, "km_cluster", feat_names,
                    "Variáveis do Iris por cluster (K-means)"),
        clear_figure=True
    )

with col7:
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

st.caption("Silhueta: média geral e média por cluster exibidas no título. "
           "No gráfico do FCM, o fundo cinza representa a silhueta 'dura' (threshold=0).")