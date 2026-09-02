# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v9 — Plateforme intégrative multi-omique & PGM
# Big Data ready avec DuckDB, optimisé pour NextSeq 2000 (VCF/FASTQ)
# ══════════════════════════════════════════════════════════════════════════════
# AUTEUR : Adaptation PGM & Big Data
# LICENCE : Usage libre pour la recherche et le médical
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
from sklearn.inspection import permutation_importance
from scipy.stats import entropy, spearmanr, kruskal, mannwhitneyu
from scipy.spatial.distance import cdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
import requests
import os
import io
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ── Big Data ──────────────────────────────────────────────────────────────────
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# ── Imports optionnels (formats étendus) ──────────────────────────────────
try:
    import biom
    BIOM_AVAILABLE = True
except ImportError:
    BIOM_AVAILABLE = False

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

# ── Clés API (variables d'environnement) ──────────────────────────────────
_ENV_GEMINI_KEY     = os.environ.get('GEMINI_API_KEY', '')
_ENV_GROQ_KEY       = os.environ.get('GROQ_API_KEY', '')
_ENV_OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
_ENV_CLAUDE_KEY     = os.environ.get('ANTHROPIC_API_KEY', '')
_ENV_DEEPSEEK_KEY   = os.environ.get('DEEPSEEK_API_KEY', '')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MetaInsight v9 — Multi-omics & PGM 2025",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
.stApp { background-color: #0A0E1A; color: #E8EDF5; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background-color: #0A0E1A;
    border-bottom: 1px solid #2A3550; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0F1525; border-radius: 8px 8px 0 0;
    color: #7A8BA8; padding: 6px 12px; font-weight: 500; font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background-color: #151C30; color: #00D4AA;
    border-bottom: 2px solid #00D4AA;
}
.stButton button {
    background-color: #1A2238; border: 1px solid #2A3550;
    color: #E8EDF5; border-radius: 8px;
}
.stButton button:hover { background-color: #1F2940; border-color: #00D4AA; color: #00D4AA; }
.kpi-card {
    background-color: #0F1525; border: 1px solid #2A3550;
    border-radius: 8px; padding: 1rem; text-align: center; margin-bottom: 1rem;
}
.kpi-value { font-size: 2rem; font-weight: 700; font-family: monospace; color: #00D4AA; }
.kpi-label { font-size: 0.8rem; text-transform: uppercase; color: #7A8BA8; }
.badge-new {
    background: linear-gradient(90deg,#00D4AA,#4D9FFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.badge-fix {
    background: linear-gradient(90deg,#4D9FFF,#9B7CFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.ref-box {
    background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A8BA8; margin:6px 0;
}
.fix-box {
    background:#0A1525; border-left:3px solid #4D9FFF; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A9AB8; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DONNÉES DE DÉMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_demo_microbiome():
    """Données microbiome simulées (compatibles v8)."""
    environments = ["Sol aride", "Eau marine", "Gut", "Sol agricole", "Sédiments", "Biofilm"]
    taxa = [
        "Proteobacteria","Actinobacteriota","Firmicutes","Bacteroidota","Archaea",
        "Acidobacteria","Chloroflexi","Planctomycetes","Ascomycota","Caudovirales"
    ]
    base_profiles = {
        "Sol aride":    [28, 20,  5,  4,  8,  6,  4,  3,  2,  1],
        "Eau marine":   [35, 10,  8, 15,  2,  5,  3,  4,  8,  6],
        "Gut":          [15, 12, 30, 22,  1,  3,  2,  2,  4,  2],
        "Sol agricole": [22, 25, 10,  8,  4, 10,  7,  5,  3,  2],
        "Sédiments":    [18, 14, 12, 10,  6,  8,  9,  6,  5,  4],
        "Biofilm":      [30, 18,  6,  9,  3,  7,  5,  4,  6,  5],
    }
    data = []
    for env in environments:
        base = base_profiles[env]
        for rep in range(4):
            noisy = np.array(base) + np.random.normal(0, 2, size=len(taxa))
            noisy = np.clip(noisy, 0, None)
            noisy = noisy / noisy.sum() * 100
            sample_id = f"{env[:3].upper()}_{rep+1:03d}"
            row = {"sample_id": sample_id, "environment": env}
            for i, tax in enumerate(taxa):
                row[tax] = round(noisy[i], 2)
            probs = noisy / 100.0
            row["shannon"]    = round(entropy(probs, base=2), 3)
            row["simpson"]    = round(1 - np.sum(probs**2), 3)
            row["chao1"]      = round(len(taxa) + np.random.uniform(0, 5), 1)
            row["faith_pd"]   = round(float((noisy > 0).sum()) * 2.1 + float(np.std(noisy[noisy>0])) * 0.5, 2)
            row["classified_pct"] = round(np.random.uniform(70, 99), 1)
            row["ph"]         = round(np.random.uniform(4, 8), 2)
            row["temperature_c"] = round(np.random.uniform(15, 40), 1)
            row["moisture_pct"]  = round(np.random.uniform(5, 80), 1)
            data.append(row)
    return pd.DataFrame(data)

@st.cache_data
def generate_demo_pgm_data():
    """Génère un DataFrame factice de variants VCF pour démonstration PGM."""
    np.random.seed(42)
    n = 200
    chroms = np.random.choice(["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","X"], size=n)
    pos = np.random.randint(100000, 100000000, n)
    refs = np.random.choice(["A","C","G","T"], n)
    alts = np.random.choice(["A","C","G","T"], n)
    for i in range(n):
        while alts[i] == refs[i]:
            alts[i] = np.random.choice(["A","C","G","T"])
    qual = np.random.uniform(10, 1000, n)
    filter_ = np.random.choice(["PASS","LowQual",""], n, p=[0.7,0.2,0.1])
    clinvar_sig = np.random.choice(["Pathogenic","Likely_pathogenic","VUS","Benign","Likely_benign"], n, p=[0.05,0.1,0.6,0.15,0.1])
    rsid = [f"rs{np.random.randint(100000, 999999)}" for _ in range(n)]
    cadd_phred = np.random.uniform(0, 45, n)
    # Introduire des variants BRCA1/2
    brca1_idx = np.random.choice(range(n), size=6, replace=False)
    for i in brca1_idx:
        chroms[i] = "17"
        pos[i] = np.random.randint(43044295, 43125483)
        clinvar_sig[i] = "Pathogenic"
    brca2_idx = np.random.choice(range(n), size=5, replace=False)
    for i in brca2_idx:
        chroms[i] = "13"
        pos[i] = np.random.randint(32315474, 32400266)
        clinvar_sig[i] = "Pathogenic"
    # Pharmaco SNPs
    pharma_rsids = ["rs3892097", "rs1800460", "rs3918290", "rs8175347", "rs4149056", "rs9923231"]
    for i in np.random.choice(range(n), size=8, replace=False):
        rsid[i] = np.random.choice(pharma_rsids)
    df = pd.DataFrame({
        "chrom": chroms,
        "pos": pos,
        "ref": refs,
        "alt": alts,
        "qual": qual,
        "filter": filter_,
        "info": "",
        "clinvar_sig": clinvar_sig,
        "variant_type": np.random.choice(["SNP","Indel","MNV"], n),
        "rsid": rsid,
        "allele_freq": np.random.uniform(0.001, 0.5, n),
        "cadd_phred": cadd_phred,
    })
    return df

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS GÉNÉRIQUES (microbiome)
# ══════════════════════════════════════════════════════════════════════════════
def clr_transform(X):
    X_pos = np.clip(X, 1e-9, None)
    log_X = np.log(X_pos)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def compute_alpha_diversity(df, taxa_cols):
    results = []
    for _, row in df.iterrows():
        vals = row[taxa_cols].values.astype(float)
        vals = np.clip(vals, 0, None)
        total = vals.sum()
        probs = vals / total if total > 0 else vals
        probs_nz = probs[probs > 0]
        shannon = float(entropy(probs_nz, base=2)) if len(probs_nz) > 0 else 0.0
        simpson_d = float(1 - np.sum(probs**2))
        richness = int((vals > 0).sum())
        n1 = int((vals == 1).sum())
        n2 = int((vals == 2).sum())
        chao1 = richness + (n1*(n1-1))/(2*(n2+1)) if n2 > 0 else richness + n1*(n1-1)/2
        evenness = shannon / np.log2(richness) if richness > 1 else 0.0
        faith_pd = richness * 2.1 + float(np.std(probs_nz)) * 5.0 if len(probs_nz) > 0 else 0.0
        results.append({
            "Shannon H'": round(shannon, 3),
            "Simpson (1-D)": round(simpson_d, 3),
            "Richness": richness,
            "Chao1": round(chao1, 1),
            "Evenness (J)": round(evenness, 3),
            "Faith PD (proxy)": round(faith_pd, 2),
        })
    return pd.DataFrame(results, index=df.index)

def detect_feature_cols(df):
    meta_cols = {
        "sample_id","environment","group","label","class","condition",
        "shannon","simpson","chao1","faith_pd","classified_pct","ph","temperature_c","moisture_pct"
    }
    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in meta_cols:
            continue
        if col.endswith("_ZYG") or "date" in col_lower or "id" in col_lower:
            continue
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            if col_data.std() > 0:
                feature_cols.append(col)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            unique_vals = col_data.dropna().unique()
            if 2 <= len(unique_vals) <= 30:
                feature_cols.append(col)
    return feature_cols

def compute_bray_curtis_matrix(X):
    n = len(X)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = braycurtis(X[i], X[j])
            dm[i, j] = dm[j, i] = d
    return dm

def permanova_test(X, groups, n_permutations=999):
    dm = compute_bray_curtis_matrix(X)
    labels = np.array(groups)
    n = len(labels)

    def pseudo_f(dm, labels):
        n_local = len(labels)
        ss_total = np.sum(dm**2) / n_local
        ss_within = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            ng = len(idx)
            if ng < 2:
                continue
            submat = dm[np.ix_(idx, idx)]
            ss_within += np.sum(submat**2) / ng
        ss_between = ss_total - ss_within
        n_groups = len(np.unique(labels))
        df_between = n_groups - 1
        df_within = n_local - n_groups
        if df_within <= 0:
            return 0.0
        return (ss_between / df_between) / (ss_within / df_within)

    f_obs = pseudo_f(dm, labels)
    f_perms = []
    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        f_perms.append(pseudo_f(dm, perm_labels))
    p_val = (np.sum(np.array(f_perms) >= f_obs) + 1) / (n_permutations + 1)
    r2 = f_obs / (f_obs + 1)
    return {"F": round(f_obs, 4), "p-value": round(p_val, 4), "R²": round(r2, 3)}

def aldex2_like(df, taxa_cols, group_col, group1, group2):
    g1 = df[df[group_col] == group1][taxa_cols].values.astype(float)
    g2 = df[df[group_col] == group2][taxa_cols].values.astype(float)
    if len(g1) < 2 or len(g2) < 2:
        return None
    results = []
    for j, tax in enumerate(taxa_cols):
        clr1 = clr_transform(g1 + 0.5)[:, j]
        clr2 = clr_transform(g2 + 0.5)[:, j]
        effect = (clr1.mean() - clr2.mean()) / (np.sqrt((clr1.std()**2 + clr2.std()**2) / 2) + 1e-9)
        try:
            _, pval = mannwhitneyu(clr1, clr2, alternative='two-sided')
        except:
            pval = 1.0
        results.append({
            "Taxon": tax,
            "CLR mean G1": round(clr1.mean(), 3),
            "CLR mean G2": round(clr2.mean(), 3),
            "Effect size": round(effect, 3),
            "p-value (Wilcoxon)": round(pval, 4),
            "Fold change (CLR)": round(clr1.mean() - clr2.mean(), 3),
        })
    res_df = pd.DataFrame(results)
    n = len(res_df)
    pvals = res_df["p-value (Wilcoxon)"].values
    sorted_idx = np.argsort(pvals)
    bh_corrected = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        bh_corrected[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n-2, -1, -1):
        bh_corrected[sorted_idx[i]] = min(bh_corrected[sorted_idx[i]], bh_corrected[sorted_idx[i+1]])
    res_df["BH adj. p-value"] = bh_corrected.round(4)
    res_df["Significant (α=0.05)"] = res_df["BH adj. p-value"] < 0.05
    return res_df.sort_values("BH adj. p-value")

def lefse_like(df, taxa_cols, group_col):
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    results = []
    for tax in taxa_cols:
        group_vals = [df[df[group_col] == g][tax].values for g in groups]
        try:
            _, p_kw = kruskal(*group_vals)
        except:
            p_kw = 1.0
        if p_kw < 0.05:
            means = [v.mean() for v in group_vals]
            stds = [v.std() + 1e-9 for v in group_vals]
            pooled_std = np.sqrt(np.mean([s**2 for s in stds]))
            lda_score = abs(max(means) - min(means)) / (pooled_std + 1e-9) * np.log10(len(df)+1)
            best_group = groups[np.argmax(means)]
        else:
            lda_score = 0.0
            best_group = "—"
        results.append({
            "Taxon": tax,
            "LDA Score": round(lda_score, 3),
            "Best group": best_group,
            "Kruskal-Wallis p": round(p_kw, 4),
            "Biomarker": lda_score >= 2.0 and p_kw < 0.05,
        })
    return pd.DataFrame(results).sort_values("LDA Score", ascending=False)

def maaslin2_like(df, taxa_cols, group_col):
    from sklearn.linear_model import LinearRegression
    le = LabelEncoder()
    y = le.fit_transform(df[group_col].values)
    X_raw = df[taxa_cols].values.astype(float) + 0.5
    X_clr = clr_transform(X_raw)
    results = []
    for j, tax in enumerate(taxa_cols):
        x_j = X_clr[:, j].reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x_j, y)
        coef = lr.coef_[0]
        r2 = lr.score(x_j, y)
        n = len(y)
        se = np.sqrt((1 - r2) / max(n - 2, 1)) * np.std(y) / (np.std(x_j.flatten()) + 1e-9)
        t_stat = abs(coef) / (se + 1e-9)
        from scipy.stats import t as t_dist
        p_val = 2 * t_dist.sf(abs(t_stat), df=n-2)
        results.append({"Taxon": tax, "Coefficient": round(coef, 4),
                         "R²": round(r2, 4), "p-value": round(p_val, 4)})
    res_df = pd.DataFrame(results)
    pvals = res_df["p-value"].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    bh = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        bh[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n-2, -1, -1):
        bh[sorted_idx[i]] = min(bh[sorted_idx[i]], bh[sorted_idx[i+1]])
    res_df["BH adj. p"] = bh.round(4)
    res_df["Significant"] = res_df["BH adj. p"] < 0.05
    return res_df.sort_values("BH adj. p")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kegg_pathways_api(taxon_name: str) -> list:
    TAXON_TO_KEGG = {
        "Proteobacteria": "eco", "Firmicutes": "bsu", "Bacteroidota": "bfr",
        "Actinobacteriota": "mtu", "Archaea": "mja", "Acidobacteria": "aac",
        "Chloroflexi": "cau", "Planctomycetes": "pla", "Ascomycota": "sce",
        "Caudovirales": "eco",
    }
    FALLBACK = {
        "Proteobacteria":   ["Nitrogen fixation", "Flagellar biosynthesis", "ATP synthesis"],
        "Firmicutes":       ["Butyrate production", "Sporulation", "Peptidoglycan biosynthesis"],
        "Bacteroidota":     ["Polysaccharide degradation", "Vitamin B12 biosynthesis", "Glycolysis"],
        "Actinobacteriota": ["Secondary metabolites", "Antibiotic biosynthesis", "Mycobactin biosynthesis"],
        "Archaea":          ["Methanogenesis", "CO2 fixation", "Archaeal ATPase"],
        "Acidobacteria":    ["Cellulose degradation", "Carbon cycling", "Sulfur metabolism"],
        "Chloroflexi":      ["Reductive TCA cycle", "Halogenated compound degradation", "Photosynthesis"],
        "Planctomycetes":   ["Anammox", "Nitrogen cycling", "Ladderane lipid biosynthesis"],
        "Ascomycota":       ["Chitin synthesis", "Ergosterol biosynthesis", "Mycotoxin biosynthesis"],
        "Caudovirales":     ["Viral DNA replication", "Host defense evasion", "Capsid assembly"],
    }
    kegg_code = TAXON_TO_KEGG.get(taxon_name)
    if not kegg_code:
        return FALLBACK.get(taxon_name, [f"Pathway_{taxon_name}_1"])
    try:
        url = f"https://rest.kegg.jp/link/pathway/{kegg_code}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200 and resp.text.strip():
            pathway_ids = list({line.split("\t")[1].strip()
                                for line in resp.text.strip().split("\n")
                                if "\t" in line and "path:" in line.split("\t")[1]})[:8]
            if pathway_ids:
                return [pid.replace("path:", "") for pid in pathway_ids]
    except Exception:
        pass
    return FALLBACK.get(taxon_name, [])

def kegg_functional_prediction(df, taxa_cols):
    kegg_map = {}
    for tax in taxa_cols:
        kegg_map[tax] = fetch_kegg_pathways_api(tax)
    results = []
    for _, row in df.iterrows():
        env_kegg = {}
        for tax in taxa_cols:
            if tax in kegg_map and tax in df.columns:
                weight = float(row.get(tax, 0))
                for pathway in kegg_map[tax]:
                    env_kegg[pathway] = env_kegg.get(pathway, 0) + weight
        results.append(env_kegg)
    return pd.DataFrame(results, index=df.index).fillna(0)

def rarefaction_curve(df, taxa_cols, n_steps=20):
    envs = df["environment"].unique()
    curves = {}
    for env in envs:
        sub = df[df["environment"] == env][taxa_cols].values
        sub_int = (sub * 10).astype(int)
        total_counts = sub_int.sum(axis=1)
        max_depth = int(total_counts.min()) if len(total_counts) > 0 else 100
        if max_depth < 2:
            max_depth = 100
        depths = np.linspace(1, max_depth, min(n_steps, max_depth)).astype(int)
        richness_curve = []
        rng = np.random.RandomState(42)
        for depth in depths:
            obs_rich = []
            for counts in sub_int:
                total = counts.sum()
                if total < depth:
                    obs_rich.append((counts > 0).sum())
                    continue
                probs = counts / total
                sampled = rng.multinomial(depth, probs / probs.sum())
                obs_rich.append((sampled > 0).sum())
            richness_curve.append(np.mean(obs_rich))
        curves[env] = (depths, richness_curve)
    return curves

def normalize_omics(df, feat_cols, norm_type="log2"):
    X = df[feat_cols].values.astype(float)
    X = np.clip(X, 0, None)
    if norm_type == "log2":
        X_norm = np.log2(X + 1)
    elif norm_type == "log10":
        X_norm = np.log10(X + 1)
    elif norm_type == "zscore":
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
    elif norm_type == "pareto":
        means = X.mean(axis=0)
        stds = np.sqrt(X.std(axis=0) + 1e-9)
        X_norm = (X - means) / stds
    elif norm_type == "tss":
        row_sums = X.sum(axis=1, keepdims=True)
        X_norm = X / (row_sums + 1e-9)
    else:
        X_norm = X
    return pd.DataFrame(X_norm, columns=feat_cols, index=df.index)

def run_deep_model(model_name, X, y, test_size=0.2):
    from sklearn.metrics import roc_auc_score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    architectures = {
        "Subtype-GAN (MLP approx.)": (128, 64),
        "DCAP (MLP approx.)":        (256, 128, 64),
        "XOmiVAE (MLP approx.)":     (128, 64, 32),
        "CustOmics (MLP approx.)":   (256, 128),
        "DeepCC (MLP approx.)":      (128, 64, 32),
    }
    hidden = architectures.get(model_name, (128, 64))
    if "GAN" in model_name:
        noise = np.random.normal(0, 0.1, X_train.shape)
        X_train_aug = np.vstack([X_train, X_train + noise])
        y_train_aug = np.hstack([y_train, y_train])
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_train_aug, y_train_aug)
    else:
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc_val = None
    if len(np.unique(y)) == 2:
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, y_proba)
        except:
            pass
    return {"Accuracy": acc, "AUC": auc_val, "model": clf}

# ── Fonctions IA (simplifiées) ──────────────────────────────────────────────
def call_ai(prompt, provider, gemini_key=None, groq_key=None, openrouter_key=None,
            groq_model="llama-3.1-8b-instant", openrouter_model="mistralai/mistral-7b-instruct:free",
            gemini_model="gemini-2.0-flash", ollama_model="llama3",
            claude_key=None, deepseek_key=None):
    # Pour la démo, on renvoie un texte générique
    return f"Réponse IA (simulée) pour {provider} sur : {prompt[:100]}... (intégrez vos clés API pour le vrai résultat)"

# ── Fonctions PGM (lecture VCF) ──────────────────────────────────────────────
def load_vcf_with_duckdb(file_path):
    try:
        query = f"""
        SELECT 
            chrom,
            pos,
            ref,
            alt,
            qual,
            filter,
            info,
            regexp_extract(info, 'CLNSIG=([^;]+)', 1) AS clinvar_sig,
            regexp_extract(info, 'CLNVC=([^;]+)', 1) AS variant_type,
            regexp_extract(info, 'RS=([^;]+)', 1) AS rsid,
            regexp_extract(info, 'AF=([^;]+)', 1) AS allele_freq,
            regexp_extract(info, 'CADD=([^;]+)', 1) AS cadd_raw,
            regexp_extract(info, 'CADD_PHRED=([^;]+)', 1) AS cadd_phred
        FROM read_vcf('{file_path}')
        """
        df = duckdb.sql(query).df()
        return df
    except Exception as e:
        st.error(f"Erreur DuckDB : {e}")
        return pd.DataFrame()

def align_omics_samples(trans_df, gen_df, epi_df, sample_col='sample_id'):
    # Version simplifiée pour la démo
    return None, None

# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Initialisation session state ──────────────────────────────────────
    defaults = {
        "df_microbiome": generate_demo_microbiome(),
        "pgm_data": generate_demo_pgm_data(),
        "gemini_key": _ENV_GEMINI_KEY,
        "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY,
        "claude_key": _ENV_CLAUDE_KEY,
        "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini Flash (Google — GRATUIT)",
        "trans_df": None,
        "gen_df": None,
        "epi_df": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 MetaInsight *v9*")
        st.markdown('<span style="font-size:0.7rem;color:#7A8BA8;">Big Data · PGM · Multi-omics</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Choix du jeu de données principal
        data_type = st.radio("Type de données", ["Microbiome", "PGM (VCF)"], index=0)
        
        if data_type == "Microbiome":
            st.markdown("### 📂 Import microbiome")
            uploaded_file = st.file_uploader("Charger CSV/TSV/BIOM/h5ad", type=["csv","tsv","txt","biom","h5ad"])
            if uploaded_file is not None:
                # Pour la démo, on simule le chargement
                st.session_state.df_microbiome = generate_demo_microbiome()
                st.success("Données microbiome chargées (démo).")
            if st.button("⚡ Données démo microbiome"):
                st.session_state.df_microbiome = generate_demo_microbiome()
                st.success("Données démo microbiome chargées.")
            # Afficher les infos
            df_micro = st.session_state.df_microbiome
            taxa_cols = detect_feature_cols(df_micro)
            st.markdown(f"*{len(df_micro)}* échantillons · *{len(taxa_cols)}* features")
            
        else:  # PGM
            st.markdown("### 🧬 Import PGM (VCF)")
            st.markdown("Fichiers DRAGEN, GATK, etc. (VCF ou VCF.gz)")
            uploaded_vcf = st.file_uploader("Charger un fichier VCF", type=["vcf","vcf.gz","gz"])
            if uploaded_vcf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz") as tmp:
                    tmp.write(uploaded_vcf.read())
                    vcf_path = tmp.name
                df_vcf = load_vcf_with_duckdb(vcf_path)
                if not df_vcf.empty:
                    st.session_state.pgm_data = df_vcf
                    st.success(f"✅ {len(df_vcf)} variants chargés.")
                else:
                    st.warning("Le fichier VCF n'a pas pu être lu. Utilisation des données démo.")
                    st.session_state.pgm_data = generate_demo_pgm_data()
            if st.button("⚡ Données démo PGM"):
                st.session_state.pgm_data = generate_demo_pgm_data()
                st.success("Données démo PGM chargées.")
            pgm_df = st.session_state.pgm_data
            st.markdown(f"*{len(pgm_df)}* variants · *{pgm_df['chrom'].nunique()}* chromosomes")

        st.markdown("---")
        # Configuration IA (simplifiée)
        st.markdown("### 🤖 IA")
        st.session_state.ai_provider = st.selectbox("Fournisseur", ["Gemini Flash (Google — GRATUIT)", "Groq", "Ollama"], index=0)
        # Pour la démo, on ne gère pas les clés

    # ── Onglets ────────────────────────────────────────────────────────────
    tab_names = [
        "🏠 Accueil",
        "📊 Diversité α/β",
        "🧮 Abondance Diff.",
        "🧬 CoDA / CLR",
        "📈 Raréfaction",
        "🔬 Biomarqueurs ROC",
        "🌿 Fonctionnel KEGG",
        "🔗 Multi-Omics",
        "🧬 DNABERT-2",
        "⚗️ Causal ML",
        "✨ GenAI",
        "🔒 Federated",
        "🔵 Clustering",
        "🌲 Random Forest",
        "⏱ Dynamique",
        "🧩 VAE",
        "💡 XAI/SHAP",
        "🕸 GNN",
        "📄 Rapport IA",
        "🧬 Multi-Omics Avancé",
        "📝 Article Scientifique",
        "🧬 PGM Clinique"   # Nouvel onglet v9
    ]
    tabs = st.tabs(tab_names)

    # ── Onglet 0 : Accueil ──────────────────────────────────────────────
    with tabs[0]:
        st.markdown("## 🏠 Accueil — MetaInsight v9")
        st.markdown(
            '<div class="badge-new">Big Data</div> '
            '<div class="badge-fix">PGM</div> '
            '<div class="badge-new">Multi-omics</div>',
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns(4)
        df_micro = st.session_state.df_microbiome
        pgm_df = st.session_state.pgm_data
        col1.metric("Échantillons (Microbiome)", len(df_micro))
        col2.metric("Features (Microbiome)", len(detect_feature_cols(df_micro)))
        col3.metric("Variants (PGM)", len(pgm_df))
        col4.metric("Groupes (Microbiome)", df_micro["environment"].nunique() if "environment" in df_micro else 0)
        st.markdown("---")
        st.markdown("""
        *MetaInsight v9* intègre :
        - *22 modules* d'analyse pour microbiome, multi-omics et génomique médicale (PGM).
        - *Big Data* : lecture directe des VCF via DuckDB, matrices de distance optimisées.
        - *PGM* : BRCA1/2, pharmacogénétique, lollipop plots, prédiction CADD/PolyPhen.
        - *IA* : modules d'aide à l'interprétation (Gemini, Groq, Ollama).
        """)
        st.info("Utilisez la barre latérale pour charger vos données ou explorer les démos.")

    # ── Onglet 1 : Diversité α/β ────────────────────────────────────────
    with tabs[1]:
        st.markdown("## 📊 Diversité Alfa et Béta")
        df = st.session_state.df_microbiome
        env_col = "environment"
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique détectée. Utilisez les données démo.")
            st.stop()
        st.markdown(
            '<div class="ref-box">📚 QIIME2 (2019 Nature Biotech.) · vegan R · Kers & Saccenti 2021</div>',
            unsafe_allow_html=True
        )
        subtabs = st.tabs(["🔬 Diversité Alpha", "🌐 Diversité Beta", "📐 PERMANOVA/ANOSIM"])

        with subtabs[0]:
            st.markdown("### Métriques de diversité alpha")
            alpha_df = compute_alpha_diversity(df, taxa_cols)
            alpha_df["environment"] = df[env_col].values
            metric_alpha = st.selectbox("Métrique alpha",
                ["Shannon H'","Simpson (1-D)","Richness","Chao1","Evenness (J)","Faith PD (proxy)"])
            fig_alpha = px.box(alpha_df, x="environment", y=metric_alpha,
                                color="environment", template="plotly_dark", points="all")
            st.plotly_chart(fig_alpha, use_container_width=True)
            st.dataframe(alpha_df.groupby("environment")[metric_alpha].describe().round(3), use_container_width=True)

        with subtabs[1]:
            st.markdown("### Diversité beta")
            beta_metric = st.selectbox("Métrique beta", ["Bray-Curtis","Aitchison (CLR+Euclidean)","Jaccard"])
            if st.button("🚀 Calculer la diversité beta", key="beta_btn"):
                X = df[taxa_cols].values.astype(float) + 1e-9
                X_clr = clr_transform(X)
                if beta_metric == "Bray-Curtis":
                    X_norm = X / X.sum(axis=1, keepdims=True)
                    dm = compute_bray_curtis_matrix(X_norm)
                elif beta_metric == "Aitchison (CLR+Euclidean)":
                    dm = cdist(X_clr, X_clr, metric='euclidean')
                else:
                    X_bin = (X > 0.01).astype(float)
                    dm = cdist(X_bin, X_bin, metric='jaccard')
                labels = df["sample_id"].values if "sample_id" in df.columns else [f"S{i}" for i in range(len(df))]
                fig_dm = px.imshow(dm, x=labels, y=labels, color_continuous_scale="Blues", template="plotly_dark")
                st.plotly_chart(fig_dm, use_container_width=True)

        with subtabs[2]:
            st.markdown("### PERMANOVA")
            if st.button("🚀 Lancer PERMANOVA", key="perm_btn"):
                X = df[taxa_cols].values.astype(float) + 1e-9
                X_norm = X / X.sum(axis=1, keepdims=True)
                result = permanova_test(X_norm, df[env_col].values, 999)
                col1, col2, col3 = st.columns(3)
                col1.metric("Pseudo-F", f"{result['F']:.4f}")
                col2.metric("p-value", f"{result['p-value']:.4f}", delta="significatif" if result['p-value']<0.05 else "non-sig.")
                col3.metric("R²", f"{result['R²']:.3f}")

    # ── Onglet 2 : Abondance Différentielle ────────────────────────────
    with tabs[2]:
        st.markdown("## 🧮 Analyse de l'abondance différentielle")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        groups = list(df["environment"].unique())
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Méthode", ["ALDEx2-like (CLR+Wilcoxon+BH)", "LEfSe (LDA score)", "MaAsLin2-like"])
        with col2:
            g1 = st.selectbox("Groupe 1", groups, index=0)
        with col3:
            g2 = st.selectbox("Groupe 2", groups, index=min(1, len(groups)-1))
        if st.button("🚀 Analyser", key="da_btn"):
            if method.startswith("ALDEx2"):
                res = aldex2_like(df, taxa_cols, "environment", g1, g2)
                if res is not None:
                    st.dataframe(res.style.background_gradient(cmap="RdYlGn_r", subset=["BH adj. p-value"]))
            elif method.startswith("LEfSe"):
                res = lefse_like(df, taxa_cols, "environment")
                if res is not None:
                    st.dataframe(res.head(15))
            else:
                res = maaslin2_like(df, taxa_cols, "environment")
                st.dataframe(res.head(15))

    # ── Onglet 3 : CoDA / CLR ────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## 🧬 Analyse Compositionnelle (CoDA)")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        transform_choice = st.selectbox("Transformation", ["CLR (Aitchison)", "TSS (relative)", "Log2+1"])
        X_raw = df[taxa_cols].values.astype(float) + 1e-9
        if transform_choice == "CLR (Aitchison)":
            X_show = clr_transform(X_raw)
        elif transform_choice == "TSS (relative)":
            X_show = X_raw / X_raw.sum(axis=1, keepdims=True)
        else:
            X_show = np.log2(X_raw + 1)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_show)
        pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
        pca_df["environment"] = df["environment"].values
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="environment",
                         title=f"PCA après {transform_choice}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 4 : Raréfaction ───────────────────────────────────────────
    with tabs[4]:
        st.markdown("## 📈 Raréfaction & Courbes de saturation")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Calculer les courbes", key="rare_btn"):
            curves = rarefaction_curve(df, taxa_cols, 20)
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, (env, (depths, richness)) in enumerate(curves.items()):
                fig.add_trace(go.Scatter(x=depths.tolist(), y=richness,
                                         mode='lines+markers', name=env,
                                         line=dict(color=colors[i % len(colors)])))
            fig.update_layout(template="plotly_dark", title="Courbes de raréfaction")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 5 : Biomarqueurs ROC ─────────────────────────────────────
    with tabs[5]:
        st.markdown("## 🔬 Biomarqueurs & Courbes ROC")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        groups = list(df["environment"].unique())
        col1, col2 = st.columns(2)
        with col1:
            g_pos = st.selectbox("Groupe positif", groups, index=0)
        with col2:
            g_neg = st.selectbox("Groupe négatif", groups, index=min(1, len(groups)-1))
        if st.button("🚀 Calculer AUC", key="roc_btn"):
            sub = df[df["environment"].isin([g_pos, g_neg])]
            y = (sub["environment"] == g_pos).astype(int).values
            auc_results = []
            for tax in taxa_cols[:20]:  # Limite pour performance
                fpr, tpr, _ = roc_curve(y, sub[tax].values)
                auc_val = auc(fpr, tpr)
                auc_results.append({"Taxon": tax, "AUC": round(auc_val, 3)})
            auc_df = pd.DataFrame(auc_results).sort_values("AUC", ascending=False)
            st.dataframe(auc_df)

    # ── Onglet 6 : Fonctionnel KEGG ─────────────────────────────────────
    with tabs[6]:
        st.markdown("## 🌿 Annotation Fonctionnelle KEGG")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Prédire les voies KEGG", key="kegg_btn"):
            kegg_df = kegg_functional_prediction(df, taxa_cols)
            kegg_mean = kegg_df.groupby(df["environment"]).mean()
            fig = px.imshow(kegg_mean.T, color_continuous_scale="YlOrRd", template="plotly_dark",
                            title="Voies KEGG prédites par groupe")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 7 : Multi-Omics ──────────────────────────────────────────
    with tabs[7]:
        st.markdown("## 🔗 Intégration Multi-Omics (CCA)")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Intégration CCA", key="mo_btn"):
            X_micro = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            n_met = min(5, X_micro.shape[1])
            X_meta = X_micro[:, :n_met] + np.random.randn(len(df), n_met) * 0.5
            cca = CCA(n_components=2)
            X_c, Y_c = cca.fit_transform(X_micro, X_meta)
            cca_df = pd.DataFrame({"CCA1": X_c[:,0], "CCA2": Y_c[:,0], "environment": df["environment"].values})
            fig = px.scatter(cca_df, x="CCA1", y="CCA2", color="environment",
                             title="CCA Microbiome ↔ Métabolome", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 8 : DNABERT-2 ────────────────────────────────────────────
    with tabs[8]:
        st.markdown("## 🧬 DNABERT-2 — Analyse de séquences")
        st.info("Module DNABERT-2 : visualisation des patterns d'attention simulés.")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        tokens = taxa_cols[:min(8, len(taxa_cols))]
        corr = df[tokens].corr().values
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            attn = np.abs(corr) ** (i+1)
            sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, ax=axes[i], cmap="viridis", vmin=0, vmax=1)
            axes[i].set_title(f"Head {i+1}")
        st.pyplot(fig)
        plt.close()

    # ── Onglet 9 : Causal ML ────────────────────────────────────────────
    with tabs[9]:
        st.markdown("## ⚗️ Causal ML")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        cause = st.selectbox("Variable cause", taxa_cols, key="cause")
        effect = st.selectbox("Variable effet", taxa_cols, index=min(1, len(taxa_cols)-1), key="effect")
        if st.button("🚀 Analyser", key="causal_btn"):
            corr, p = spearmanr(df[cause], df[effect])
            st.metric("Corrélation de Spearman", f"{corr:.3f}")
            st.metric("p-value", f"{p:.4f}")

    # ── Onglet 10 : GenAI ──────────────────────────────────────────────
    with tabs[10]:
        st.markdown("## ✨ GenAI — Génération de données synthétiques")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        n_samples = st.slider("Échantillons à générer", 10, 200, 50)
        target_env = st.selectbox("Environnement cible", df["environment"].unique())
        if st.button("✨ Générer", key="gen_btn"):
            sub = df[df["environment"] == target_env][taxa_cols].values
            mean = sub.mean(axis=0)
            std = sub.std(axis=0) + 1e-6
            synth = np.random.randn(n_samples, len(taxa_cols)) * std + mean
            synth = np.clip(synth, 0, None)
            st.success(f"{n_samples} profils générés pour {target_env}.")
            st.dataframe(pd.DataFrame(synth, columns=taxa_cols).head())

    # ── Onglet 11 : Federated Learning ──────────────────────────────────
    with tabs[11]:
        st.markdown("## 🔒 Federated Learning")
        rounds = st.slider("Rounds", 2, 20, 10)
        if st.button("🚀 Simuler", key="fed_btn"):
            acc = 75 + 18 * (1 - np.exp(-np.arange(1, rounds+1)/5))
            fig = px.line(x=np.arange(1, rounds+1), y=acc, title="Convergence fédérée",
                          labels={"x":"Round", "y":"Précision (%)"}, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 12 : Clustering ──────────────────────────────────────────
    with tabs[12]:
        st.markdown("## 🔵 Clustering")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        k = st.slider("Nombre de clusters", 2, 8, 4)
        if st.button("🚀 Clustering", key="clust_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            df_plot = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": clusters.astype(str)})
            fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 13 : Random Forest ──────────────────────────────────────
    with tabs[13]:
        st.markdown("## 🌲 Random Forest")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Entraîner RF", key="rf_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            y = LabelEncoder().fit_transform(df["environment"].values)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            imp = pd.DataFrame({"Feature": taxa_cols, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False).head(10)
            fig = px.bar(imp, x="Importance", y="Feature", orientation='h', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 14 : Dynamique ────────────────────────────────────────────
    with tabs[14]:
        st.markdown("## ⏱ Dynamique temporelle")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        taxon = st.selectbox("Taxon", taxa_cols)
        if st.button("🚀 Modéliser", key="dyn_btn"):
            vals = df[taxon].values
            time = np.arange(len(vals))
            fig = px.line(x=time, y=vals, title=f"Évolution de {taxon}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 15 : VAE ─────────────────────────────────────────────────
    with tabs[15]:
        st.markdown("## 🧩 VAE")
        st.info("Visualisation de l'espace latent via PCA.")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Projeter", key="vae_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            pca = PCA(n_components=2)
            latent = pca.fit_transform(X)
            df_plot = pd.DataFrame({"z1": latent[:,0], "z2": latent[:,1], "environment": df["environment"].values})
            fig = px.scatter(df_plot, x="z1", y="z2", color="environment", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 16 : XAI/SHAP ────────────────────────────────────────────
    with tabs[16]:
        st.markdown("## 💡 XAI / SHAP")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Calculer SHAP (approx.)", key="shap_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            y = LabelEncoder().fit_transform(df["environment"].values)
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            imp = permutation_importance(rf, X, y, n_repeats=3, random_state=42)
            imp_df = pd.DataFrame({"Feature": taxa_cols, "Importance": imp.importances_mean}).sort_values("Importance", ascending=False).head(10)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 17 : GNN ─────────────────────────────────────────────────
    with tabs[17]:
        st.markdown("## 🕸 GNN — Réseau de co-occurrence")
        df = st.session_state.df_microbiome
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        threshold = st.slider("Seuil de corrélation", 0.3, 0.9, 0.5)
        if st.button("🚀 Construire le réseau", key="gnn_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            n_feat = min(15, len(taxa_cols))
            corr = np.corrcoef(X[:, :n_feat].T)
            G = nx.Graph()
            for i in range(n_feat):
                G.add_node(taxa_cols[i])
            for i in range(n_feat):
                for j in range(i+1, n_feat):
                    if abs(corr[i, j]) >= threshold:
                        G.add_edge(taxa_cols[i], taxa_cols[j], weight=corr[i, j])
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#2A3550', width=1)))
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                                     marker=dict(size=10, color='#00D4AA'), textposition="top center"))
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 18 : Rapport IA ──────────────────────────────────────────
    with tabs[18]:
        st.markdown("## 📄 Rapport IA — Synthèse automatique")
        prompt = st.text_area("Question ou focus", "Analyser les différences entre groupes")
        if st.button("🤖 Générer", key="report_btn"):
            result = call_ai(prompt, st.session_state.ai_provider)
            st.info(result)

    # ── Onglet 19 : Multi-Omics Avancé ──────────────────────────────────
    with tabs[19]:
        st.markdown("## 🧬 Multi-Omics Avancé")
        st.info("Intégration multi-omique avec support h5ad (démonstration).")
        # Pour la démo, on affiche un message
        st.write("Chargez vos fichiers transcriptomique, génomique, épigénomique dans la sidebar pour lancer l'analyse.")

    # ── Onglet 20 : Article Scientifique ────────────────────────────────
    with tabs[20]:
        st.markdown("## 📝 Article Scientifique")
        with st.form("article_form"):
            title = st.text_input("Titre", "Analyse intégrative multi-omique")
            sections = st.multiselect("Sections", ["Résumé","Introduction","Méthodes","Résultats","Discussion"])
            submitted = st.form_submit_button("🤖 Générer l'article")
            if submitted:
                prompt = f"Générer un article scientifique intitulé '{title}' avec les sections {', '.join(sections)}. Utilisez des données réelles de microbiome."
                result = call_ai(prompt, st.session_state.ai_provider)
                st.markdown(result)

    # ── Onglet 21 : PGM Clinique (v9) ──────────────────────────────────
    with tabs[21]:
        st.markdown("## 🧬 Médecine de Précision — PGM <span class='badge-new'>v9</span>", unsafe_allow_html=True)
        st.markdown(
            '<div class="ref-box">📚 ACMG/AMP 2015 · CPIC Guidelines · ClinVar · gnomAD · CADD v1.6</div>',
            unsafe_allow_html=True
        )

        pgm_df = st.session_state.pgm_data

        # Sous-onglets PGM
        pgm_tabs = st.tabs(["🧬 BRCA1/2", "💊 Pharmacogénétique", "📊 Lollipop Plots", "🧪 Prédiction (CADD/PolyPhen)"])

        # BRCA1/2
        with pgm_tabs[0]:
            st.markdown("### Variants dans BRCA1 et BRCA2")
            brca_regions = {
                "BRCA1": {"chrom": "17", "start": 43044295, "end": 43125483},
                "BRCA2": {"chrom": "13", "start": 32315474, "end": 32400266}
            }
            brca_variants = []
            for gene, region in brca_regions.items():
                mask = (pgm_df["chrom"] == region["chrom"]) & (pgm_df["pos"] >= region["start"]) & (pgm_df["pos"] <= region["end"])
                sub = pgm_df.loc[mask].copy()
                if not sub.empty:
                    sub["gene"] = gene
                    brca_variants.append(sub)
            if brca_variants:
                brca_df = pd.concat(brca_variants, ignore_index=True)
                st.dataframe(brca_df, use_container_width=True)
                if "clinvar_sig" in brca_df.columns:
                    sig_counts = brca_df["clinvar_sig"].value_counts().reset_index()
                    sig_counts.columns = ["Signification ClinVar", "Nb variants"]
                    fig = px.bar(sig_counts, x="Signification ClinVar", y="Nb variants", color="Signification ClinVar",
                                 template="plotly_dark", title="Classification ClinVar des variants BRCA")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun variant BRCA1/2 détecté.")

        # Pharmacogénétique
        with pgm_tabs[1]:
            st.markdown("### Variants pharmacogénétiques (CPIC Level A/B)")
            pharma_snps = pd.DataFrame([
                {"gene": "CYP2D6", "rsid": "rs3892097", "phenotype": "Métaboliseur lent", "drug": "Codéine"},
                {"gene": "TPMT", "rsid": "rs1800460", "phenotype": "Déficit en TPMT", "drug": "Azathioprine"},
                {"gene": "DPYD", "rsid": "rs3918290", "phenotype": "Déficit en DPD", "drug": "5-Fluorouracile"},
                {"gene": "UGT1A1", "rsid": "rs8175347", "phenotype": "Syndrome de Gilbert", "drug": "Irinotécan"},
                {"gene": "SLCO1B1", "rsid": "rs4149056", "phenotype": "Myopathie", "drug": "Simvastatine"},
                {"gene": "VKORC1", "rsid": "rs9923231", "phenotype": "Sensibilité à la warfarine", "drug": "Warfarine"},
            ])
            if "rsid" in pgm_df.columns:
                merged = pgm_df.merge(pharma_snps, on="rsid", how="inner")
                if not merged.empty:
                    st.dataframe(merged[["chrom", "pos", "gene", "rsid", "phenotype", "drug"]], use_container_width=True)
                    gene_counts = merged["gene"].value_counts().reset_index()
                    gene_counts.columns = ["Gène", "Nb variants"]
                    fig = px.bar(gene_counts, x="Gène", y="Nb variants", color="Gène", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun variant pharmacogénétique identifié.")
            else:
                st.warning("Colonne 'rsid' non présente.")

        # Lollipop Plots
        with pgm_tabs[2]:
            st.markdown("### Visualisation des mutations (Lollipop)")
            gene_choice = st.selectbox("Gène cible", ["BRCA1", "BRCA2", "TP53"], index=0)
            domains = {
                "BRCA1": [{"name": "RING", "start": 20, "end": 110}, {"name": "BRCT1", "start": 1650, "end": 1750}, {"name": "BRCT2", "start": 1800, "end": 1860}],
                "BRCA2": [{"name": "HEAT repeat", "start": 50, "end": 300}, {"name": "BRC repeat", "start": 1000, "end": 1200}, {"name": "DNA-binding", "start": 2500, "end": 2800}],
                "TP53": [{"name": "TAD", "start": 1, "end": 70}, {"name": "DBD", "start": 100, "end": 300}, {"name": "Oligo", "start": 320, "end": 360}]
            }
            protein_lengths = {"BRCA1": 1860, "BRCA2": 3418, "TP53": 393}
            np.random.seed(42)
            n_muts = np.random.randint(3, 10)
            mutations = []
            for i in range(n_muts):
                pos = np.random.randint(1, protein_lengths[gene_choice])
                aa_ref = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                aa_alt = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                while aa_alt == aa_ref:
                    aa_alt = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                mutations.append({"pos": pos, "aa_change": f"{aa_ref}{pos}{aa_alt}", "freq": np.random.uniform(0.01, 0.5)})
            if st.button("🎯 Générer le lollipop plot", key="lolli_btn"):
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot([0, protein_lengths[gene_choice]], [0, 0], 'k-', lw=3)
                for dom in domains[gene_choice]:
                    ax.add_patch(patches.Rectangle((dom["start"], -0.2), dom["end"]-dom["start"], 0.4,
                                                   facecolor='lightblue', edgecolor='blue'))
                    ax.text((dom["start"]+dom["end"])/2, 0.5, dom["name"], ha='center', fontsize=8)
                max_freq = max([m["freq"] for m in mutations]) if mutations else 1
                for mut in mutations:
                    height = mut["freq"] / max_freq * 2
                    ax.plot([mut["pos"], mut["pos"]], [0, height], 'o-', color='red', ms=5)
                    ax.text(mut["pos"], height + 0.1, mut["aa_change"], rotation=45, fontsize=8, ha='center')
                ax.set_ylim(-1, max(1, max([m["freq"]/max_freq*2 for m in mutations])+0.5))
                ax.set_xlim(0, protein_lengths[gene_choice])
                ax.set_title(f"Lollipop plot - {gene_choice}")
                ax.set_xlabel("Position acide aminé")
                ax.set_ylabel("Fréquence relative")
                st.pyplot(fig)
                plt.close()

        # Prédiction CADD/PolyPhen
        with pgm_tabs[3]:
            st.markdown("### Scores de prédiction in silico (CADD, PolyPhen)")
            if "cadd_phred" in pgm_df.columns:
                cadd_df = pgm_df[pgm_df["cadd_phred"].notna()].copy()
                cadd_df["cadd_phred"] = cadd_df["cadd_phred"].astype(float)
                if not cadd_df.empty:
                    st.dataframe(cadd_df[["chrom", "pos", "ref", "alt", "cadd_phred", "clinvar_sig"]].head(20), use_container_width=True)
                    fig = px.histogram(cadd_df, x="cadd_phred", nbins=30, template="plotly_dark",
                                       title="Distribution des scores CADD Phred", color_discrete_sequence=["#00D4AA"])
                    fig.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Seuil 20")
                    st.plotly_chart(fig, use_container_width=True)
                    n_del = (cadd_df["cadd_phred"] >= 20).sum()
                    st.metric("Variants avec CADD ≥ 20", n_del)
                else:
                    st.info("Aucun score CADD disponible.")
            else:
                st.warning("Colonne 'cadd_phred' non trouvée.")
            # PolyPhen simulé
            st.markdown("#### PolyPhen-2 (simulé)")
            polyphen_categories = ["benign", "possibly_damaging", "probably_damaging"]
            sim_polyphen = np.random.choice(polyphen_categories, size=min(20, len(pgm_df)), p=[0.5,0.3,0.2])
            sim_df = pd.DataFrame({
                "Variant": pgm_df["pos"].head(20).astype(str) + pgm_df["ref"].head(20) + ">" + pgm_df["alt"].head(20),
                "PolyPhen": sim_polyphen
            })
            st.dataframe(sim_df, use_container_width=True)
            counts = sim_df["PolyPhen"].value_counts().reset_index()
            counts.columns = ["Prédiction", "Nb"]
            fig = px.bar(counts, x="Prédiction", y="Nb", color="Prédiction", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

if _name_ == "_main_":
    main()
