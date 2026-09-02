# 🔬 MetaInsight v9 — Plateforme intégrative multi‑omique & PGM

**MetaInsight v9** est une application Streamlit conçue pour l’analyse de données multi‑omiques (microbiome, transcriptomique, etc.) et de génomique médicale de précision (PGM). Elle est prête pour le **Big Data** grâce à l’intégration de **DuckDB** et optimisée pour les fichiers issus des séquenceurs **NextSeq 2000** (VCF/FASTQ).

---

## ✨ Fonctionnalités principales

- **Microbiome** : diversité α/β, PERMANOVA, abondance différentielle (ALDEx2, LEfSe, MaAsLin2), CLR, raréfaction, prédiction fonctionnelle KEGG, biomarqueurs ROC.
- **Génomique clinique (PGM)** : filtrage des variants BRCA1/2, pharmacogénétique (CPIC), graphiques « lollipop », scores prédictifs CADD/PolyPhen.
- **Big Data** : lecture directe des fichiers VCF via DuckDB, matrices de distance optimisées.
- **Apprentissage automatique** : Random Forest, clustering, SHAP, GNN, VAE, CCA, MLP.
- **IA générative** : assistance à l’interprétation via Gemini, Groq, Ollama (clés API à configurer).
- **Rapports automatisés** : génération d’articles scientifiques et de synthèses.

---

## 📦 Prérequis

- Python **3.9** ou supérieur
- Pip (gestionnaire de paquets)

---

## 🚀 Installation

1. **Cloner le dépôt** (ou copier le fichier principal `app.py`)
   ```bash
   git clone https://github.com/votre-utilisateur/metainsight-v9.git
   cd metainsight-v9
