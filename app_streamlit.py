"""
Interface de prédiction de consommation de carburant
CRS Liberia - Développé dans le cadre du mémoire de Master
Auteur : Labass BAH - ESMT Dakar
Version 2 - Entraîné sur données réelles du VMS (783 cycles plein-à-plein)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="CRS Liberia - Gestion Carburant",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1a3a5c, #2E86AB);
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2E86AB;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #222;
    }
    .alert-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #333;
    }
    .alert-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #333;
    }
    .alert-info {
        background: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        color: #333;
    }
    .prediction-result {
        background: #1a3a5c;
        color: white;
        padding: 2rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        border: none;
        padding: 0.6rem 2rem;
        border-radius: 4px;
        font-weight: bold;
        width: 100%;
    }
    .stButton>button:hover { background-color: #1a3a5c; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DES DONNÉES RÉELLES + ENTRAÎNEMENT
# ============================================================
DATA_FILE = "cycles_reels.csv"

@st.cache_resource
def charger_modele():
    """Charge les 783 cycles réels (VMS, oct.2024-nov.2025) et entraîne le modèle."""
    if not os.path.exists(DATA_FILE):
        st.error(f"Fichier de données introuvable : {DATA_FILE}. "
                 f"Il doit être présent dans le même dossier que l'application.")
        st.stop()

    df = pd.read_csv(DATA_FILE)

    le_route  = LabelEncoder().fit(df['route_dominante'])
    le_saison = LabelEncoder().fit(df['saison'])
    le_veh    = LabelEncoder().fit(df['vehicule'])

    df['route_enc']  = le_route.transform(df['route_dominante'])
    df['saison_enc'] = le_saison.transform(df['saison'])
    df['veh_enc']    = le_veh.transform(df['vehicule'])

    X = df[['km_cycle', 'route_enc', 'saison_enc', 'veh_enc', 'age_vehicule']]
    y = df['litres']

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_route, le_saison, le_veh, df


model, le_route, le_saison, le_veh, df_historique = charger_modele()

# ============================================================
# EN-TÊTE
# ============================================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">⛽ Système de Prédiction Carburant - CRS Liberia</h2>
    <p style="margin:0.3rem 0 0 0; opacity:0.85">
        Modèle entraîné sur 783 cycles réels extraits du Vehicle Management System (VMS)
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### ⛽ CRS Liberia")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction Mission", "📊 Tableau de Bord", "ℹ️ À propos du modèle"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle actuel :** Gradient Boosting")
st.sidebar.markdown("**Données :** 783 cycles réels (VMS)")
st.sidebar.markdown("**Période :** Oct. 2024 – Nov. 2025")
st.sidebar.markdown("**R² (test) :** 0.38")
st.sidebar.markdown("**MAE :** 13.4 litres")
st.sidebar.info(
    "⚠️ Modèle en phase de validation. La performance actuelle reflète "
    "la variabilité réelle du terrain (comportement chauffeur, état des "
    "routes, entretien) - voir page 'À propos'.",
    icon="ℹ️"
)

# ============================================================
# PAGE 1 - PRÉDICTION
# ============================================================
if page == "🔮 Prédiction Mission":
    st.subheader("Estimer la consommation d'un cycle de mission")
    st.markdown(
        "Cette estimation porte sur un **cycle plein-à-plein** - c'est-à-dire l'ensemble "
        "des trajets qu'un véhicule effectue entre deux ravitaillements, et non un trajet unique. "
        "C'est la méthode utilisée pour entraîner le modèle à partir des données réelles du VMS."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Paramètres du cycle**")
        km_cycle = st.number_input(
            "Distance totale du cycle (km)",
            min_value=20, max_value=2500, value=450,
            help="Somme des kilomètres parcourus depuis le dernier plein"
        )
        type_route = st.selectbox(
            "Type de route dominant",
            ["urbaine", "mixte", "piste"],
            format_func=lambda x: {"urbaine": "Urbaine (Monrovia)",
                                    "mixte": "Mixte (Gbarnga, Buchanan...)",
                                    "piste": "Piste (Zwedru, Voinjama...)"}[x]
        )
        saison = st.selectbox(
            "Saison", ["seche", "pluies"],
            format_func=lambda x: "🌞 Saison sèche" if x == "seche" else "🌧️ Saison des pluies"
        )

    with col2:
        st.markdown("**Véhicule**")
        vehicule = st.selectbox(
            "Véhicule",
            sorted(df_historique['vehicule'].unique())
        )
        age_vehicule = float(
            df_historique.loc[df_historique['vehicule'] == vehicule, 'age_vehicule'].iloc[0]
        )
        st.metric("Âge du véhicule", f"{age_vehicule:.1f} ans")

        st.markdown("&nbsp;")
        st.markdown("""
        <div class="alert-info">
        ℹ️ La vitesse moyenne et la charge transportée ne sont pas
        enregistrées dans le VMS actuel. Elles ne sont donc pas
        utilisées par ce modèle - voir Perspectives du mémoire.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("⛽ Calculer la consommation estimée"):
        route_enc  = le_route.transform([type_route])[0]
        saison_enc = le_saison.transform([saison])[0]
        veh_enc    = le_veh.transform([vehicule])[0]

        X_new = np.array([[km_cycle, route_enc, saison_enc, veh_enc, age_vehicule]])
        prediction = model.predict(X_new)[0]
        marge = 13.4  # MAE du modèle sur données réelles

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown(f"""
            <div class="prediction-result">
                <div style="font-size:0.9rem; opacity:0.8">Consommation estimée du cycle</div>
                <div style="font-size:2.8rem; font-weight:bold">{prediction:.1f} L</div>
                <div style="font-size:0.8rem; opacity:0.7">± {marge:.1f} L (MAE du modèle)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            conso_100 = (prediction / km_cycle) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666">Consommation aux 100 km</div>
                <div style="font-size:1.8rem; font-weight:bold; color:#1a3a5c">{conso_100:.1f} L/100</div>
            </div>
            """, unsafe_allow_html=True)

            tom_recommande = prediction * 1.15
            prix_litre = 1.19  # ≈ 4.5 USD/gal ÷ 3.785
            cout_usd = tom_recommande * prix_litre
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666">Recharge TOM recommandée (+15%)</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#1a3a5c">{tom_recommande:.0f} L</div>
                <div style="font-size:0.8rem; color:#666">≈ USD {cout_usd:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r3:
            if type_route == 'piste' and saison == 'pluies':
                st.markdown("""
                <div class="alert-warning">
                    ⚠️ <strong>Attention</strong><br>
                    Piste + saison des pluies.<br>
                    Marge d'erreur du modèle plus élevée sur ce segment (peu d'observations).
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success">
                    ✅ <strong>Conditions courantes</strong><br>
                    Ce type de cycle est bien représenté<br>
                    dans les données d'entraînement.
                </div>
                """, unsafe_allow_html=True)

# ============================================================
# PAGE 2 - TABLEAU DE BORD
# ============================================================
elif page == "📊 Tableau de Bord":
    st.subheader("Tableau de bord - Analyse des cycles réels")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cycles analysés", f"{len(df_historique)}", help="Cycles plein-à-plein, VMS")
    with col2:
        st.metric("Conso. moyenne/cycle", f"{df_historique['litres'].mean():.1f} L")
    with col3:
        st.metric("Distance moy./cycle", f"{df_historique['km_cycle'].mean():.0f} km")
    with col4:
        st.metric("L/100 moyen", f"{df_historique['L_100km'].mean():.1f}")

    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Consommation moyenne (L/100km) par type de route**")
        fig, ax = plt.subplots(figsize=(7, 4))
        moy = df_historique.groupby('route_dominante')['L_100km'].mean().sort_values()
        ax.bar(moy.index, moy.values, color=['#27AE60', '#F39C12', '#E74C3C'], edgecolor='white')
        ax.set_ylabel('L/100km')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.markdown("**Distribution de la consommation par véhicule**")
        fig, ax = plt.subplots(figsize=(7, 4))
        vehicules_list = df_historique['vehicule'].unique()
        data_box = [df_historique[df_historique['vehicule'] == v]['L_100km'].values
                    for v in vehicules_list]
        bp = ax.boxplot(data_box, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#2E86AB')
            patch.set_alpha(0.7)
        ax.set_xticklabels(vehicules_list, rotation=30, ha='right', fontsize=8)
        ax.set_ylabel('L/100km')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**Répartition des cycles par type de route**")
    tableau = df_historique.groupby('route_dominante').agg(
        Cycles=('litres', 'count'),
        Distance_moy=('km_cycle', lambda x: f"{x.mean():.0f} km"),
        Conso_moy=('litres', lambda x: f"{x.mean():.1f} L"),
        L100_moy=('L_100km', lambda x: f"{x.mean():.1f}")
    ).reset_index()
    tableau.columns = ['Type de route', 'Cycles', 'Distance moy.', 'Conso. moy.', 'L/100 moy.']
    st.dataframe(tableau, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3 - À PROPOS
# ============================================================
elif page == "ℹ️ À propos du modèle":
    st.subheader("À propos du modèle de prédiction")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        **Contexte**

        Ce système a été développé dans le cadre d'un mémoire de Master 2 en Data Sciences
        et Intelligence Artificielle à l'ESMT Dakar, pour répondre à un besoin opérationnel
        de CRS Liberia : améliorer la planification du carburant pour sa flotte de véhicules.

        **Origine des données**

        Le modèle est entraîné sur **783 cycles réels** extraits du Vehicle Management System
        (VMS) de CRS Liberia, couvrant la période d'octobre 2024 à novembre 2025.

        Un point méthodologique important : le carburant acheté lors d'un trajet donné n'est
        pas la consommation de ce trajet - c'est un plein qui couvre les trajets à venir. La
        consommation réelle a donc été reconstituée en cumulant les kilomètres parcourus
        **entre deux ravitaillements successifs** du même véhicule ("cycle plein-à-plein"),
        une méthode standard en gestion de flotte.

        **Limites actuelles**

        La performance du modèle (R² ≈ 0,38) est **nettement plus modeste** que sur des
        données synthétiques, et c'est attendu : les données réelles intègrent toute la
        variabilité du terrain - style de conduite, état du véhicule, trafic, arrêts non
        enregistrés - qu'aucune variable actuelle ne capture complètement. Le modèle ne
        dispose pas non plus de la vitesse moyenne ni de la charge transportée, deux facteurs
        connus pour influencer la consommation, faute d'enregistrement dans le VMS actuel.
        """)

    with col2:
        st.markdown("**Performances du modèle (données réelles)**")
        metrics = {
            "Algorithme": "Gradient Boosting",
            "R² (test)": "0.384",
            "R² CV (5-fold)": "0.165",
            "RMSE": "23.9 litres",
            "MAE": "13.4 litres",
            "Cycles utilisés": "783",
            "Train / Test": "80% / 20%",
        }
        for k, v in metrics.items():
            st.markdown(f"""
            <div class="metric-card">
                <span style="color:#666; font-size:0.85rem">{k}</span><br>
                <strong>{v}</strong>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Variables utilisées**")
    vars_df = pd.DataFrame({
        "Variable": ["Distance du cycle (km)", "Type de route dominant", "Saison", "Véhicule", "Âge du véhicule"],
        "Type": ["Continue", "Catégorielle", "Catégorielle", "Catégorielle", "Continue"],
        "Disponible dans le VMS": ["Oui (calculée)", "Oui (déduite)", "Oui (déduite de la date)", "Oui", "Oui"],
    })
    st.dataframe(vars_df, use_container_width=True, hide_index=True)

    st.markdown("**Variables absentes - perspectives d'amélioration**")
    st.markdown("""
    - Vitesse moyenne du chauffeur (nécessiterait une intégration V-Tron)
    - Charge transportée (à ajouter au formulaire du journal de bord)
    - État de la route au moment du trajet (donnée externe, ex. météo)
    """)
