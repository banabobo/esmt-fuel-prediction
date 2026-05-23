"""
Interface de prédiction de consommation de carburant
CRS Liberia — Développé dans le cadre du mémoire de Master
Auteur : Labass BAH — ESMT Dakar
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="CRS Liberia — Gestion Carburant",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS PERSONNALISÉ (sobre, professionnel)
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
    }
    .alert-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
    .alert-success {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
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
    .stButton>button:hover {
        background-color: #1a3a5c;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DONNÉES ET MODÈLE — chargés une seule fois
# ============================================================
@st.cache_resource
def charger_modele():
    """Entraîne le modèle sur les données synthétiques CRS Liberia."""
    np.random.seed(42)
    n = 500

    trajets = {
        'Monrovia-Gbarnga':      {'distance': (180, 210), 'route': 'mixte'},
        'Monrovia-Buchanan':     {'distance': (95,  115), 'route': 'urbaine'},
        'Monrovia-Zwedru':       {'distance': (370, 430), 'route': 'piste'},
        'Monrovia-Voinjama':     {'distance': (430, 490), 'route': 'piste'},
        'Monrovia-Sanniquellie': {'distance': (340, 390), 'route': 'piste'},
        'Monrovia-local':        {'distance': (15,   45), 'route': 'urbaine'},
        'Gbarnga-Suakoko':       {'distance': (25,   40), 'route': 'mixte'},
        'Buchanan-Greenville':   {'distance': (180, 220), 'route': 'piste'},
    }
    vehicules = {
        'Toyota Prado 2020':    {'age': 4, 'conso_base': 12.5},
        'Toyota Fortuner 2020': {'age': 4, 'conso_base': 13.0},
        'Toyota Hardtop 2020':  {'age': 4, 'conso_base': 11.8},
        'Toyota Hardtop 2024':  {'age': 0, 'conso_base': 10.5},
    }

    data = []
    for _ in range(n):
        t = np.random.choice(list(trajets.keys()))
        v = np.random.choice(list(vehicules.keys()))
        saison = np.random.choice(['seche', 'pluies'], p=[0.45, 0.55])
        charge = np.random.choice(['legere', 'moyenne', 'lourde'], p=[0.4, 0.4, 0.2])
        type_route = trajets[t]['route']
        distance = np.random.uniform(*trajets[t]['distance'])
        vit_base = {'urbaine': 35, 'mixte': 55, 'piste': 38}[type_route]
        vitesse = max(20, min(90, np.random.normal(vit_base, 6)))
        age = vehicules[v]['age'] + np.random.uniform(0, 0.8)
        conso_base = vehicules[v]['conso_base']

        kr = {'urbaine': 1.0, 'mixte': 1.25, 'piste': 1.65}[type_route]
        ks = {'urbaine': 1.05, 'mixte': 1.15, 'piste': 1.30}[type_route] if saison == 'pluies' else 1.0
        kc = {'legere': 1.0, 'moyenne': 1.10, 'lourde': 1.22}[charge]
        kv = 1 + 0.003 * (vitesse - 70)**2 / 100
        ka = 1 + max(0, (age - 2)) * 0.02

        conso = conso_base * kr * ks * kc * kv * ka + np.random.normal(0, 0.8)
        conso = max(8, conso)
        data.append({
            'distance_km': distance,
            'type_route':  type_route,
            'saison':      saison,
            'vehicule':    v,
            'vitesse_moy': vitesse,
            'age_vehicule': age,
            'charge':      charge,
            'consommation': (conso / 100) * distance,
        })

    df = pd.DataFrame(data)
    le_route   = LabelEncoder().fit(df['type_route'])
    le_saison  = LabelEncoder().fit(df['saison'])
    le_charge  = LabelEncoder().fit(df['charge'])
    le_veh     = LabelEncoder().fit(df['vehicule'])

    df['route_enc']  = le_route.transform(df['type_route'])
    df['saison_enc'] = le_saison.transform(df['saison'])
    df['charge_enc'] = le_charge.transform(df['charge'])
    df['veh_enc']    = le_veh.transform(df['vehicule'])

    X = df[['distance_km','route_enc','saison_enc','veh_enc','vitesse_moy','age_vehicule','charge_enc']]
    y = df['consommation']

    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model, le_route, le_saison, le_charge, le_veh, df


model, le_route, le_saison, le_charge, le_veh, df_historique = charger_modele()

# ============================================================
# EN-TÊTE
# ============================================================
st.markdown("""
<div class="main-header">
    <h2 style="margin:0">⛽ Système de Prédiction Carburant — CRS Liberia</h2>
    <p style="margin:0.3rem 0 0 0; opacity:0.85">
        Outil d'aide à la planification logistique basé sur l'apprentissage supervisé
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# NAVIGATION SIDEBAR
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/CRS_logo.svg/320px-CRS_logo.svg.png",
                 width=160, caption="Catholic Relief Services")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["🔮 Prédiction Mission", "📊 Tableau de Bord", "ℹ️ À propos du modèle"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Modèle actuel :** Gradient Boosting")
st.sidebar.markdown("**R² :** 0.9926")
st.sidebar.markdown("**MAE :** 2.43 litres")
st.sidebar.markdown("**Données :** Synthétiques (n=500)")

# ============================================================
# PAGE 1 — PRÉDICTION
# ============================================================
if page == "🔮 Prédiction Mission":
    st.subheader("Estimer la consommation d'une mission")
    st.markdown("Remplissez les paramètres ci-dessous pour obtenir une estimation du carburant nécessaire.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Paramètres de la mission**")

        trajet = st.selectbox("Trajet", [
            "Monrovia-Gbarnga", "Monrovia-Buchanan", "Monrovia-Zwedru",
            "Monrovia-Voinjama", "Monrovia-Sanniquellie",
            "Monrovia-local", "Gbarnga-Suakoko", "Buchanan-Greenville",
            "Trajet personnalisé"
        ])

        distances_ref = {
            'Monrovia-Gbarnga': 195, 'Monrovia-Buchanan': 105,
            'Monrovia-Zwedru': 400, 'Monrovia-Voinjama': 460,
            'Monrovia-Sanniquellie': 365, 'Monrovia-local': 30,
            'Gbarnga-Suakoko': 32, 'Buchanan-Greenville': 200,
            'Trajet personnalisé': 100,
        }
        routes_ref = {
            'Monrovia-Gbarnga': 'mixte', 'Monrovia-Buchanan': 'urbaine',
            'Monrovia-Zwedru': 'piste', 'Monrovia-Voinjama': 'piste',
            'Monrovia-Sanniquellie': 'piste', 'Monrovia-local': 'urbaine',
            'Gbarnga-Suakoko': 'mixte', 'Buchanan-Greenville': 'piste',
            'Trajet personnalisé': 'mixte',
        }

        distance = st.number_input(
            "Distance (km)",
            min_value=5, max_value=600,
            value=distances_ref[trajet]
        )
        type_route = st.selectbox(
            "Type de route",
            ["urbaine", "mixte", "piste"],
            index=["urbaine", "mixte", "piste"].index(routes_ref[trajet])
        )
        saison = st.selectbox("Saison", ["seche", "pluies"],
                               format_func=lambda x: "🌞 Saison sèche" if x=="seche" else "🌧️ Saison des pluies")
        charge = st.selectbox("Charge transportée",
                               ["legere", "moyenne", "lourde"],
                               format_func=lambda x: {"legere":"Légère (personnel)","moyenne":"Moyenne (matériel mixte)","lourde":"Lourde (vivres / équipement)"}[x])

    with col2:
        st.markdown("**Paramètres du véhicule**")

        vehicule = st.selectbox("Véhicule", [
            "Toyota Prado 2020", "Toyota Fortuner 2020",
            "Toyota Hardtop 2020", "Toyota Hardtop 2024"
        ])
        ages_ref = {
            'Toyota Prado 2020': 4.2, 'Toyota Fortuner 2020': 4.2,
            'Toyota Hardtop 2020': 4.2, 'Toyota Hardtop 2024': 0.5
        }
        age_vehicule = st.slider(
            "Âge du véhicule (années)",
            min_value=0.0, max_value=10.0,
            value=ages_ref[vehicule], step=0.5
        )
        vitesse = st.slider(
            "Vitesse moyenne estimée (km/h)",
            min_value=20, max_value=90,
            value={'urbaine': 35, 'mixte': 55, 'piste': 38}[type_route]
        )

        st.markdown("&nbsp;")
        st.markdown("**Estimation du temps de trajet**")
        temps_h = distance / vitesse
        st.info(f"⏱️ Durée estimée : **{int(temps_h)}h{int((temps_h%1)*60):02d}**")

    # --- CALCUL ---
    st.markdown("---")
    if st.button("⛽ Calculer la consommation estimée"):
        route_enc  = le_route.transform([type_route])[0]
        saison_enc = le_saison.transform([saison])[0]
        charge_enc = le_charge.transform([charge])[0]
        veh_enc    = le_veh.transform([vehicule])[0]

        X_new = np.array([[distance, route_enc, saison_enc, veh_enc,
                           vitesse, age_vehicule, charge_enc]])
        prediction = model.predict(X_new)[0]
        marge = prediction * 0.05  # ±5% MAE approximative

        # Comparaison saison opposée
        saison_opp = 'seche' if saison == 'pluies' else 'pluies'
        saison_enc_opp = le_saison.transform([saison_opp])[0]
        X_opp = np.array([[distance, route_enc, saison_enc_opp, veh_enc,
                           vitesse, age_vehicule, charge_enc]])
        pred_opp = model.predict(X_opp)[0]
        diff = prediction - pred_opp

        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.markdown(f"""
            <div class="prediction-result">
                <div style="font-size:0.9rem; opacity:0.8">Consommation estimée</div>
                <div style="font-size:2.8rem; font-weight:bold">{prediction:.1f} L</div>
                <div style="font-size:0.8rem; opacity:0.7">± {marge:.1f} L (±5%)</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            conso_100 = (prediction / distance) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666">Consommation aux 100 km</div>
                <div style="font-size:1.8rem; font-weight:bold; color:#1a3a5c">{conso_100:.1f} L/100</div>
            </div>
            """, unsafe_allow_html=True)

            signe = "+" if diff > 0 else ""
            couleur = "#dc3545" if diff > 0 else "#28a745"
            label_opp = "saison sèche" if saison == 'pluies' else "saison des pluies"
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666">vs {label_opp}</div>
                <div style="font-size:1.5rem; font-weight:bold; color:{couleur}">{signe}{diff:.1f} L</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r3:
            # Alerte si surconsommation probable
            if type_route == 'piste' and saison == 'pluies':
                st.markdown("""
                <div class="alert-warning" style ="color:#333">
                    <strong>Attention</strong><br>
                    Piste + saison des pluies.<br>
                    Prévoir une réserve supplémentaire de 10%.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-success" style ="color:#333">
                    <strong>Conditions normales</strong><br>
                    Aucune alerte particulière<br>
                    pour cette mission.
                </div>
                """, unsafe_allow_html=True)

            # Recommandation carte TOM
            tom_recommande = prediction * 1.10  # +10% de marge
            prix_litre = 2.45  # USD, prix approximatif au Liberia
            cout_usd = tom_recommande * prix_litre
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#666">Recharge TOM recommandée</div>
                <div style="font-size:1.4rem; font-weight:bold; color:#1a3a5c">{tom_recommande:.0f} L</div>
                <div style="font-size:0.8rem; color:#666">≈ USD {cout_usd:.0f}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE 2 — TABLEAU DE BORD
# ============================================================
elif page == "📊 Tableau de Bord":
    st.subheader("Tableau de bord — Analyse des consommations")

    # KPIs globaux
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missions simulées", "500", help="Taille du dataset d'entraînement")
    with col2:
        st.metric("Conso. moyenne", f"{df_historique['consommation'].mean():.1f} L", help="Par mission")
    with col3:
        st.metric("Distance moyenne", f"{df_historique['distance_km'].mean():.0f} km", help="Par mission")
    with col4:
        st.metric("L/100 moyen", f"{(df_historique['consommation']/df_historique['distance_km']*100).mean():.1f}", help="Consommation normalisée")

    st.markdown("---")

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("**Consommation par type de route et saison**")
        fig, ax = plt.subplots(figsize=(7, 4))
        pivot = df_historique.groupby(['type_route','saison'])['consommation'].mean().unstack()
        pivot.plot(kind='bar', ax=ax, color=['#F39C12','#2E86AB'], edgecolor='white', width=0.7)
        ax.set_ylabel('Litres (moyenne)')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title='Saison', labels=['Pluies','Sèche'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.markdown("**Distribution de la consommation par véhicule**")
        fig, ax = plt.subplots(figsize=(7, 4))
        vehicules_list = df_historique['vehicule'].unique()
        data_box = [df_historique[df_historique['vehicule']==v]['consommation'].values
                    for v in vehicules_list]
        bp = ax.boxplot(data_box, patch_artist=True, notch=False)
        colors = ['#2E86AB','#27AE60','#F39C12','#8E44AD']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(['Prado', 'Fortuner', 'Htop\n2020', 'Htop\n2024'], fontsize=9)
        ax.set_ylabel('Litres par mission')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Tableau des trajets
    st.markdown("---")
    st.markdown("**Consommation moyenne par trajet**")
    tableau = df_historique.groupby('type_route').agg(
        Missions=('consommation','count'),
        Distance_moy=('distance_km', lambda x: f"{x.mean():.0f} km"),
        Conso_moy=('consommation', lambda x: f"{x.mean():.1f} L"),
        L100_moy=('consommation', lambda x: f"{(x/df_historique.loc[x.index,'distance_km']*100).mean():.1f}")
    ).reset_index()
    tableau.columns = ['Type de route','Missions','Distance moy.','Conso. moy.','L/100 moy.']
    st.dataframe(tableau, use_container_width=True, hide_index=True)

# ============================================================
# PAGE 3 — À PROPOS
# ============================================================
elif page == "ℹ️ À propos du modèle":
    st.subheader("À propos du modèle de prédiction")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        **Contexte**

        Ce système a été développé dans le cadre d'un mémoire de Master 2 en Data Sciences
        et Intelligence Artificielle à l'ESMT (École Supérieure Multinationale des Télécommunications,
        Dakar). Il vise à répondre à un besoin opérationnel concret de CRS Liberia : améliorer
        la planification du carburant pour une flotte de 13 véhicules 4x4 opérant dans des
        conditions logistiques difficiles.

        **Méthodologie**

        Le modèle est un **Gradient Boosting Regressor** entraîné sur 500 observations synthétiques
        générées à partir des paramètres opérationnels de CRS Liberia. Les coefficients
        de la simulation (effet de la saison, du type de route, de la charge) sont fondés sur
        des observations terrain documentées.

        **Limites**

        Ce prototype est développé sur données synthétiques. La prochaine étape est de le
        valider et ré-entraîner sur les données réelles du VMS de CRS Liberia.
        """)

    with col2:
        st.markdown("**Performances du modèle**")
        metrics = {
            "Algorithme": "Gradient Boosting",
            "R²": "0.9926",
            "R² CV (5-fold)": "0.9917",
            "RMSE": "3.70 litres",
            "MAE": "2.43 litres",
            "Taille dataset": "500 missions",
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
    st.markdown("**Variables du modèle**")
    vars_df = pd.DataFrame({
        "Variable": ["Distance (km)", "Type de route", "Saison", "Véhicule",
                     "Vitesse moyenne", "Âge véhicule", "Charge"],
        "Type": ["Continue", "Catégorielle", "Catégorielle", "Catégorielle",
                 "Continue", "Continue", "Catégorielle"],
        "Importance (RF)": ["0.929 ★★★", "0.006", "0.037 ★", "0.003",
                            "0.004", "0.014", "0.007"],
    })
    st.dataframe(vars_df, use_container_width=True, hide_index=True)
