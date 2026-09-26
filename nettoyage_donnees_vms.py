"""
Nettoyage et structuration des données VMS - CRS Liberia
=========================================================
Auteur  : Labass BAH - Master 2 Data Sciences et IA, ESMT Dakar
Objet   : Extraire, valider et restructurer l'export brut du Vehicle
          Management System (VMS), e-Log Sheet, afin d'obtenir un jeu
          de données exploitable pour le modèle de prédiction de
          consommation de carburant.

Entrée  : elog.xlsx  (export VMS, une colonne de texte non structuré)
Sortie  : cycles_reels.csv  (783 cycles exploitables, pour le modèle ML)

Étapes :
    1. Parsing du texte brut ligne par ligne (expressions régulières)
    2. Contrôles de cohérence (kilométrage, prix au gallon)
    3. Classification du type de route, de la saison et du véhicule
    4. Reconstruction "plein à plein" de la consommation réelle
    5. Export du fichier final
"""

import re
from datetime import datetime

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================
FICHIER_SOURCE = "elog.xlsx"
FICHIER_CSV_MODELE = "cycles_reels.csv"

GALLON_EN_LITRES = 3.785411784

# Plage de consommation physiquement plausible (L/100 km).
# Hors de cette plage, un cycle provient très probablement d'un
# ravitaillement oublié dans le journal de bord ou d'une double saisie.
CONSO_MIN_PLAUSIBLE = 5
CONSO_MAX_PLAUSIBLE = 60

LOCALITES_PISTE = [
    'ZWEDRU', 'VOINJAMA', 'SANNIQUELLIE', 'NIMBA', 'LOFA', 'GBARPOLU',
    'RIVERCESS', 'SINOE', 'GRAND KRU', 'GRAND GEDEH', 'MARYLAND',
    'RIVER GEE', 'BARCLAYVILLE', 'FISH TOWN', 'GREENVILLE', 'HARPER',
    'PLEEBO', 'TAPPITA', 'SACLEPEA', 'BOPOLU',
]
LOCALITES_MIXTE = [
    'GBARNGA', 'BUCHANAN', 'KAKATA', 'GANTA', 'TUBMANBURG', 'ROBERTSPORT',
    'SUAKOKO', 'BONG', 'MARGIBI', 'BOMI', 'GRAND BASSA', 'SALALA',
    'TOTOTA', 'RIA', 'HARBEL',
]

MAPPING_VEHICULE = {
    'A4286': 'Toyota Prado 2020', 'A61512': 'Toyota Hardtop 2020',
    'A64592': 'Toyota Hardtop 2020', 'A65321': 'Toyota Hardtop 2020',
    'A65989': 'Toyota Hardtop 2020', 'A67971': 'Toyota Fortuner 2020',
    'A610374': 'Toyota Hardtop 2020', 'A610379': 'Toyota Hardtop 2020',
    'A610620': 'Toyota Hardtop 2024', 'B1958': 'Toyota Hardtop 2024',
    'A1795': 'Toyota Hardtop 2020', 'A539669': 'Toyota Hardtop 2020',
    'A610370': 'Toyota Hardtop 2020', 'A610373': 'Toyota Hardtop 2020',
    'A610378': 'Toyota Hardtop 2020', 'A610388': 'Toyota Hardtop 2020',
    'A614630': 'Toyota Hardtop 2024', 'A614631': 'Toyota Hardtop 2024',
    'A615920': 'Toyota Hardtop 2024', 'A68721': 'Toyota Fortuner 2020',
    'A68916': 'Toyota Hardtop 2020', 'A69060': 'Toyota Hardtop 2020',
    'A69431': 'Toyota Hardtop 2020',
}
AGE_VEHICULE = {
    'Toyota Prado 2020': 4.5, 'Toyota Fortuner 2020': 4.5,
    'Toyota Hardtop 2020': 4.5, 'Toyota Hardtop 2024': 0.5,
}


# ============================================================
# ÉTAPE 1 - PARSING DU FICHIER BRUT
# ============================================================
def parser_fichier_brut(chemin_fichier):
    """
    L'export VMS livre chaque trajet dans une seule cellule de texte,
    par exemple :
        "26/Aug/2025 A610374 16TH STREET - CONEX ... 40,777 40,790
         40,790 13 33.00 146.21 0.00 0.00 Kollie Gayflor M."
    Structure : date | plaque | destination | km_initial | km_final |
    (km_final répété) | km_parcourus | gallons | coût USD | ... | chauffeur

    Les deux premiers nombres décimaux (format X.XX) sont toujours les
    gallons achetés et le coût en USD. Les 4 derniers entiers qui les
    précèdent sont km_initial, km_final, sa répétition et km_parcourus.
    """
    brut = pd.read_excel(chemin_fichier, header=None)
    motif_ligne = re.compile(r'^(\d{2}/\w+/\d{4})\s+([A-Z]\d+)\s+(.*)$')
    motif_decimal = re.compile(r'\b\d+\.\d{2}\b')

    lignes = []
    for valeur in brut[0]:
        if not isinstance(valeur, str):
            continue
        m = motif_ligne.match(valeur.strip())
        if not m:
            continue
        date_str, plaque, reste = m.groups()

        decimales = list(motif_decimal.finditer(reste))
        if len(decimales) < 2:
            continue

        avant = reste[:decimales[0].start()]
        entiers = [int(v.replace(',', '')) for v in re.findall(r'\b[\d,]+\b', avant)
                   if v.replace(',', '').isdigit()]
        if len(entiers) < 4:
            continue

        km_initial, km_final, _repetition, km_parcourus = entiers[-4:]
        gallons = float(decimales[0].group())
        cout_usd = float(decimales[1].group())

        position = avant.rfind(f"{km_initial:,}")
        if position == -1:
            position = avant.rfind(str(km_initial))
        destination = re.sub(r'\s+', ' ', avant[:position]).strip() if position > 0 else ''
        destination = re.sub(r'\s+N/A(\s+N/A)?\s*$', '', destination).strip()

        try:
            date = datetime.strptime(date_str, '%d/%b/%Y')
        except ValueError:
            continue

        lignes.append({
            'date': date.date(), 'mois': date.month, 'plaque': plaque,
            'destination_brute': destination[:90],
            'km_initial': km_initial, 'km_final': km_final,
            'km_parcourus': km_parcourus,
            'carburant_gallons': gallons, 'cout_usd': cout_usd,
        })
    return pd.DataFrame(lignes)


# ============================================================
# ÉTAPE 2 - CONTRÔLES DE COHÉRENCE
# ============================================================
def controler_coherence(df):
    """Vérifie que le parsing a identifié les bons champs."""
    coherence = ((df['km_final'] - df['km_initial']) == df['km_parcourus']).mean() * 100
    achats = df[df['carburant_gallons'] > 0]
    prix = achats['cout_usd'] / achats['carburant_gallons']
    print(f"[Contrôle] km_parcourus = km_final - km_initial : {coherence:.1f}% des lignes")
    print(f"[Contrôle] Prix médian au gallon : {prix.median():.2f} USD")
    return df


# ============================================================
# ÉTAPE 3 - CLASSIFICATION ROUTE, SAISON, VÉHICULE
# ============================================================
def classifier_route(destination):
    d = str(destination).upper()
    if any(l in d for l in LOCALITES_PISTE):
        return 'piste'
    if any(l in d for l in LOCALITES_MIXTE):
        return 'mixte'
    return 'urbaine'


def enrichir_donnees(df):
    df = df.copy()
    df['type_route'] = df['destination_brute'].apply(classifier_route)
    df['saison'] = np.where(df['mois'].between(5, 11), 'pluies', 'seche')
    df['vehicule'] = df['plaque'].map(MAPPING_VEHICULE).fillna('Toyota Hardtop 2020')
    df['age_vehicule'] = df['vehicule'].map(AGE_VEHICULE)
    return df


# ============================================================
# ÉTAPE 4 - RECONSTRUCTION "PLEIN À PLEIN"
# ============================================================
def reconstruire_cycles_plein_a_plein(df):
    """
    Le carburant acheté sur une ligne du journal de bord n'est pas la
    consommation de ce trajet : c'est un plein qui couvre les trajets
    suivants. On reconstitue donc, pour chaque véhicule, la consommation
    entre deux ravitaillements successifs.
    """
    df = df.sort_values(['plaque', 'date', 'km_initial'])
    cycles = []

    for plaque, groupe in df.groupby('plaque'):
        groupe = groupe.reset_index(drop=True)
        pleins = groupe.index[groupe['carburant_gallons'] > 0].tolist()

        for i in range(1, len(pleins)):
            debut, fin = pleins[i - 1], pleins[i]
            segment = groupe.loc[debut + 1: fin]
            if len(segment) == 0:
                continue

            km_cycle = segment['km_parcourus'].sum()
            litres = groupe.loc[fin, 'carburant_gallons'] * GALLON_EN_LITRES
            if not (0 < km_cycle < 3000):
                continue

            conso_100km = litres / km_cycle * 100
            if not (CONSO_MIN_PLAUSIBLE <= conso_100km <= CONSO_MAX_PLAUSIBLE):
                continue

            cycles.append({
                'cycle_id': f"{plaque}_{i}",
                'plaque': plaque,
                'vehicule': groupe.loc[fin, 'vehicule'],
                'km_cycle': km_cycle,
                'litres': round(litres, 2),
                'L_100km': round(conso_100km, 2),
                'saison': groupe.loc[fin, 'saison'],
                'route_dominante': segment['type_route'].value_counts().idxmax(),
                'age_vehicule': groupe.loc[fin, 'age_vehicule'],
            })

    return pd.DataFrame(cycles)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
def main():
    print("[1/4] Parsing du fichier brut...")
    df = parser_fichier_brut(FICHIER_SOURCE)
    print(f"      {len(df)} trajets extraits.")

    print("[2/4] Contrôles de cohérence...")
    df = controler_coherence(df)

    print("[3/4] Classification route / saison / véhicule...")
    df = enrichir_donnees(df)

    print("[4/4] Reconstruction des cycles plein à plein...")
    cycles = reconstruire_cycles_plein_a_plein(df)
    print(f"      {len(cycles)} cycles exploitables : "
          f"{cycles['route_dominante'].value_counts().to_dict()}")

    cycles.to_csv(FICHIER_CSV_MODELE, index=False)
    print(f"Fichier sauvegardé : {FICHIER_CSV_MODELE}")


if __name__ == "__main__":
    main()
