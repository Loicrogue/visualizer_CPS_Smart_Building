import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Import de la fonction d'exportation depuis ton fichier rapport.py
from rapport import generate_pdf

# Config de la page Streamlit
st.set_page_config(page_title="CPS - Visualisation Données", layout="wide", page_icon="🏢")

st.title("🏢 Cyber-Physical System (CPS) - Smart Building")

# ── 1. FONCTION DE CHARGEMENT ET NETTOYAGE INTELLIGENT ────────────────
@st.cache_data
def traiter_donnees_cps(chemin_fichier):
    # Chargement initial sans index pour analyser la colonne Date
    df = pd.read_csv(chemin_fichier)
    nom_fichier = Path(chemin_fichier).name
    
    # Conversion des dates en forçant les erreurs à devenir des "NaT" (Not a Time)
    dates_converties = pd.to_datetime(df['Date'], errors='coerce')
    
    # Détection des lignes corrompues (ex: le "2" trouvé en fin de fichier)
    lignes_corrompues = df[dates_converties.isna()]
    
    if not lignes_corrompues.empty:
        for idx, row in lignes_corrompues.iterrows():
            # index Python (0) + 1 (Entête) + 1 (Alignement Excel) = idx + 2
            numero_ligne_fichier = idx + 2
            valeur_erronee = row['Date']
            
            # Affichage d'une alerte claire dans la barre latérale
            st.sidebar.warning(
                f"⚠️ **Donnée parasite nettoyée**\n"
                f"• Fichier : `{nom_fichier}`\n"
                f"• Ligne CSV : `Ligne {numero_ligne_fichier}`\n"
                f"• Valeur : `\"{valeur_erronee}\"`\n"
                f"👉 *Ligne ignorée automatiquement.*"
            )
            
        # Suppression des lignes corrompues
        df = df[dates_converties.notna()]
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        df['Date'] = dates_converties

    # Application de l'indexation temporelle saine
    df = df.set_index('Date')
    df = df.ffill().bfill()
    
    # Détection des zones (z1, z2, etc.) et des capteurs
    colonnes = df.columns
    zones = sorted(list(set([re.match(r"(z\d+)_", col).group(1) for col in colonnes if re.match(r"(z\d+)_", col)])))
    
    cols_t = [col for col in colonnes if "(degC)" in col]
    cols_rh = [col for col in colonnes if "(RH%)" in col]
    
    # Sécurité si aucune colonne thermique n'est trouvée
    if not cols_t:
        df['BetP_comfort'] = 0.5
        df['K_conflict'] = 0.0
        df['ECMI_waste'] = 0.0
        df['AC_consumption'] = 0.0
        df['DST_decision'] = "Confort"
        df['IF_decision'] = "Confort"
        df['Recommandation_4h_consigne'] = "Confort"
        df['Moyenne_RH'] = 50.0
        df['Moyenne_T'] = 23.0
        return df

    # --- ISOLATION FOREST MULTIVARIÉ ---
    cols_if = cols_t + cols_rh
    clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    if_preds = clf.fit_predict(df[cols_if])
    df['IF_decision'] = np.where(if_preds == -1, "Inconfort", "Confort")

    # --- THÉORIE DE DEMPSTER-SHAFER (DST) ---
    seuils_t = (20.0, 26.0)
    seuils_rh = (30.0, 70.0)
    
    mat_t = df[cols_t].values
    prox_t = np.clip(1.0 - (np.abs(mat_t - 23.0) / 3.0), 0, 1)
    m_c_t = np.where((mat_t >= seuils_t[0]) & (mat_t <= seuils_t[1]), 0.5 + 0.4 * prox_t, 0.0)
    m_i_t = np.where((mat_t < seuils_t[0]) | (mat_t > seuils_t[1]), np.minimum(0.9, 0.4 + 0.1 * np.minimum(np.abs(mat_t - seuils_t[0]), np.abs(mat_t - seuils_t[1]))), 0.0)
    
    if cols_rh:
        mat_rh = df[cols_rh].values
        prox_rh = np.clip(1.0 - (np.abs(mat_rh - 50.0) / 20.0), 0, 1)
        m_c_rh = np.where((mat_rh >= seuils_rh[0]) & (mat_rh <= seuils_rh[1]), 0.5 + 0.4 * prox_rh, 0.0)
        m_i_rh = np.where((mat_rh < seuils_rh[0]) | (mat_rh > seuils_rh[1]), np.minimum(0.9, 0.4 + 0.1 * np.minimum(np.abs(mat_rh - seuils_rh[0]), np.abs(mat_rh - seuils_rh[1]))), 0.0)
        
        df['BetP_comfort'] = np.mean((m_c_t + m_c_rh) / 2.0 + (1.0 - (m_c_t+m_c_rh)/2.0 - (m_i_t+m_i_rh)/2.0) * 0.5, axis=1)
        df['K_conflict'] = np.clip((np.std(mat_t, axis=1)/10.0 + np.std(mat_rh, axis=1)/40.0) / 2.0, 0.0, 0.95)
        df['Moyenne_RH'] = np.mean(mat_rh, axis=1)
    else:
        df['BetP_comfort'] = np.mean(m_c_t + (1.0 - m_c_t - m_i_t) * 0.5, axis=1)
        df['K_conflict'] = np.clip(np.std(mat_t, axis=1) / 10.0, 0.0, 0.95)
        df['Moyenne_RH'] = 50.0

    # AJOUT : Température Moyenne Globale brute de l'étage
    df['Moyenne_T'] = np.mean(mat_t, axis=1)
    df['DST_decision'] = np.where(df['BetP_comfort'] >= 0.5, "Confort", "Inconfort")

    # --- CALCUL ÉNERGÉTIQUE ET ECMI ---
    df['AC_consumption'] = 0.0
    df['ECMI_waste'] = 0.0
    for zone in zones:
        cols_ac_zone = [col for col in colonnes if col.startswith(zone) and "(kW)" in col and "AC" in col]
        col_t_zone = f"{zone}_S1(degC)"
        if cols_ac_zone:
            conso_ac_zone = df[cols_ac_zone].sum(axis=1)
            df['AC_consumption'] += conso_ac_zone * (1.0 / 60.0)
            if col_t_zone in df.columns:
                t_zone = df[col_t_zone].values
                gaspillage_zone = np.where((t_zone >= 21.5) & (t_zone <= 24.5) & (conso_ac_zone > 0.05), conso_ac_zone, 0.0)
                df['ECMI_waste'] += gaspillage_zone * (1.0 / 60.0)

    # Anticipation +4 heures (240 minutes)
    df['Recommandation_4h_consigne'] = df['DST_decision'].shift(-240).fillna(df['DST_decision'])
    
    return df


# ── 2. INTERFACE SIDEBAR (CONFIGURATION) ─────────────────────────────
st.sidebar.header("📊 Configuration FUTURE")

annee_sel = st.sidebar.selectbox("📅 Sélectionner l'Année", [2018, 2019])
etage_sel = st.sidebar.selectbox("🏢 Choisir l'Étage", [3, 4, 5, 6, 7])
gran_sel = st.sidebar.selectbox("📊 Granularité temporelle", ["Mensuel (M)", "Quotidien (D)", "Horaire (h)", "Minute (min)"])

map_resample = {"Mensuel (M)": "ME", "Quotidien (D)": "D", "Horaire (h)": "h", "Minute (min)": "min"}

# Construction du chemin du fichier cible
chemin_final = f"data/raw/{annee_sel}Floor{etage_sel}.csv"

if not Path(chemin_final).exists():
    st.error(f"Le fichier `{chemin_final}` est introuvable. Veuillez verifier votre dossier `data/`.")
else:
    # Exécution du traitement de données sécurisé
    df_traite = traiter_donnees_cps(chemin_final)
    
    # Rééchantillonnage temporel selon la granularité choisie
    df_visu = df_traite.resample(map_resample[gran_sel]).mean(numeric_only=True)

    # ── 3. AFFICHAGE DES CARTES DE MÉTRIQUES (4 COLONNES) ─────────────
    t_moyen = df_visu['Moyenne_T'].mean()
    rh_moyen = df_visu['Moyenne_RH'].mean()
    betp_moyen = df_visu['BetP_comfort'].mean() * 100
    conflit_moyen = df_visu['K_conflict'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confort Global Moyen (BetP)", f"{betp_moyen:.1f} %")
    with col2:
        st.metric("Niveau de Conflit (K)", f"{conflit_moyen:.2f}")
    with col3:
        st.metric("Température Moyenne", f"{t_moyen:.1f} °C")
    with col4:
        st.metric("Humidité Relative Moyenne", f"{rh_moyen:.1f} %")

    st.markdown("---")

    # ── 4. CONTEXTE ET SHIFT PRÉDICTIF (+4H) ──────────────────────────
    st.subheader("🔮 Anticipation de Consigne Thermique / Hygrométrique (+4h)")
    derniere_pred = df_traite['Recommandation_4h_consigne'].iloc[-1]
    
    if "confort" in derniere_pred.lower():
        st.success("✅ **Stable** : Confort thermique et respiratoire garanti sur les 4 prochaines heures.")
    else:
        st.error("⚠️ **Alerte Inconfort** : Une derive d'ambiance est prevue d'ici 4 heures. Ajustement de la consigne requis.")

    # ── 5. ONGLETS DE NAVIGATION DU DASHBOARD ──────────────────────────
    onglet_dst, onglet_comp, onglet_ener = st.tabs([
        "📈 DST & Confort Multi-variable", 
        "🤖 Comparaison DST vs Isolation Forest", 
        "⚡ ECMI & Gaspillage Énergétique"
    ])

    # ── ONGLET 1 : GRAPHIQUE DOUBLE AXE AVEC TEMPÉRATURE & HUMIDITÉ ────
    with onglet_dst:
        st.write("### Évolution temporelle : Indicateurs DST vs Environnement")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Axe principal gauche (0 à 100%) : DST & Humidité
        fig.add_trace(go.Scatter(x=df_visu.index, y=df_visu['BetP_comfort']*100, name="Confort Global (BetP %)", line=dict(color='#2ECC71', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_visu.index, y=df_visu['K_conflict']*100, name="Conflit Multi-capteurs (K %)", line=dict(color='#E74C3C', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_visu.index, y=df_visu['Moyenne_RH'], name="Humidité Relative (RH %)", line=dict(color='#3498DB', width=1.5)), secondary_y=False)
        
        # Axe secondaire droit : Température en °C
        fig.add_trace(go.Scatter(x=df_visu.index, y=df_visu['Moyenne_T'], name="Température Moyenne (°C)", line=dict(color='#F39C12', width=2.5, dash='dash')), secondary_y=True)
        
        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        fig.update_yaxes(
        title_text="<b>DST</b> : <span style='color:#2ECC71'>Confort</span>, <span style='color:#E74C3C'>Conflit</span> & <span style='color:#3498DB'>Humidité</span> (%)", 
        secondary_y=False, 
        range=[0, 105]
    )    
    fig.update_yaxes(
        title_text="<b>Environnement</b> : <span style='color:#F39C12'>Température</span> (°C)", 
        secondary_y=True, 
        range=[18, 30]
    )
        
    st.plotly_chart(fig, use_container_width=True)

    # ── ONGLET 2 : COMPARAISONS DES MODÈLES ───────────────────────────
    with onglet_comp:
        st.header("Validation Croisée : Modèle Physique Avancé vs Statistique")
        accord = (df_traite['DST_decision'] == df_traite['IF_decision']).mean() * 100
        st.metric("Taux d'accord global DST / Isolation Forest", f"{accord:.1f} %")
        df_div = df_traite[df_traite['DST_decision'] != df_traite['IF_decision']]
        st.dataframe(df_div[['DST_decision', 'IF_decision', 'BetP_comfort', 'K_conflict', 'Moyenne_RH']].head(50))

        df_match = df_traite.resample(map_resample[gran_sel]).last()
        
        fig_comp = px.line(df_visu, y=['BetP_comfort'], title="Stabilite du Critere de Croyance Pignistique (BetP)")
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── ONGLET 3 : ANALYSE ÉNERGÉTIQUE (ECMI) ─────────────────────────
    with onglet_ener:
        st.header("Energy Consumption Waste Indicator (ECMI)")
        tot_ac = df_traite['AC_consumption'].sum()
        tot_waste = df_traite['ECMI_waste'].sum()
        perte = (tot_waste / tot_ac) * 100 if tot_ac > 0 else 0
        
        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Consommation Totale Climatisation", f"{tot_ac:,.1f} kWh")
        col_e2.metric("Gaspillage Énergétique Global (ECMI)", f"{tot_waste:,.1f} kWh", f"{perte:.1f}% de perte", delta_color="inverse")
        
        fig_e = px.area(df_visu, y='ECMI_waste', title="Profil temporel de gaspillage (Sur-climatisation)", color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_e, use_container_width=True)

    # ── 6. BOUTON DE TÉLÉCHARGEMENT PDF (SIDEBAR DÉPLOYABLE) ──────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Exportation du Rapport")
    
    try:
        # Génération immédiate du flux binaire textuel
        pdf_raw = generate_pdf(df_traite, annee_sel, etage_sel)
        
        st.sidebar.download_button(
            label="📥 Télécharger le Rapport PDF", 
            data=bytes(pdf_raw), 
            file_name=f"Rapport_Performance_{annee_sel}_Floor{etage_sel}.pdf", 
            mime="application/pdf", 
            use_container_width=True
        )
    except Exception as e:
        st.sidebar.error(f"Erreur lors du calcul du PDF : {e}")