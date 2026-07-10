import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from pathlib import Path
from sklearn.ensemble import IsolationForest

from rapport import generate_pdf
import dst_engine as dst

st.set_page_config(page_title="CPS - Visualisation Données", layout="wide", page_icon="🏢")
st.title("🏢 Cyber-Physical System (CPS) - Smart Building")

DATA_DIR = Path("data/raw")
ANNEES_DISPONIBLES = [2018, 2019]
ETAGES_DISPONIBLES = [3, 4, 5, 6, 7]


# ── 1. CHARGEMENT BRUT + NETTOYAGE DES DATES ──────────────────────────
@st.cache_data(show_spinner=False)
def charger_csv_brut(chemin_fichier):
    """
    Charge un CSV, détecte et retire les lignes dont la Date est
    corrompue. Renvoie (df indexé par Date, liste d'alertes).
    Le ffill/bfill est appliqué ici (comportement conservé tel quel).
    """
    df = pd.read_csv(chemin_fichier)
    nom_fichier = Path(chemin_fichier).name

    dates_converties = pd.to_datetime(df["Date"], errors="coerce")
    lignes_corrompues = df[dates_converties.isna()]

    alertes = []
    if not lignes_corrompues.empty:
        for idx, row in lignes_corrompues.iterrows():
            numero_ligne_fichier = idx + 2
            alertes.append((nom_fichier, numero_ligne_fichier, row["Date"]))
        df = df[dates_converties.notna()]
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        df["Date"] = dates_converties

    df = df.set_index("Date")
    df = df.ffill().bfill()
    return df, alertes


# ── 2. CALIBRATION GLOBALE DST (alpha, seuils, m_n_max) ───────────────
@st.cache_resource(show_spinner="Calibration DST globale (alpha, seuils, m_n_max) — peut prendre plusieurs minutes la première fois...")
def calibration_globale_dst():
    """
    Calcule alpha (fiabilité), les seuils P5/P50/P95 et les paramètres
    de masse (m_n_max_T/H) sur TOUS les fichiers CSV disponibles dans
    data/raw/, toutes années confondues, par étage.

    Note : le notebook calibre uniquement sur l'année 2019. Ici, on
    calibre sur toutes les années disponibles pour un étage donné (2018 +
    2019 si les deux fichiers existent), conformément au choix de
    calibration globale sur tous les fichiers.

    Mis en cache_resource : ne se relance pas à chaque interaction, mais
    UNE SEULE FOIS pour la session (STL + Spearman sur toutes les séries
    est coûteux).
    """
    dataframes_par_etage = {}
    toutes_alertes = []

    for etage in ETAGES_DISPONIBLES:
        dfs_etage = []
        for annee in ANNEES_DISPONIBLES:
            chemin = DATA_DIR / f"{annee}Floor{etage}.csv"
            if chemin.exists():
                df, alertes = charger_csv_brut(str(chemin))
                dfs_etage.append(df)
                toutes_alertes.extend(alertes)
        if dfs_etage:
            df_combine = pd.concat(dfs_etage).sort_index()
            dataframes_par_etage[etage] = df_combine

    if not dataframes_par_etage:
        return None, None, None, toutes_alertes

    df_alpha = dst.calc_alpha_table(dataframes_par_etage)
    seuils = dst.calibrer_seuils(dataframes_par_etage)
    params = dst.calibrer_params_masses(dataframes_par_etage, seuils)
    return df_alpha, seuils, params, toutes_alertes


# ── 3. TRAITEMENT D'UN ÉTAGE/ANNÉE : FUSION DST + ECMI ────────────────
@st.cache_data(show_spinner="Fusion Dempster en cours (par heure)...")
def traiter_donnees_cps(chemin_fichier, etage, _df_alpha, _seuils, _params, zone_aveugle):
    """
    Pipeline complet pour un fichier (année, étage) donné :
      1. Charge et nettoie (ffill/bfill déjà appliqué par charger_csv_brut)
      2. Calcule AC_consumption / ECMI_waste à résolution native (minute)
      3. Ré-échantillonne les capteurs Temp/Hum à l'heure et lance la
         VRAIE fusion Dempster (dst_engine.fusionner_etage_dst_df) —
         c'est ici que le bug de fond (moyenne numpy au lieu de fusion
         Dempster + absence de pondération par fiabilité) est corrigé.
      4. Recale l'IsolationForest sur les données horaires pour une
         comparaison DST vs IF à la même résolution temporelle.

    Note de performance/résolution : la fusion Dempster capteur-par-capteur
    (avec pondération alpha) est appliquée à l'heure, comme dans le
    notebook (Bloc 3 fusionne toujours sur du df.resample('H')). L'appliquer
    à la minute impliquerait ~525 000 fusions Dempster par étage/année,
    ce qui n'est pas praticable dans une app interactive. AC_consumption
    et ECMI_waste, eux, restent calculés à la minute (résolution native)
    car ils ne dépendent pas de la fusion DST.
    """
    df, _ = charger_csv_brut(chemin_fichier)
    colonnes = df.columns
    zones = sorted(list(set([re.match(r"(z\d+)_", col).group(1) for col in colonnes if re.match(r"(z\d+)_", col)])))

    ct = dst.get_col_types(df)
    cols_t = ct["Température"]
    cols_rh = ct["Humidité"]

    if not cols_t:
        df_h = df.resample("h").mean(numeric_only=True)
        df_h["BetP_comfort"] = 0.5
        df_h["K_conflict"] = 0.0
        df_h["ECMI_waste"] = 0.0
        df_h["AC_consumption"] = 0.0
        df_h["DST_decision"] = "Confort"
        df_h["IF_decision"] = "Confort"
        df_h["Recommandation_4h_consigne"] = "Confort"
        df_h["Moyenne_RH"] = 50.0
        df_h["Moyenne_T"] = 23.0
        return df_h

    # --- CONSOMMATION AC & ECMI (résolution native, minute) ------------
    ac_consumption = pd.Series(0.0, index=df.index)
    ecmi_waste = pd.Series(0.0, index=df.index)
    for zone in zones:
        cols_ac_zone = [col for col in colonnes if col.startswith(zone) and "(kW)" in col and "AC" in col]
        col_t_zone = f"{zone}_S1(degC)"
        if cols_ac_zone:
            conso_ac_zone = df[cols_ac_zone].sum(axis=1)
            ac_consumption += conso_ac_zone * (1.0 / 60.0)
            if col_t_zone in df.columns:
                t_zone = df[col_t_zone].values
                gaspillage_zone = np.where(
                    (t_zone >= 21.5) & (t_zone <= 24.5) & (conso_ac_zone > 0.05), conso_ac_zone, 0.0
                )
                ecmi_waste += gaspillage_zone * (1.0 / 60.0)

    df_energie_h = pd.DataFrame({"AC_consumption": ac_consumption, "ECMI_waste": ecmi_waste}).resample("h").sum()

    # --- FUSION DST — Blocs 0/1/2/3, résolution horaire ----------------
    df_floor_h = df.resample("h").mean(numeric_only=True)

    if _df_alpha is not None and not _df_alpha.empty:
        df_dst_h = dst.fusionner_etage_dst_df(df_floor_h, etage, _df_alpha, _seuils, _params)
    else:
        df_dst_h = pd.DataFrame()

    if df_dst_h.empty:
        # Pas de calibration disponible : dashboard vide mais fonctionnel.
        df_dst_h = pd.DataFrame(index=df_floor_h.index)
        df_dst_h["BetP_comfort"] = 0.5
        df_dst_h["K_conflict"] = 0.0
        df_dst_h["DST_decision"] = "disconfort"
        df_dst_h["BetP_disconfort"] = 0.5
        df_dst_h["m_Omega"] = 1.0

    # --- BLOC 4 — Zone aveugle (IDW) + fusion étage complet -------------
    # Résultat final "comme dans le notebook" : Dempster(BBA_8capteurs,
    # BBA_zone_aveugle). C'est cette étape qui fait monter BetP(C) par
    # rapport à la fusion Bloc 3 seule (renforcement Dempster quand les
    # deux sources sont d'accord).
    df_dst_instrumente = df_dst_h.copy()  # conservé pour comparaison dans l'UI

    if not df_dst_h.empty and zone_aveugle is not None:
        df_za = dst.estimer_zone_aveugle_df(df_floor_h, etage, zone_aveugle, _seuils, _params)
        if not df_za.empty:
            df_etage_complet = dst.fusionner_etage_complet(df_dst_h, df_za)
            if not df_etage_complet.empty:
                # Le résultat étage complet devient la source de vérité.
                df_dst_h = df_dst_h.join(
                    df_etage_complet[["BetP_comfort_etage", "m_Omega_etage", "K_etage", "DST_decision_etage"]],
                    how="left",
                )
                df_dst_h["BetP_comfort"] = df_dst_h["BetP_comfort_etage"].fillna(df_dst_h["BetP_comfort"])
                df_dst_h["K_conflict"] = df_dst_h["K_etage"].fillna(df_dst_h["K_conflict"])
                df_dst_h["DST_decision"] = df_dst_h["DST_decision_etage"].fillna(df_dst_h["DST_decision"])
                df_dst_h["m_Omega"] = df_dst_h["m_Omega_etage"].fillna(df_dst_h["m_Omega"])
            df_dst_h["BetP_comfort_zone_aveugle"] = df_za["BetP_comfort_za"]
            df_dst_h["pseudo_temp_zone_aveugle"] = df_za["pseudo_temp"]
            df_dst_h["pseudo_hum_zone_aveugle"] = df_za["pseudo_hum"]

    # --- ISOLATION FOREST, même résolution que le DST (horaire) --------
    cols_if = cols_t + cols_rh
    df_if_input = df_floor_h[cols_if].dropna()
    if len(df_if_input) > 10:
        clf = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        if_preds = clf.fit_predict(df_if_input)
        if_decision = pd.Series(
            np.where(if_preds == -1, "Inconfort", "Confort"), index=df_if_input.index
        )
    else:
        if_decision = pd.Series("Confort", index=df_floor_h.index)

    # --- ASSEMBLAGE FINAL -------------------------------------------------
    df_traite = df_floor_h[[]].copy()
    df_traite["Moyenne_T"] = df_floor_h[cols_t].mean(axis=1)
    df_traite["Moyenne_RH"] = df_floor_h[cols_rh].mean(axis=1) if cols_rh else 50.0

    colonnes_dst = ["BetP_comfort", "K_conflict", "m_Omega", "Bel_comfort", "Pl_comfort"]
    colonnes_za = [
        c
        for c in ["BetP_comfort_zone_aveugle", "pseudo_temp_zone_aveugle", "pseudo_hum_zone_aveugle"]
        if c in df_dst_h.columns
    ]
    df_traite = df_traite.join(df_dst_h[colonnes_dst + colonnes_za], how="left")
    df_traite["BetP_comfort_instrumente"] = df_dst_instrumente["BetP_comfort"].reindex(df_traite.index)
    df_traite["DST_decision"] = df_dst_h["DST_decision"].map({"confort": "Confort", "disconfort": "Inconfort"})
    df_traite["IF_decision"] = if_decision.reindex(df_traite.index)
    df_traite = df_traite.join(df_energie_h, how="left")

    df_traite[["BetP_comfort", "K_conflict", "AC_consumption", "ECMI_waste"]] = df_traite[
        ["BetP_comfort", "K_conflict", "AC_consumption", "ECMI_waste"]
    ].fillna(0.0)
    df_traite["DST_decision"] = df_traite["DST_decision"].fillna("Inconfort")
    df_traite["IF_decision"] = df_traite["IF_decision"].fillna("Confort")

    # Anticipation +4h : à résolution horaire, un décalage de 4 lignes = 4h.
    df_traite["Recommandation_4h_consigne"] = (
        df_traite["DST_decision"].shift(-4).fillna(df_traite["DST_decision"])
    )

    return df_traite


# ── 4. INTERFACE SIDEBAR ──────────────────────────────────────────────
st.sidebar.header("📊 Configuration FUTURE")

annee_sel = st.sidebar.selectbox("📅 Sélectionner l'Année", ANNEES_DISPONIBLES)
etage_sel = st.sidebar.selectbox("🏢 Choisir l'Étage", ETAGES_DISPONIBLES)
gran_sel = st.sidebar.selectbox("📊 Granularité d'affichage", ["Horaire (h)", "Quotidien (D)", "Mensuel (M)"])
st.sidebar.caption(
    "La fusion Dempster est calculée à l'heure (comme dans le notebook) ; "
    "la granularité Minute a été retirée du dashboard car il n'existe pas "
    "de décision DST en dessous de l'heure."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧩 Zone aveugle (Bloc 4)")
inclure_zone_aveugle = st.sidebar.checkbox(
    "Intégrer la zone aveugle (IDW) au résultat final", value=True
)
zone_aveugle_sel = (
    st.sidebar.number_input(
        "Numéro de zone traitée comme aveugle",
        min_value=1, max_value=8, value=3, step=1,
    )
    if inclure_zone_aveugle
    else None
)
st.sidebar.caption(
    "Le notebook ne démontre cette méthode que sur z3/Floor 3 ; on applique "
    "ici le même numéro de zone à tous les étages par défaut. Décoche la "
    "case pour revenir à la fusion Bloc 3 seule (capteurs instrumentés)."
)

map_resample = {"Mensuel (M)": "ME", "Quotidien (D)": "D", "Horaire (h)": "h"}

chemin_final = DATA_DIR / f"{annee_sel}Floor{etage_sel}.csv"

if not chemin_final.exists():
    st.error(f"Le fichier `{chemin_final}` est introuvable. Veuillez vérifier votre dossier `data/`.")
else:
    df_alpha, seuils, params, alertes_chargement = calibration_globale_dst()

    for nom_fichier, numero_ligne, valeur in alertes_chargement:
        st.sidebar.warning(
            f"⚠️ **Donnée parasite nettoyée**\n"
            f"• Fichier : `{nom_fichier}`\n"
            f"• Ligne CSV : `Ligne {numero_ligne}`\n"
            f"• Valeur : `\"{valeur}\"`\n"
            f"👉 *Ligne ignorée automatiquement.*"
        )

    if df_alpha is None:
        st.error("Aucun fichier trouvé dans `data/raw/` pour la calibration DST globale.")
        st.stop()

    df_traite = traiter_donnees_cps(str(chemin_final), etage_sel, df_alpha, seuils, params, zone_aveugle_sel)

    df_visu = df_traite.resample(map_resample[gran_sel]).mean(numeric_only=True)

    # ── 5. CARTES DE MÉTRIQUES ─────────────────────────────────────────
    t_moyen = df_visu["Moyenne_T"].mean()
    rh_moyen = df_visu["Moyenne_RH"].mean()
    betp_moyen = df_visu["BetP_comfort"].mean() * 100
    conflit_moyen = df_visu["K_conflict"].mean()

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

    # ── 6. ANTICIPATION +4H ─────────────────────────────────────────────
    st.subheader("🔮 Anticipation de Consigne Thermique / Hygrométrique (+4h)")
    derniere_pred = df_traite["Recommandation_4h_consigne"].iloc[-1]

    if "confort" in derniere_pred.lower():
        st.success("✅ **Stable** : Confort thermique et respiratoire garanti sur les 4 prochaines heures.")
    else:
        st.error("⚠️ **Alerte Inconfort** : Une dérive d'ambiance est prévue d'ici 4 heures. Ajustement de la consigne requis.")

    # ── 7. ONGLETS ────────────────────────────────────────────────────
    onglet_dst, onglet_comp, onglet_ener, onglet_za = st.tabs(
        [
            "📈 DST & Confort Multi-variable",
            "🤖 Comparaison DST vs Isolation Forest",
            "⚡ ECMI & Gaspillage Énergétique",
            "🧩 Zone aveugle (IDW)",
        ]
    )

    # ── ONGLET 1 : DOUBLE AXE TEMPÉRATURE & HUMIDITÉ ────────────────────
    with onglet_dst:
        st.write("### Évolution temporelle : Indicateurs DST vs Environnement")

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=df_visu.index, y=df_visu["BetP_comfort"] * 100, name="Confort Global (BetP %)", line=dict(color="#2ECC71", width=2)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df_visu.index, y=df_visu["K_conflict"] * 100, name="Conflit Multi-capteurs (K %)", line=dict(color="#E74C3C", width=2)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df_visu.index, y=df_visu["Moyenne_RH"], name="Humidité Relative (RH %)", line=dict(color="#3498DB", width=1.5)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df_visu.index, y=df_visu["Moyenne_T"], name="Température Moyenne (°C)", line=dict(color="#F39C12", width=2.5, dash="dash")),
            secondary_y=True,
        )

        fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        fig.update_yaxes(
            title_text="<b>DST</b> : <span style='color:#2ECC71'>Confort</span>, <span style='color:#E74C3C'>Conflit</span> & <span style='color:#3498DB'>Humidité</span> (%)",
            secondary_y=False,
            range=[0, 105],
        )
        fig.update_yaxes(
            title_text="<b>Environnement</b> : <span style='color:#F39C12'>Température</span> (°C)",
            secondary_y=True,
            range=[18, 30],
        )

        st.plotly_chart(fig, use_container_width=True)

    # ── ONGLET 2 : COMPARAISON DES MODÈLES ──────────────────────────────
    with onglet_comp:
        st.header("Validation Croisée : Modèle Physique Avancé vs Statistique")
        accord = (df_traite["DST_decision"] == df_traite["IF_decision"]).mean() * 100
        st.metric("Taux d'accord global DST / Isolation Forest", f"{accord:.1f} %")
        df_div = df_traite[df_traite["DST_decision"] != df_traite["IF_decision"]]
        st.dataframe(df_div[["DST_decision", "IF_decision", "BetP_comfort", "K_conflict", "Moyenne_RH"]].head(50))

        fig_comp = px.line(df_visu, y=["BetP_comfort"], title="Stabilité du Critère de Croyance Pignistique (BetP)")
        st.plotly_chart(fig_comp, use_container_width=True)

    # ── ONGLET 3 : ANALYSE ÉNERGÉTIQUE (ECMI) ──────────────────────────
    with onglet_ener:
        st.header("Energy Consumption Waste Indicator (ECMI)")
        tot_ac = df_traite["AC_consumption"].sum()
        tot_waste = df_traite["ECMI_waste"].sum()
        perte = (tot_waste / tot_ac) * 100 if tot_ac > 0 else 0

        col_e1, col_e2 = st.columns(2)
        col_e1.metric("Consommation Totale Climatisation", f"{tot_ac:,.1f} kWh")
        col_e2.metric("Gaspillage Énergétique Global (ECMI)", f"{tot_waste:,.1f} kWh", f"{perte:.1f}% de perte", delta_color="inverse")

        fig_e = px.area(df_visu, y="ECMI_waste", title="Profil temporel de gaspillage (Sur-climatisation)", color_discrete_sequence=["#FF4B4B"])
        st.plotly_chart(fig_e, use_container_width=True)

    # ── ONGLET 4 : ZONE AVEUGLE (BLOC 4) ────────────────────────────────
    with onglet_za:
        st.header("Estimation IDW de la zone aveugle & fusion étage complet")
        if not inclure_zone_aveugle or "BetP_comfort_zone_aveugle" not in df_traite.columns:
            st.info(
                "Zone aveugle désactivée (case décochée dans la barre latérale) : "
                "le résultat affiché dans les autres onglets correspond à la fusion "
                "Bloc 3 seule (capteurs instrumentés uniquement)."
            )
        else:
            betp_instr_moy = df_visu["BetP_comfort_instrumente"].mean() * 100 if "BetP_comfort_instrumente" in df_visu.columns else np.nan
            betp_za_moy = df_visu["BetP_comfort_zone_aveugle"].mean() * 100
            betp_etage_moy = df_visu["BetP_comfort"].mean() * 100

            col_z1, col_z2, col_z3 = st.columns(3)
            col_z1.metric("BetP(C) — Étage instrumenté seul (Bloc 3)", f"{betp_instr_moy:.1f} %")
            col_z2.metric(f"BetP(C) — Zone aveugle z{zone_aveugle_sel} (IDW)", f"{betp_za_moy:.1f} %")
            col_z3.metric("BetP(C) — Étage COMPLET (Bloc 4, résultat final)", f"{betp_etage_moy:.1f} %")

            st.caption(
                "L'étage complet fusionne (règle de Dempster) la BBA de l'étage instrumenté "
                "avec la BBA de la zone aveugle. Quand les deux sources s'accordent, la "
                "fusion renforce la confiance et fait monter BetP(C) — c'est cet écart que "
                "tu observais entre le notebook et le dashboard avant cette intégration."
            )

            fig_za = make_subplots(specs=[[{"secondary_y": True}]])
            fig_za.add_trace(
                go.Scatter(x=df_visu.index, y=df_visu["BetP_comfort_instrumente"] * 100, name="BetP(C) instrumenté (Bloc 3)", line=dict(color="#3498DB", width=2)),
                secondary_y=False,
            )
            fig_za.add_trace(
                go.Scatter(x=df_visu.index, y=df_visu["BetP_comfort_zone_aveugle"] * 100, name=f"BetP(C) zone aveugle z{zone_aveugle_sel} (IDW)", line=dict(color="#F39C12", width=2)),
                secondary_y=False,
            )
            fig_za.add_trace(
                go.Scatter(x=df_visu.index, y=df_visu["BetP_comfort"] * 100, name="BetP(C) étage complet (final)", line=dict(color="#2ECC71", width=2.5)),
                secondary_y=False,
            )
            fig_za.add_trace(
                go.Scatter(x=df_visu.index, y=df_visu["pseudo_temp_zone_aveugle"], name=f"Pseudo-température z{zone_aveugle_sel} (IDW, °C)", line=dict(color="#95A5A6", width=1.5, dash="dot")),
                secondary_y=True,
            )
            fig_za.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_za.update_yaxes(title_text="BetP(C) (%)", secondary_y=False, range=[0, 105])
            fig_za.update_yaxes(title_text="Pseudo-température (°C)", secondary_y=True)
            st.plotly_chart(fig_za, use_container_width=True)

    # ── 8. EXPORT PDF ──────────────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Exportation du Rapport")

    try:
        pdf_raw = generate_pdf(df_traite, annee_sel, etage_sel)
        st.sidebar.download_button(
            label="📥 Télécharger le Rapport PDF",
            data=bytes(pdf_raw),
            file_name=f"Rapport_Performance_{annee_sel}_Floor{etage_sel}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as e:
        st.sidebar.error(f"Erreur lors du calcul du PDF : {e}")
