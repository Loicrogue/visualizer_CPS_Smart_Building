"""
dst_engine.py — Moteur Dempster-Shafer pour le confort thermique/hygrométrique.

Réécriture fidèle des Blocs 0 (BBA), 1 (fiabilité alpha) et 2 (calibration des
masses) du notebook FUTUREV0. Contrairement à la version précédente, ce module
n'est plus du code mort : il est importé et utilisé par app.py pour calculer
BetP_comfort et K_conflict via une vraie fusion de Dempster, capteur par
capteur, plutôt qu'une moyenne/écart-type numpy.

Différences volontaires par rapport au notebook, actées avec l'utilisateur :
  - Calibration (alpha, seuils, m_n_max) faite GLOBALEMENT sur tous les
    fichiers CSV disponibles (toutes années, tous étages), pas seulement
    sur 2019 comme dans le notebook.
  - Le ffill/bfill de app.py est conservé EN AMONT du calcul des masses
    (donc le cas NaN -> m(disconfort)=1 de bba_depuis_mesure ne se
    déclenche plus qu'aux tout premiers instants d'une série, avant le
    premier bfill).

Deux incohérences trouvées dans le notebook lui-même sont tranchées ici
explicitement (voir constantes ci-dessous) :
  - THETA_CONFORT : le notebook définit la constante à 0.5 mais tout le
    texte, les tableaux de sensibilité et les graphiques parlent de 0.45
    et s'appuient dessus ("theta=0.45 retenu"). On retient 0.45.
  - Bornes hautes d'humidité confort : le commentaire cite [40,60]%
    (ASHRAE 62.1) mais la constante réellement utilisée dans le code est
    C_HAUT_H = 70.0. On garde 70.0 (comportement réellement exécuté par
    le notebook), à changer en 60.0 si tu préfères suivre le commentaire.
"""

import re
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import STL
from scipy.stats import spearmanr


# ════════════════════════════════════════════════════════════════
# CONSTANTES
# ════════════════════════════════════════════════════════════════
H_BINAIRE = ["comfort", "disconfort"]

DELTA_MILD = 2.0          # °C/%RH — Wong & Khoo (2003), zone de transition
THETA_CONFORT = 0.45       # ISO 17772-1 §6.2 — voir note d'incohérence ci-dessus
CALIBRATION_MOIS = [1, 2, 3, 4, 5, 6]

C_BAS_T, C_HAUT_T = 22.0, 27.0    # ASHRAE 55 Fig.5.3.1 + Sikram (2020)
C_BAS_H, C_HAUT_H = 40.0, 70.0    # Arundel (1986) + ASHRAE 62.1 — voir note ci-dessus

# Numéro de zone traité comme "aveugle" (Bloc 4). Le notebook ne teste
# cette méthodologie que sur z3 pour le Floor 3 (KEY=(2019,3)) ; on
# généralise ici en appliquant le même numéro de zone à tous les étages
# de l'app, à défaut d'indication contraire.
ZONE_AVEUGLE_DEFAUT = 3


def get_col_types(df):
    return {
        "Énergie (kW)": [c for c in df.columns if "kW" in c],
        "Température": [c for c in df.columns if "degC" in c],
        "Humidité": [c for c in df.columns if "RH%" in c],
        "Luminosité": [c for c in df.columns if "lux" in c],
    }


# ════════════════════════════════════════════════════════════════
# BLOC 0 — Moteur BBA
# ════════════════════════════════════════════════════════════════
class BBA:
    """
    Basic Belief Assignment — moteur DST.
    Implémente : discounting, fusion Dempster, règle disjonctive, BetP, WoE.
    """

    def __init__(self, hypotheses):
        self.H = list(hypotheses)
        self.OMEGA = frozenset(hypotheses)
        self.m = {}

    def set(self, A, val):
        key = frozenset([A]) if isinstance(A, str) else frozenset(A)
        self.m[key] = float(val)
        return self

    def set_omega(self, val):
        self.m[self.OMEGA] = float(val)
        return self

    def bel(self, A):
        Af = frozenset([A]) if isinstance(A, str) else frozenset(A)
        return sum(v for k, v in self.m.items() if k and k.issubset(Af))

    def pl(self, A):
        Af = frozenset([A]) if isinstance(A, str) else frozenset(A)
        return sum(v for k, v in self.m.items() if k & Af)

    def omega_mass(self):
        return self.m.get(self.OMEGA, 0.0)

    def incertitude(self, A):
        return self.pl(A) - self.bel(A)

    def discount(self, alpha):
        """m^alpha(A) = alpha*m(A) pour A != Omega ; le reste va dans m(Omega)."""
        r = BBA(self.H)
        for A, mA in self.m.items():
            if A == self.OMEGA:
                r.m[A] = r.m.get(A, 0.0) + mA * alpha
            else:
                r.m[A] = mA * alpha
        r.m[self.OMEGA] = r.m.get(self.OMEGA, 0.0) + (1.0 - alpha)
        return r

    def conjunctive_raw(self, other):
        """Produit conjonctif brut — K = masse sur l'ensemble vide = conflit."""
        r, K = BBA(self.H), 0.0
        for A, mA in self.m.items():
            for B, mB in other.m.items():
                inter = A & B
                if not inter:
                    K += mA * mB
                else:
                    r.m[inter] = r.m.get(inter, 0.0) + mA * mB
        return r, K

    def dempster(self, other):
        """Règle de Dempster : normalise le conflit K."""
        _, K = self.conjunctive_raw(other)
        if K >= 1.0:
            r = BBA(self.H)
            return r.set_omega(1.0), K
        r = BBA(self.H)
        fac = 1.0 - K
        for A, mA in self.m.items():
            for B, mB in other.m.items():
                inter = A & B
                if inter:
                    r.m[inter] = r.m.get(inter, 0.0) + (mA * mB) / fac
        return r, K

    def disjunctive(self, other):
        r = BBA(self.H)
        for A, mA in self.m.items():
            for B, mB in other.m.items():
                union = A | B
                r.m[union] = r.m.get(union, 0.0) + mA * mB
        return r

    def woe(self, h):
        pl_h = np.clip(self.pl(h), 1e-10, 1 - 1e-10)
        bel_h = np.clip(self.bel(h), 0.0, 1 - 1e-10)
        return float(np.log(pl_h / max(1.0 - bel_h, 1e-10)))

    def betp(self, h=None):
        scores = {hi: 0.0 for hi in self.H}
        for A, mA in self.m.items():
            if not A:
                continue
            for hi in A:
                if hi in scores:
                    scores[hi] += mA / len(A)
        return scores.get(h, 0.0) if h else scores

    def __repr__(self):
        lines = ["BBA:"]
        for k, v in sorted(self.m.items(), key=lambda x: -x[1]):
            if v > 1e-5:
                label = "{" + ",".join(sorted(k)) + "}" if k else "∅"
                lines.append(f"  m({label}) = {v:.4f}")
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# BLOC 1 — Facteur alpha de fiabilité des capteurs
# ════════════════════════════════════════════════════════════════
def calc_dispo(serie):
    """Disponibilité = fraction de mesures valides (non-NaN)."""
    return round(float(1 - serie.isnull().mean()), 4)


def calc_s_temp(serie):
    """Stabilité temporelle via décomposition STL (période 24h)."""
    s = serie.resample("h").mean().interpolate(limit=6).dropna()
    if len(s) < 72 or s.std() < 1e-6:
        return 0.5
    try:
        resid = STL(s, period=24, robust=True).fit().resid
        return float(np.clip(1 - resid.std() / s.std(), 0, 1))
    except Exception:
        return 0.5


def calc_s_spat(df, col):
    """Cohérence spatiale : |corrélation Spearman| avec capteurs du même type."""
    ct = get_col_types(df)
    type_col = next((t for t, cols in ct.items() if col in cols), None)
    if not type_col:
        return 0.5
    voisins = [c for c in ct[type_col] if c != col]
    if not voisins:
        return 0.5
    dfh = df[[col] + voisins].resample("h").mean().dropna(how="all")
    corrs = []
    for v in voisins:
        sub = dfh[[col, v]].dropna()
        if len(sub) > 50:
            r, _ = spearmanr(sub[col], sub[v])
            if not np.isnan(r):
                corrs.append(abs(r))
    return round(float(np.mean(corrs)), 4) if corrs else 0.5


def calc_alpha(df, col, w1=0.4, w2=0.3, w3=0.3):
    """alpha = sd^w1 * st^w2 * ss^w3. Retourne (alpha, sd, st, ss)."""
    sd = calc_dispo(df[col])
    st_ = calc_s_temp(df[col])
    ss = calc_s_spat(df, col)
    alpha = round(float((sd**w1) * (st_**w2) * (ss**w3)), 4)
    return alpha, sd, st_, ss


def calc_alpha_table(dataframes_par_etage, w1=0.4, w2=0.3, w3=0.3):
    """
    dataframes_par_etage : dict {etage: df} — un df par étage, éventuellement
    déjà concaténé sur plusieurs années (calibration globale, cf. app.py).
    Retourne un DataFrame long : floor, capteur, type, sd, st, ss, alpha.
    """
    rows = []
    for floor, df in dataframes_par_etage.items():
        ct = get_col_types(df)
        for col in ct["Température"] + ct["Humidité"]:
            a, sd, st_, ss = calc_alpha(df, col, w1, w2, w3)
            rows.append(
                {
                    "floor": floor,
                    "capteur": col,
                    "type": "Temp" if "degC" in col else "Hum",
                    "sd": sd,
                    "st": st_,
                    "ss": ss,
                    "alpha": a,
                }
            )
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# BLOC 2 — Calibration des masses BBA (Omega binaire)
# ════════════════════════════════════════════════════════════════
def calibrer_seuils(dataframes_par_etage, floors=None):
    """
    Percentiles P5/P50/P95 par étage sur la période de calibration
    (Jan-Juin, toutes années disponibles confondues dans le df fourni),
    + un jeu de seuils 'global' agrégeant tous les étages.
    """
    if floors is None:
        floors = list(dataframes_par_etage.keys())

    seuils, all_temp, all_hum = {}, [], []
    for floor in floors:
        if floor not in dataframes_par_etage:
            continue
        df = dataframes_par_etage[floor]
        df_cal = df[df.index.month.isin(CALIBRATION_MOIS)]
        ct = get_col_types(df_cal)
        tv = df_cal[ct["Température"]].values.flatten() if ct["Température"] else np.array([])
        hv = df_cal[ct["Humidité"]].values.flatten() if ct["Humidité"] else np.array([])
        tv = tv[~np.isnan(tv)]
        hv = hv[~np.isnan(hv)]
        all_temp.extend(tv)
        all_hum.extend(hv)
        seuils[floor] = {
            "Temp": (
                (np.percentile(tv, 5), np.percentile(tv, 50), np.percentile(tv, 95))
                if len(tv) > 0
                else (C_BAS_T, 23.0, C_HAUT_T)
            ),
            "Hum": (
                (np.percentile(hv, 5), np.percentile(hv, 50), np.percentile(hv, 95))
                if len(hv) > 0
                else (C_BAS_H, 50.0, C_HAUT_H)
            ),
        }

    seuils["global"] = {
        "Temp": (
            (np.percentile(all_temp, 5), np.percentile(all_temp, 50), np.percentile(all_temp, 95))
            if all_temp
            else (C_BAS_T, 23.0, C_HAUT_T)
        ),
        "Hum": (
            (np.percentile(all_hum, 5), np.percentile(all_hum, 50), np.percentile(all_hum, 95))
            if all_hum
            else (C_BAS_H, 50.0, C_HAUT_H)
        ),
    }
    return seuils


def calibrer_params_masses(dataframes_par_etage, seuils, floors=None):
    """
    Calibre m_n_max_T et m_n_max_H (proportion d'heures en confort sur la
    période Jan-Juin), globalement sur tous les étages/années fournis.
    """
    if floors is None:
        floors = list(dataframes_par_etage.keys())

    n_total_T, n_comfort_T = 0, 0
    n_total_H, n_comfort_H = 0, 0

    for floor in floors:
        if floor not in dataframes_par_etage:
            continue
        df = dataframes_par_etage[floor]
        df_cal = df[df.index.month.isin(CALIBRATION_MOIS)]
        ct = get_col_types(df_cal)

        for col in ct["Température"]:
            serie = df_cal[col].dropna()
            n_total_T += len(serie)
            n_comfort_T += int(((serie >= C_BAS_T) & (serie <= C_HAUT_T)).sum())

        for col in ct["Humidité"]:
            serie = df_cal[col].dropna()
            n_total_H += len(serie)
            n_comfort_H += int(((serie >= C_BAS_H) & (serie <= C_HAUT_H)).sum())

    prop_T = n_comfort_T / n_total_T if n_total_T > 0 else 0.72
    prop_H = n_comfort_H / n_total_H if n_total_H > 0 else 0.55
    m_n_max_T = float(np.clip(prop_T, 0.50, 0.88))
    m_n_max_H = float(np.clip(prop_H, 0.50, 0.88))
    pente_T = m_n_max_T - 0.08
    pente_H = m_n_max_H - 0.08

    return {
        "m_n_max": m_n_max_T,
        "pente": pente_T,
        "m_n_max_T": m_n_max_T,
        "pente_T": pente_T,
        "m_n_max_H": m_n_max_H,
        "pente_H": pente_H,
    }


def bba_depuis_mesure(val, signal_type, floor, seuils, params):
    """
    BBA sur Omega = {comfort, disconfort}.
    NaN -> m(disconfort)=1 (défaillance capteur, pas ignorance).
    CAS 1/2/3 selon la distance à la zone de confort — cf. docstring notebook.
    """
    m = BBA(H_BINAIRE)

    if pd.isna(val):
        # NaN = défaillance capteur -> m(disconfort)=1.0, PAS m(Omega)=1.0.
        # Le notebook documente ce choix (éviter BetP(C)=0.5 > theta ->
        # "confort" pour un capteur en panne) mais son code exécuté faisait
        # l'inverse (m.set_omega(1.0)) : corrigé ici pour respecter
        # l'intention documentée.
        m.set("comfort", 0.0)
        m.set("disconfort", 1.0)
        m.set_omega(0.0)
        return m

    if signal_type == "Temp":
        m_n_max_sig = params.get("m_n_max_T", params["m_n_max"])
        pente_sig = params.get("pente_T", params["pente"])
        C_BAS, C_HAUT = C_BAS_T, C_HAUT_T
    else:
        m_n_max_sig = params.get("m_n_max_H", params["m_n_max"])
        pente_sig = params.get("pente_H", params["pente"])
        C_BAS, C_HAUT = C_BAS_H, C_HAUT_H

    s = seuils.get(floor, seuils.get("global", {}))
    if signal_type not in s:
        m.set("comfort", 0.5)
        m.set("disconfort", 0.5)
        return m

    p5, _, p95 = s[signal_type]
    d_range = max(p95 - p5, 1e-6)

    if signal_type == "Temp":
        c_bas = max(p5, C_BAS)
        c_haut = min(p95, C_HAUT)
    else:
        c_bas = C_BAS
        c_haut = min(p95, C_HAUT)

    d_c = max(max(0.0, val - c_haut), max(0.0, c_bas - val))

    if d_c == 0.0:
        m_c = m_n_max_sig
        m_d = 0.08
        m_o = max(1.0 - m_c - m_d, 0.06)
    elif d_c <= DELTA_MILD:
        t = d_c / DELTA_MILD
        m_c = max(m_n_max_sig - pente_sig * t, 0.08)
        m_d = min(0.08 + pente_sig * t, 0.80)
        m_o = max(1.0 - m_c - m_d, 0.06)
    else:
        d_norm = min((d_c - DELTA_MILD) / max(d_range * 0.3, 1e-6), 1.0)
        m_c = 0.08 + (0.04 - 0.08) * d_norm
        m_d = m_n_max_sig + (0.88 - m_n_max_sig) * d_norm
        m_o = max(1.0 - m_c - m_d, 0.06)

    total = m_c + m_d + m_o
    m.set("comfort", m_c / total)
    m.set("disconfort", m_d / total)
    m.set_omega(m_o / total)
    return m


def betp_comfort(bba):
    """BetP(C) = m(C) + m(Omega)/2  [Smets 1994, |Omega|=2]."""
    return bba.bel("comfort") + bba.omega_mass() / 2.0


def entropie_pignistique(bba):
    bc = betp_comfort(bba)
    bd = 1.0 - bc
    hp = 0.0
    for p in [bc, bd]:
        if p > 1e-12:
            hp -= p * np.log2(p)
    return float(hp)


def decision_depuis_betp(bba_ou_val, theta=THETA_CONFORT):
    """BetP(C) > theta -> 'confort', sinon 'disconfort' (strict, précaution)."""
    if isinstance(bba_ou_val, BBA):
        val = betp_comfort(bba_ou_val)
    else:
        val = float(bba_ou_val)
    return "confort" if val > theta else "disconfort"


# ════════════════════════════════════════════════════════════════
# BLOC 3 — Fusion Dempster intra-étage
# ════════════════════════════════════════════════════════════════
def _fusionner_groupe(cols, sig, row, floor, df_alpha, seuils, params):
    """
    Fusionne séquentiellement tous les capteurs d'un même type pour UNE ligne
    (une mesure/timestamp). row : pandas.Series indexée par nom de colonne.
    """
    bba_fus = None
    K_values = []
    n_val = 0
    df_alpha_floor = df_alpha[df_alpha["floor"] == floor] if df_alpha is not None else None

    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        alp = 0.50
        if df_alpha_floor is not None:
            r = df_alpha_floor[df_alpha_floor["capteur"] == col]
            if len(r) > 0:
                alp = float(r["alpha"].iloc[0])
        bba_c = bba_depuis_mesure(val, sig, floor, seuils, params).discount(alp)
        if not pd.isna(val):
            n_val += 1
        if bba_fus is None:
            bba_fus = bba_c
        else:
            _, K_reel = bba_fus.conjunctive_raw(bba_c)
            K_values.append(K_reel)
            bba_fus, _ = bba_fus.dempster(bba_c)

    K_groupe = float(np.mean(K_values)) if K_values else 0.0
    return bba_fus, K_groupe, n_val


def _betp_binaire(bba):
    m_c = bba.bel("comfort")
    m_d = bba.bel("disconfort")
    m_o = bba.omega_mass()
    return m_c + m_o / 2.0, m_d + m_o / 2.0, m_c, m_d, m_o


def fusionner_etage_dst_row(row, floor, df_alpha, seuils, params, cols_temp, cols_hum, theta=THETA_CONFORT):
    """
    Fusion structurée en 3 étapes pour une ligne (un timestamp) :
      1. Dempster intra-Temp -> BBA_temp
      2. Dempster intra-Hum  -> BBA_hum
      3. Dempster(BBA_temp, BBA_hum) -> BBA_étage
    """
    if not cols_temp:
        return None

    bba_temp, K_temp, n_temp = _fusionner_groupe(cols_temp, "Temp", row, floor, df_alpha, seuils, params)
    if bba_temp is None:
        return None

    bba_hum, K_hum, n_hum = None, 0.0, 0
    if cols_hum:
        bba_hum, K_hum, n_hum = _fusionner_groupe(cols_hum, "Hum", row, floor, df_alpha, seuils, params)

    if bba_hum is not None:
        _, K_cross = bba_temp.conjunctive_raw(bba_hum)
        bba_etage, _ = bba_temp.dempster(bba_hum)
        K_global = float(np.mean([v for v in [K_temp, K_hum, K_cross] if v > 0] or [K_cross]))
    else:
        bba_etage = bba_temp
        K_global = K_temp

    n_val = n_temp + n_hum
    betp_c, betp_d, m_c, m_d, m_o = _betp_binaire(bba_etage)
    decision = "confort" if betp_c > theta else "disconfort"

    return {
        "Bel_comfort": bba_etage.bel("comfort"),
        "Pl_comfort": bba_etage.pl("comfort"),
        "BetP_comfort": betp_c,
        "BetP_disconfort": betp_d,
        "m_comfort": m_c,
        "m_disconfort": m_d,
        "m_Omega": m_o,
        "K_conflict": K_global,
        "K_temp": K_temp,
        "K_hum": K_hum,
        "DST_decision": decision,
        "n_capteurs": n_val,
    }


def fusionner_etage_dst_df(df_floor_h, floor, df_alpha, seuils, params, theta=THETA_CONFORT):
    """
    Applique fusionner_etage_dst_row à chaque ligne d'un DataFrame déjà
    ré-échantillonné (typiquement à l'heure — cf. note de performance dans
    app.py). Retourne un DataFrame indexé comme df_floor_h.
    """
    ct = get_col_types(df_floor_h)
    cols_temp = ct.get("Température", [])
    cols_hum = ct.get("Humidité", [])

    records = []
    index_valides = []
    for ts, row in df_floor_h.iterrows():
        res = fusionner_etage_dst_row(row, floor, df_alpha, seuils, params, cols_temp, cols_hum, theta)
        if res is not None:
            records.append(res)
            index_valides.append(ts)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records, index=pd.Index(index_valides, name=df_floor_h.index.name))


# ════════════════════════════════════════════════════════════════
# BLOC 4 — Zone aveugle (IDW) + fusion avec l'étage instrumenté
# ════════════════════════════════════════════════════════════════
def extraire_num_zone(col_name):
    m = re.search(r"z(\d+)", col_name)
    return int(m.group(1)) if m else None


def idw_avec_fallback(df_floor_h, voisins_cols, zone_cible_num, timestamp):
    """
    IDW (Shepard 1968) : w_i = 1/dist_i^2, dist = |zone_i - zone_cible|.
    Fallback temporel ffill/bfill (limit=2) avant calcul.
    """
    if timestamp not in df_floor_h.index:
        return np.nan
    df_interp = df_floor_h[voisins_cols].ffill(limit=2).bfill(limit=2)
    vals, poids = [], []
    for col in voisins_cols:
        if col not in df_interp.columns:
            continue
        v = df_interp.loc[timestamp, col] if timestamp in df_interp.index else np.nan
        if pd.isna(v):
            continue
        num = extraire_num_zone(col)
        dist = max(abs(num - zone_cible_num) if num is not None else 2, 0.5)
        poids.append(1.0 / dist**2)
        vals.append(v)
    if not vals:
        return np.nan
    p = np.array(poids)
    return float(np.sum(p * np.array(vals)) / np.sum(p))


def estimer_zone_aveugle_df(df_floor_h, floor, zone_num, seuils, params, timestamps=None):
    """
    Estimation IDW de la zone aveugle `zone_num`, à partir des CAPTEURS
    VOISINS uniquement (zone_num est explicitement exclu de ses propres
    voisins). Construit BBA_zone_aveugle = Dempster(BBA_pseudo_temp,
    BBA_pseudo_hum), SANS discounting.

    Choix documenté du notebook (conservé ici) : appliquer un alpha à une
    zone qui, par définition, n'a pas de capteur direct n'a pas de sens.
    L'incertitude de l'estimation IDW est déjà encodée dans m(Omega) via
    bba_depuis_mesure : une pseudo-mesure proche de la borne de confort
    génère naturellement un m(Omega) plus grand.
    """
    if timestamps is None:
        timestamps = df_floor_h.index

    ct = get_col_types(df_floor_h)
    voisins_t = [c for c in ct["Température"] if extraire_num_zone(c) != zone_num]
    voisins_h = [c for c in ct["Humidité"] if extraire_num_zone(c) != zone_num]
    if not voisins_t:
        return pd.DataFrame()

    records = []
    index_valides = []
    for ts in timestamps:
        pseudo_t = idw_avec_fallback(df_floor_h, voisins_t, zone_num, ts)
        pseudo_h = idw_avec_fallback(df_floor_h, voisins_h, zone_num, ts) if voisins_h else np.nan

        bba_t = bba_depuis_mesure(pseudo_t, "Temp", floor, seuils, params)
        if voisins_h and not pd.isna(pseudo_h):
            bba_h = bba_depuis_mesure(pseudo_h, "Hum", floor, seuils, params)
            bba_za, K_za = bba_t.dempster(bba_h)
        else:
            bba_za, K_za = bba_t, 0.0

        m_c = bba_za.bel("comfort")
        m_d = bba_za.bel("disconfort")
        m_o = bba_za.omega_mass()
        betp_c = m_c + m_o / 2.0

        records.append(
            {
                "pseudo_temp": pseudo_t,
                "pseudo_hum": pseudo_h,
                "K_zone_aveugle": K_za,
                "Bel_comfort_za": bba_za.bel("comfort"),
                "Pl_comfort_za": bba_za.pl("comfort"),
                "BetP_comfort_za": betp_c,
                "m_comfort_za": m_c,
                "m_disconfort_za": m_d,
                "m_Omega_za": m_o,
                "decision_za": "confort" if betp_c > THETA_CONFORT else "disconfort",
                "_bba_za": bba_za,
            }
        )
        index_valides.append(ts)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records, index=pd.Index(index_valides, name=df_floor_h.index.name))


def fusionner_etage_complet(df_dst_h, df_zone_aveugle, theta=THETA_CONFORT):
    """
    Étape finale (Bloc 4) : Dempster(BBA_étage_instrumenté, BBA_zone_aveugle)
    -> décision étage COMPLET. BBA_étage_instrumenté est reconstruite depuis
    les colonnes BetP_comfort / BetP_disconfort / m_Omega de df_dst_h
    (sortie de fusionner_etage_dst_df, Bloc 3), par inversion de Smets 1994 :
    m(comfort) = BetP(comfort) - m(Omega)/2.
    """
    ts_commun = df_dst_h.index.intersection(df_zone_aveugle.index)
    records = []
    for ts in ts_commun:
        row_f = df_dst_h.loc[ts]
        bba_za = df_zone_aveugle.loc[ts, "_bba_za"]

        m_o8 = float(row_f["m_Omega"])
        m_c8 = max(0.0, float(row_f["BetP_comfort"]) - m_o8 / 2.0)
        m_d8 = max(0.0, float(row_f["BetP_disconfort"]) - m_o8 / 2.0)

        bba_instrumente = BBA(H_BINAIRE)
        bba_instrumente.set("comfort", m_c8)
        bba_instrumente.set("disconfort", m_d8)
        bba_instrumente.set_omega(m_o8)

        bba_etage, K_etage = bba_instrumente.dempster(bba_za)
        m_c_et = bba_etage.bel("comfort")
        m_o_et = bba_etage.omega_mass()
        betp_c_et = m_c_et + m_o_et / 2.0

        records.append(
            {
                "timestamp": ts,
                "BetP_comfort_etage": betp_c_et,
                "Bel_comfort_etage": bba_etage.bel("comfort"),
                "Pl_comfort_etage": bba_etage.pl("comfort"),
                "m_comfort_etage": m_c_et,
                "m_disconfort_etage": bba_etage.bel("disconfort"),
                "m_Omega_etage": m_o_et,
                "K_etage": K_etage,
                "DST_decision_etage": "confort" if betp_c_et > theta else "disconfort",
            }
        )

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).set_index("timestamp")
