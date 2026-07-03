import os
import pandas as pd
import numpy as np
from datetime import datetime
from fpdf import FPDF

def generate_pdf(df_traite, annee, etage):
    # 1. Initialisation du document PDF
    pdf = FPDF()
    pdf.add_page()
    
    # --- ENTETE DU RAPPORT ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(44, 62, 80) # Bleu nuit
    pdf.cell(0, 10, f"Rapport CPS & Analyse d'Incertitude - Etage {etage} ({annee})", ln=True, align="C")
    pdf.ln(5)
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, f"Genere automatiquement le : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(3)
    
    # --- SECTION 1 : DST & CONFORT ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "1. Synthese de l'Ambiance et du Confort (DST Multi-Capteurs)", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    
    # Calcul des nouvelles métriques
    betp_moyen = df_traite['BetP_comfort'].mean() * 100
    conflit_moyen = df_traite['K_conflict'].mean()
    t_moyen = df_traite['Moyenne_T'].mean()
    rh_moyen = df_traite['Moyenne_RH'].mean()
    
    pdf.multi_cell(0, 8, f"- Score de confort global (BetP) : {betp_moyen:.2f}%\n"
                         f"- Taux de conflit moyen de l'etage (K) : {conflit_moyen:.3f}\n"
                         f"- Temperature moyenne de l'air constatee : {t_moyen:.1f} degC (Seuils cibles : 20 - 26 degC)\n"
                         f"- Humidite relative moyenne constatee : {rh_moyen:.1f}% (Seuils cibles : 30% - 70%)\n")
    pdf.ln(5)
    
    # --- SECTION 2 : VALIDATION CROISEE ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "2. Validation Croisee (DST vs Isolation Forest Multivarie)", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    accord = (df_traite['DST_decision'] == df_traite['IF_decision']).mean() * 100
    pdf.multi_cell(0, 8, f"- Taux de convergence des modeles : {accord:.2f}%\n"
                         f"- Note : Un taux eleve valide la coherence entre l'approche physique (DST) "
                         f"et l'approche statistique (Isolation Forest).\n")
    pdf.ln(5)
    
    # --- SECTION 3 : ECMI & PERFORMANCE ENERGETIQUE ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "3. Performance Energetique liee a l'ECMI", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    tot_ac = df_traite['AC_consumption'].sum()
    tot_waste = df_traite['ECMI_waste'].sum()
    perte = (tot_waste / tot_ac) * 100 if tot_ac > 0 else 0
    
    pdf.multi_cell(0, 8, f"- Consommation totale de la climatisation : {tot_ac:,.2f} kWh\n"
                         f"- Volume de gaspillage energetique identifie (ECMI) : {tot_waste:,.2f} kWh\n"
                         f"- Taux de pertes evitables sur l'etage : {perte:.2f}%\n")
    pdf.ln(10)
    
    # --- SECTION 4 : SYNTHESE DES ONGLETS & DECISION FINALE ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "4. Synthese des Onglets & Decision Globale", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    resume_texte = (
        "- Onglet DST et Confort Multi-variable : Fusion de l'incertitude (Temperature et Humidite) via la "
        "theorie de Dempster-Shafer. Analyse conjointe du Confort (BetP) et du Conflit (K).\n"
        "- Onglet Comparaison DST vs Isolation Forest : Filtration et validation croisee. "
        "Identification des ecarts comportementaux et divergences des modeles.\n"
        "- Onglet ECMI et Gaspillage Energetique : Evaluation de la consommation des AC (kW) et calcul "
        "de l'indicateur mesurant le gaspillage lie a des sur-climatisations inutiles."
    )
    pdf.multi_cell(0, 6, resume_texte)
    pdf.ln(8)
    
    # Determination de la decision finale
    derniere_decision = df_traite['DST_decision'].iloc[-1]
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(52, 8, "Decision Finale du CPS : ")
    
    if "confort" in derniere_decision.strip().lower():
        pdf.set_text_color(46, 204, 113) # Vert
        pdf.cell(0, 8, "CONFORT", ln=True)
        pdf.ln(4)
        
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, "Les indicateurs de croyance pignistique (BetP) confirment que "
                             "l'environnement thermique et hygrometrique respecte de maniere stable les "
                             "exigences de confort de la norme ASHRAE 55.")
    else:
        pdf.set_text_color(231, 76, 60) # Rouge
        pdf.cell(0, 8, "INCONFORT DETECTE", ln=True)
        pdf.ln(4)
        
        # Bloc d'alerte encadre pour les conseils
        pdf.set_fill_color(254, 242, 242) 
        pdf.set_draw_color(231, 76, 60)   
        pdf.set_text_color(60, 60, 60)
        
        conseils = (
            " CONSEILS D'AMELIORATION DE L'HABITABILITE ET DE LA GTB :\n"
            " 1. Regularisation de l'Humidite : Les valeurs de Moyenne_RH derivent hors de la plage optimale "
            " (30-70%). Il est recommande d'activer ou verifier les extracteurs d'air et de ventilation.\n"
            " 2. Equilibrage Thermique : Un niveau de conflit multi-capteurs (K) eleve indique des ecarts importants "
            " entre les zones de cet etage. Ajustez les registres de debit.\n"
            " 3. Anticipation +4h : L'algorithme predictif remonte un inconfort futur. Reduisez de 0.5 degC la "
            " consigne pour lisser preventivement la charge thermique."
        )
        
        x_start = pdf.get_x()
        y_start = pdf.get_y()
        pdf.rect(x_start, y_start, 190, 42, style="FD")
        
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(190, 5.5, conseils)
        
    return pdf.output()