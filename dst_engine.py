import numpy as np

class BBA:
    def __init__(self, masses=None):
        if masses is None:
            self.masses = {frozenset(['comfort', 'inconfort']): 1.0}
        else:
            self.masses = {frozenset(k): float(v) for k, v in masses.items() if v > 0}
        self._normaliser()

    def _normaliser(self):
        total = sum(self.masses.values())
        if total > 0:
            self.masses = {k: v / total for k, v in self.masses.items()}

    def affaiblir(self, alpha):
        nouvelles_masses = {}
        omega = frozenset(['comfort', 'inconfort'])
        for focal, val in self.masses.items():
            if focal == omega:
                continue
            nouvelles_masses[focal] = alpha * val
        somme_affaiblie = sum(nouvelles_masses.values())
        nouvelles_masses[omega] = nouvelles_masses.get(omega, 0.0) + (1.0 - somme_affaiblie)
        self.masses = {k: v for k, v in nouvelles_masses.items() if v > 0}

    def fusion_dempster(self, other):
        combinaisons = {}
        conflit = 0.0
        for f1, m1 in self.masses.items():
            for f2, m2 in other.masses.items():
                intersection = f1.intersection(f2)
                produit = m1 * m2
                if not intersection:
                    conflit += produit
                else:
                    combinaisons[intersection] = combinaisons.get(intersection, 0.0) + produit
        if conflit >= 1.0:
            return BBA(), 1.0
        nouvelles_masses = {k: v / (1.0 - conflit) for k, v in combinaisons.items()}
        return BBA(nouvelles_masses), conflit

    def betp(self, hypothese):
        pignistique = 0.0
        for focal, val in self.masses.items():
            if hypothese in focal:
                pignistique += val / len(focal)
        return pignistique

def bba_depuis_mesure(val, seuils):
    if np.isnan(val) or val == 0:
        return BBA()
    mini, maxi = seuils
    omega = frozenset(['comfort', 'inconfort'])
    if mini <= val <= maxi:
        centre = (mini + maxi) / 2
        distance_max = (maxi - mini) / 2
        facteur_proximite = 1.0 - (abs(val - centre) / distance_max) if distance_max > 0 else 1.0
        m_comfort = 0.5 + 0.4 * facteur_proximite
        return BBA({frozenset(['comfort']): m_comfort, omega: 1.0 - m_comfort})
    else:
        ecart = min(abs(val - mini), abs(val - maxi))
        m_inconfort = min(0.9, 0.4 + 0.1 * ecart)
        return BBA({frozenset(['inconfort']): m_inconfort, omega: 1.0 - m_inconfort})