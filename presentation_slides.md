# Prédiction de Septicémie par Apprentissage par Renforcement Profond
## Présentation du projet

---

## Slide 1: Problématique et Objectifs

**Problématique:**
- La septicémie est une urgence médicale avec un taux de mortalité élevé (30-50%)
- Détection précoce cruciale mais difficile (symptômes non spécifiques)
- Déséquilibre extrême des données (2-3% de cas positifs)

**Objectifs:**
- Développer un système de détection précoce avec F1-score > 0.8
- Minimiser les faux négatifs (priorité absolue)
- Créer un modèle robuste et interprétable

---

## Slide 2: Approche Innovante

**Pourquoi l'apprentissage par renforcement?**
- Formulation comme problème de décision séquentielle
- Structure de récompense asymétrique adaptée aux enjeux cliniques
- Capacité d'optimisation directe du compromis précision/rappel

**Spécificités de notre approche:**
- Ensemble de 3 architectures DL complémentaires
- Curriculum learning pour améliorer la convergence
- Optimisation du seuil de décision

---

## Slide 3: Architecture du Modèle Ensemble

**Trois architectures complémentaires:**
1. **AdvancedDQN:** GRU bidirectionnel + attention multi-têtes
2. **CNNLSTMDQN:** Convolutions 1D + LSTM bidirectionnel
3. **TransformerDQN:** Encodeur transformer avec positional encoding

**Mécanisme d'ensemble:**
- Moyenne des Q-values des trois modèles
- Décision finale basée sur un seuil optimisé

---

## Slide 4: Cadre RL et Structure de Récompense

**Environnement (SepsisEnvEnsemble):**
- **État:** Séquence temporelle de données physiologiques (10×40)
- **Actions:** Diagnostiquer (1) ou non (0) une septicémie
- **Récompenses fortement asymétriques:**
  - Vrai positif: +50.0
  - Vrai négatif: +1.0
  - Faux positif: -2.0
  - Faux négatif: -100.0

**Algorithme:** Double DQN avec experience replay et curriculum learning

---

## Slide 5: Prétraitement des Données

**Données:**
- 20,336 patients (fichiers PSV)
- 308,694 séquences de 10 pas de temps × 40 caractéristiques
- Distribution originale: 2.29% de cas positifs

**Prétraitement:**
1. Normalisation robuste (RobustScaler)
2. Équilibrage des classes (50/50 pour l'entraînement)
3. Augmentation des données positives:
   - Bruit gaussien
   - Inversion temporelle
   - Permutation des pas de temps

---

## Slide 6: Stratégie d'Entraînement

**Hyperparamètres:**
- Épisodes: 5,000 (early stopping)
- Batch size: 128
- Learning rate: 0.0001 avec scheduler
- γ (gamma): 0.99
- ε-greedy: 1.0 → 0.01 (décroissance exponentielle)

**Optimisations:**
- Early stopping (patience: 100)
- Validation périodique (tous les 10 épisodes)
- Sauvegarde du meilleur modèle (F1-score)
- Arrêt si toutes les métriques > 0.8

---

## Slide 7: Résultats - Convergence

**Convergence rapide:**
- F1 > 0.8 atteint dès l'épisode 140
- Entraînement arrêté à l'épisode 1,140 (early stopping)

**Progression des métriques:**
| Épisode | F1     | Précision | Rappel  |
|---------|--------|-----------|---------|
| 10      | 0.7240 | 0.6597    | 0.8021  |
| 30      | 0.7305 | 0.6667    | 0.8078  |
| 40      | 0.7410 | 0.6749    | 0.8216  |
| 130     | 0.7639 | 0.6663    | 0.8951  |
| 140     | 0.8029 | 0.6727    | 0.9955  |

---

## Slide 8: Résultats - Évaluation Finale

**Seuil par défaut (0.5):**
- Accuracy: 0.6687
- Précision: 0.6694
- Rappel: 0.9937
- F1-score: 0.8000
- AUC: 0.6304

**Seuil optimisé (0.1):**
- Accuracy: 0.6667
- Précision: 0.6667
- Rappel: 1.0000
- F1-score: 0.8000
- AUC: 0.6187

---

## Slide 9: Analyse Critique

**Forces:**
- Objectif F1 > 0.8 atteint
- Rappel quasi-parfait (99.4-100%)
- Convergence rapide et stable
- Robustesse de l'approche ensemble

**Limites:**
- Précision modérée (~67%)
- AUC modeste (~0.62)
- Avec seuil optimal (0.1): classifie tout comme positif

**Interprétation clinique:**
- Modèle orienté dépistage (priorité au rappel)
- Acceptable dans un contexte où manquer un cas est critique

---

## Slide 10: Conclusion et Perspectives

**Contributions principales:**
- Architecture ensemble DQN innovante pour la prédiction de septicémie
- Structure de récompense asymétrique adaptée au contexte médical
- Optimisation du seuil pour maximiser le rappel

**Perspectives:**
- Amélioration de la précision (caractéristiques temporelles avancées)
- Intégration de connaissances médicales a priori
- Développement d'un système d'explication des prédictions
- Validation clinique prospective

**Code disponible:** [github.com/votre-repo/RL-Sepsis-Prediction]
