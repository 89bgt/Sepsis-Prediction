# Architecture du Modèle Ensemble pour la Prédiction de Septicémie

## Vue d'Ensemble

Ce projet implémente un système avancé de prédiction de la septicémie utilisant un ensemble (ensemble learning) de trois modèles d'apprentissage par renforcement profond. L'approche combine les forces de différentes architectures de réseaux de neurones pour améliorer la robustesse et la précision des prédictions.

## Architecture du Modèle

### 1. Modèle 1: AdvancedDQN (Réseau de Neurones Profonds Avancé)

**Architecture :**
- **Couche d'embedding** : Réduction de la dimensionnalité des caractéristiques d'entrée
- **GRU Bidirectionnel** : Capture les dépendances temporelles dans les deux sens
  - Taille de couche cachée : 256
  - Nombre de couches : 2
  - Dropout : 0.5
- **Mécanisme d'Attention Multi-Têtes** : 4 têtes d'attention
- **Réseau de Valeur Profond** avec connexions résiduelles
  - 4 couches fully-connected avec normalisation par couche
  - Fonction d'activation ReLU
  - Dropout de 0.5

### 2. Modèle 2: CNNLSTMDQN (Réseau CNN-LSTM)

**Architecture :**
- **Module CNN** :
  - 3 couches de convolution 1D
  - Taille des noyaux : 3
  - Batch Normalization après chaque couche
  - Fonction d'activation ReLU
- **Couche LSTM** :
  - Bidirectionnelle
  - Taille de couche cachée : 256
  - 2 couches empilées
- **Mécanisme d'Attention** :
  - Une seule tête d'attention
- **Réseau de Valeur** :
  - 2 couches fully-connected
  - Dropout de 0.5

### 3. Modèle 3: TransformerDQN (Architecture Transformer)

**Architecture :**
- **Couche d'Embedding** :
  - Projection des caractéristiques d'entrée
- **Positional Encoding** :
  - Encodage apprenable des positions
- **Encodeur Transformer** :
  - 2 couches d'encodeur
  - 4 têtes d'attention
  - Dimension du feed-forward : 1024
  - Dropout : 0.5
  - Activation GELU
- **Tête de Sortie** :
  - 2 couches fully-connected
  - Normalisation par couche

## Mécanisme d'Ensemble

### Agrégation des Prédictions
Les trois modèles fonctionnent en parallèle et leurs prédictions sont combinées selon une moyenne pondérée où chaque modèle contribue également à la décision finale.

### Apprentissage
- **Optimiseur** : AdamW pour chaque modèle
- **Taux d'apprentissage** : 0.0001
- **Pénalité L2** : 1e-4
- **Scheduler** : ReduceLROnPlateau avec facteur 0.5 et patience 5
- **Taille du batch** : 128
- **Mémoire de rejeu** : 50 000 transitions

## Structure des Récompenses

La fonction de récompense est conçue pour pénaliser fortement les faux négatifs :
- Vrai Positif : +50.0
- Vrai Négatif : +1.0
- Faux Positif : -2.0
- Faux Négatif : -100.0

## Données et Prétraitement

### Format des Données
- **Entrée** : Séquences temporelles de 10 pas de temps
- **Caractéristiques** : Variables physiologiques et cliniques
- **Cible** : Étiquette binaire de septicémie (0 ou 1)

### Prétraitement
1. **Nettoyage** :
   - Suppression des doublons
   - Traitement des valeurs manquantes par propagation avant/arrière
2. **Normalisation** :
   - RobustScaler pour une meilleure résistance aux valeurs aberrantes
3. **Équilibrage des Classes** :
   - Sous-échantillonnage de la classe majoritaire
   - Augmentation des données pour la classe minoritaire
   - Ratio cible : 50/50 entre les classes

## Entraînement

### Paramètres
- **Nombre d'épisodes** : 5000
- **Taille du batch** : 128
- **γ (gamma)** : 0.99
- **ε (epsilon)** : 1.0 → 0.01 (décroissance exponentielle)
- **Taux de mise à jour du réseau cible** : Tous les 5 épisodes

### Stratégies d'Entraînement
1. **Curriculum Learning** :
   - Augmentation progressive de la difficulté
   - Adaptation dynamique des hyperparamètres
2. **Early Stopping** :
   - Patience : 100 épisodes
   - Basé sur le F1-score de validation
3. **Réduction du Taux d'Apprentissage** :
   - Réduction par 2 après 5 époques sans amélioration

## Évaluation

### Métriques Principales
1. **F1-Score** : Métrique principale d'optimisation
2. **Précision** : Éviter les faux positifs
3. **Rappel** : Détecter le maximum de vrais positifs
4. **Spécificité** : Éviter les faux négatifs
5. **AUC-ROC** : Performance globale à différents seuils

### Validation Croisée
- Division 80/20 des données
- Évaluation sur un ensemble de test indépendant
- Validation périodique pendant l'entraînement

## Utilisation

### Prérequis
- Python 3.7+
- PyTorch 1.8+
- NumPy, Pandas, Scikit-learn
- Gym

### Entraînement
```python
from sepsis_ensemble_80plus import main
main()
```

### Chargement d'un Modèle Existant
```python
agent = EnsembleDQNAgent(input_dim, action_size)
agent.load("chemin/vers/modele.pth")
```

## Intégration du Deep Learning et du Reinforcement Learning

### 1. Approche Hybride DL-RL

Ce projet combine le Deep Learning (DL) et le Reinforcement Learning (RL) dans une architecture hybride où :
- **Le Deep Learning** fournit la capacité d'apprentissage de représentations complexes
- **Le Reinforcement Learning** fournit le cadre d'apprentissage par essai-erreur guidé par les récompenses

### 2. Rôle du Deep Learning

#### 2.1. Approximateur de Fonction
- Les trois modèles (AdvancedDQN, CNNLSTMDQN, TransformerDQN) sont des réseaux de neurones profonds
- Ils apprennent à estimer la fonction de valeur Q(s,a) - la qualité d'une action dans un état donné

#### 2.2. Extraction de Caractéristiques
- Les couches CNN extraient des motifs locaux dans les séquences temporelles
- Les couches récurrentes (GRU/LSTM) capturent les dépendances temporelles
- Les mécanismes d'attention identifient les parties les plus informatives

### 3. Cadre du Reinforcement Learning

#### 3.1. Environnement (SepsisEnvEnsemble)
- **État** : Données physiologiques du patient sur une fenêtre temporelle
- **Actions** : 
  - 0 : Ne pas diagnostiquer de septicémie
  - 1 : Diagnostiquer une septicémie
- **Récompenses** :
  - Vrai positif : +50.0
  - Vrai négatif : +1.0
  - Faux positif : -2.0
  - Faux négatif : -100.0 (forte pénalité)

#### 3.2. Algorithme DQN Amélioré

##### a) Experience Replay
```python
def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))
```
- Stocke les expériences pour un rééchantillonnage ultérieur
- Brise les corrélations temporelles

##### b) Double DQN
- Utilise deux réseaux (courant et cible) pour éviter la surestimation des valeurs Q
- Mise à jour du réseau cible périodique

##### c) Exploration vs Exploitation
- Stratégie ε-greedy avec décroissance exponentielle
- Passe d'une exploration aléatoire à une exploitation des connaissances acquises

### 4. Boucle d'Apprentissage

1. **Initialisation** :
   ```python
   agent = EnsembleDQNAgent(input_dim, action_size=2)
   env = SepsisEnvEnsemble(X_train, y_train)
   ```

2. **Boucle Principale** :
   - Pour chaque épisode (patient) :
     1. Réinitialisation de l'environnement
     2. Pour chaque pas de temps :
        - Sélection d'une action (exploration/exploitation)
        - Exécution de l'action dans l'environnement
        - Stockage de l'expérience
        - Apprentissage par lots (mini-batch)
        - Mise à jour périodique du réseau cible

### 5. Améliorations par Rapport à un DQN Standard

1. **Architecture d'Ensemble** :
   - Combinaison de trois architectures différentes pour une meilleure robustesse
   - Réduction de la variance des prédictions

2. **Stratégie de Récompense** :
   - Récompenses fortement asymétriques pour pénaliser davantage les faux négatifs
   - Encourage la détection précoce des cas de septicémie

3. **Mécanismes d'Attention** :
   - Permettent au modèle de se concentrer sur les parties les plus informatives
   - Améliorent l'interprétabilité des décisions

4. **Curriculum Learning** :
   - Adaptation progressive de la difficulté pendant l'entraînement
   - Commence par des cas faciles avant de passer aux plus complexes

## Performance

L'objectif de performance est d'atteindre un score F1 > 0.8 sur l'ensemble de test, avec un bon équilibre entre précision et rappel.

## Auteur

[Votre Nom]
[Votre Institution/Organisation]
[Année]
