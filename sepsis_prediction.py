"""
Modèle d'ensemble avancé pour la prédiction de septicémie
Objectif: Toutes les métriques > 0.8
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import gym
from gym import spaces
from collections import deque
import random
from sklearn.utils import class_weight
import warnings
warnings.filterwarnings('ignore')

# Configuration du seed pour la reproductibilité
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fonction pour charger et prétraiter les données avec techniques avancées
def load_sepsis_data_advanced(data_path="data/training_setA", 
                             seq_length=10, 
                             test_size=0.2, 
                             random_state=42,
                             balance_ratio=0.5,  # 50% de positifs
                             augment_factor=5):  # Multiplier les cas positifs par 5
    """
    Charge et prétraite les données de septicémie avec équilibrage des classes et augmentation avancée.
    """
    print(f"Chargement des données depuis: {data_path}")
    
    # Liste tous les fichiers PSV dans le dossier
    files = [f for f in os.listdir(data_path) if f.endswith('.psv')]
    print(f"Nombre total de fichiers: {len(files)}")
    
    # Initialisation des listes pour stocker les données
    X_sequences = []
    y_labels = []
    patient_ids = []
    
    # Chargement et prétraitement des données
    for i, file in enumerate(tqdm(files, desc="Chargement des fichiers")):
        file_path = os.path.join(data_path, file)
        
        # Chargement du fichier PSV
        df = pd.read_csv(file_path, sep='|')
        
        # Extraction des caractéristiques (toutes les colonnes sauf SepsisLabel)
        features = df.drop(columns=['SepsisLabel'])
        
        # Remplacement des valeurs manquantes avec des stratégies avancées
        # D'abord, remplir avec la dernière valeur valide
        features = features.ffill()
        # Ensuite, remplir avec la prochaine valeur valide
        features = features.bfill()
        # Pour les valeurs toujours manquantes, utiliser 0
        features = features.fillna(0)
        
        # Création de séquences de longueur seq_length avec chevauchement
        for j in range(0, len(df) - seq_length + 1, max(1, seq_length // 5)):  # Chevauchement de 80%
            seq = features.iloc[j:j+seq_length].values
            X_sequences.append(seq)
            
            # L'étiquette est 1 si le patient a eu une septicémie dans cette séquence
            label = 1 if df['SepsisLabel'].iloc[j:j+seq_length].max() > 0 else 0
            y_labels.append(label)
            
            # Enregistrement de l'ID du patient
            patient_ids.append(i)
    
    # Conversion en tableaux numpy
    X = np.array(X_sequences)
    y = np.array(y_labels)
    patient_ids = np.array(patient_ids)
    
    print(f"Forme des données originales: {X.shape}, {y.shape}")
    print(f"Pourcentage de cas positifs original: {np.mean(y) * 100:.2f}%")
    
    # Normalisation des données avec RobustScaler (plus résistant aux valeurs aberrantes)
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(n_samples * n_timesteps, n_features)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Division en ensembles d'entraînement et de test par patient
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    n_test_patients = int(n_patients * test_size)
    
    # Mélange aléatoire des patients
    np.random.seed(random_state)
    np.random.shuffle(unique_patients)
    
    test_patients = unique_patients[:n_test_patients]
    train_patients = unique_patients[n_test_patients:]
    
    # Création des masques pour la division
    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    X_train_full, X_test = X[train_mask], X[test_mask]
    y_train_full, y_test = y[train_mask], y[test_mask]
    
    # Équilibrage des classes pour l'ensemble d'entraînement
    pos_indices = np.where(y_train_full == 1)[0]
    neg_indices = np.where(y_train_full == 0)[0]
    
    # Augmentation avancée des données positives
    X_pos = X_train_full[pos_indices]
    y_pos = y_train_full[pos_indices]
    
    # Créer des variations des exemples positifs avec diverses techniques
    X_pos_augmented = []
    y_pos_augmented = []
    
    # Ajouter les exemples originaux
    X_pos_augmented.append(X_pos)
    y_pos_augmented.append(y_pos)
    
    # 1. Ajout de bruit gaussien
    for noise_level in [0.02, 0.05, 0.1]:
        noise = np.random.normal(0, noise_level, X_pos.shape)
        augmented = X_pos + noise
        X_pos_augmented.append(augmented)
        y_pos_augmented.append(np.ones_like(y_pos))
    
    # 2. Inversion temporelle (retourner les séquences)
    flipped = np.flip(X_pos, axis=1).copy()
    X_pos_augmented.append(flipped)
    y_pos_augmented.append(np.ones_like(y_pos))
    
    # 3. Permutation des pas de temps
    permuted = X_pos.copy()
    for i in range(len(permuted)):
        # Permuter légèrement l'ordre des pas de temps
        idx = np.arange(n_timesteps)
        np.random.shuffle(idx)
        permuted[i] = permuted[i, idx, :]
    X_pos_augmented.append(permuted)
    y_pos_augmented.append(np.ones_like(y_pos))
    
    # Concaténation de toutes les augmentations
    X_pos = np.vstack(X_pos_augmented)
    y_pos = np.concatenate(y_pos_augmented)
    
    # Sous-échantillonnage de la classe négative
    n_pos = len(X_pos)
    n_neg_to_keep = int(n_pos * (1 - balance_ratio) / balance_ratio)
    
    # Sous-échantillonnage aléatoire de la classe majoritaire (négative)
    np.random.seed(random_state)
    neg_indices_to_keep = np.random.choice(neg_indices, size=min(n_neg_to_keep, len(neg_indices)), replace=False)
    
    # Combinaison des exemples positifs (augmentés) et négatifs (sous-échantillonnés)
    X_neg = X_train_full[neg_indices_to_keep]
    y_neg = y_train_full[neg_indices_to_keep]
    
    X_train = np.concatenate([X_pos, X_neg])
    y_train = np.concatenate([y_pos, y_neg])
    
    # Mélange des données d'entraînement
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"Forme des données d'entraînement équilibrées: {X_train.shape}, {y_train.shape}")
    print(f"Pourcentage de cas positifs (entraînement équilibré): {np.mean(y_train) * 100:.2f}%")
    print(f"Forme des données de test: {X_test.shape}, {y_test.shape}")
    print(f"Pourcentage de cas positifs (test): {np.mean(y_test) * 100:.2f}%")
    
    # Calcul des poids de classe pour l'ensemble d'entraînement
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"Poids des classes: {class_weights}")
    
    return X_train, X_test, y_train, y_test, class_weights


# Environnement pour l'apprentissage par renforcement avec focus sur les cas positifs
class SepsisEnvEnsemble(gym.Env):
    """
    Environnement pour l'apprentissage par renforcement de la prédiction de septicémie
    avec une structure de récompense extrêmement asymétrique.
    """
    def __init__(self, X, y, 
                 reward_correct_negative=1.0,
                 reward_correct_positive=50.0,  # Récompense très élevée pour les vrais positifs
                 reward_false_positive=-2.0,    # Pénalité légère pour les faux positifs
                 reward_false_negative=-100.0): # Pénalité extrême pour les faux négatifs
        super(SepsisEnvEnsemble, self).__init__()
        
        self.X = X
        self.y = y
        self.n_samples = len(X)
        
        # Structure des récompenses fortement asymétrique
        self.reward_correct_negative = reward_correct_negative
        self.reward_correct_positive = reward_correct_positive
        self.reward_false_positive = reward_false_positive
        self.reward_false_negative = reward_false_negative
        
        # Définition de l'espace d'observation (état)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=X[0].shape, dtype=np.float32
        )
        
        # Définition de l'espace d'action (binaire: pas de septicémie / septicémie)
        self.action_space = spaces.Discrete(2)
        
        # État actuel
        self.current_idx = 0
        self.current_state = None
        self.done = False
        
        # Métriques de performance
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Indices des échantillons positifs pour l'échantillonnage stratifié
        self.positive_indices = np.where(self.y == 1)[0]
        self.negative_indices = np.where(self.y == 0)[0]
        
        # Compteur pour l'échantillonnage alterné
        self.sample_counter = 0
    
    def reset(self):
        """
        Réinitialise l'environnement avec échantillonnage stratifié.
        """
        # Échantillonnage alterné: deux exemples positifs, puis un négatif
        self.sample_counter += 1
        
        if self.sample_counter % 3 != 0 and len(self.positive_indices) > 0:
            self.current_idx = np.random.choice(self.positive_indices)
        else:
            self.current_idx = np.random.choice(self.negative_indices)
            
        self.current_state = self.X[self.current_idx]
        self.done = False
        return self.current_state
    
    def step(self, action):
        """
        Exécute une action dans l'environnement et renvoie l'état suivant, la récompense et un indicateur de fin.
        """
        if self.done:
            raise RuntimeError("Episode is done, call reset() to start a new episode.")
        
        # Vérification de la prédiction
        true_label = self.y[self.current_idx]
        
        # Calcul de la récompense avec structure fortement asymétrique
        if action == 1 and true_label == 1:  # Vrai positif
            reward = self.reward_correct_positive
            self.true_positives += 1
        elif action == 0 and true_label == 0:  # Vrai négatif
            reward = self.reward_correct_negative
            self.true_negatives += 1
        elif action == 1 and true_label == 0:  # Faux positif
            reward = self.reward_false_positive
            self.false_positives += 1
        else:  # action == 0 and true_label == 1, Faux négatif
            reward = self.reward_false_negative
            self.false_negatives += 1
        
        # Passage à un nouvel état (dans ce cas, on termine l'épisode après une prédiction)
        self.done = True
        
        # Informations supplémentaires
        info = {
            'true_label': true_label,
            'prediction': action,
            'tp': self.true_positives,
            'tn': self.true_negatives,
            'fp': self.false_positives,
            'fn': self.false_negatives
        }
        
        return self.current_state, reward, self.done, info
    
    def get_metrics(self):
        """
        Calcule et renvoie les métriques de performance actuelles.
        """
        tp = self.true_positives
        tn = self.true_negatives
        fp = self.false_positives
        fn = self.false_negatives
        
        accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
# Architecture avancée du modèle DQN avec attention et connexions résiduelles
class AdvancedDQN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(AdvancedDQN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couche d'embedding pour réduire la dimensionnalité
        self.embedding = nn.Linear(input_dim[1], hidden_size)
        
        # GRU bidirectionnel (plus léger que LSTM et souvent aussi performant)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Mécanisme d'attention multi-tête
        self.attention_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, 1),
                nn.Tanh()
            ) for _ in range(4)  # 4 têtes d'attention
        ])
        
        # Couches fully connected avec connexions résiduelles
        self.fc1 = nn.Linear(hidden_size * 2 * 4, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, output_size)
        
        # Connexions résiduelles
        self.res1 = nn.Linear(hidden_size * 2 * 4, hidden_size * 2)
        self.res2 = nn.Linear(hidden_size * 2, hidden_size)
        self.res3 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Layer normalization (plus stable que batch normalization)
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.layer_norm3 = nn.LayerNorm(hidden_size // 2)
        
        # Dropout pour éviter le surapprentissage
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Embedding des caractéristiques
        embedded = self.embedding(x)
        
        # GRU
        gru_out, _ = self.gru(embedded)  # [batch_size, seq_len, hidden_size*2]
        
        # Attention multi-tête
        attention_outputs = []
        for head in self.attention_heads:
            # Calcul des poids d'attention
            attention_weights = head(gru_out)  # [batch_size, seq_len, 1]
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Appliquer l'attention
            context = torch.bmm(attention_weights.transpose(1, 2), gru_out)  # [batch_size, 1, hidden_size*2]
            context = context.squeeze(1)  # [batch_size, hidden_size*2]
            attention_outputs.append(context)
        
        # Concaténation des sorties des têtes d'attention
        multi_head_context = torch.cat(attention_outputs, dim=1)  # [batch_size, hidden_size*2*4]
        
        # Couches fully connected avec connexions résiduelles et layer normalization
        # Bloc 1
        out1 = self.fc1(multi_head_context)
        res1 = self.res1(multi_head_context)
        out1 = out1 + res1  # Connexion résiduelle
        out1 = self.layer_norm1(out1)
        out1 = F.gelu(out1)  # GELU au lieu de ReLU
        out1 = self.dropout(out1)
        
        # Bloc 2
        out2 = self.fc2(out1)
        res2 = self.res2(out1)
        out2 = out2 + res2  # Connexion résiduelle
        out2 = self.layer_norm2(out2)
        out2 = F.gelu(out2)
        out2 = self.dropout(out2)
        
        # Bloc 3
        out3 = self.fc3(out2)
        res3 = self.res3(out2)
        out3 = out3 + res3  # Connexion résiduelle
        out3 = self.layer_norm3(out3)
        out3 = F.gelu(out3)
        out3 = self.dropout(out3)
        
        # Couche de sortie
        out = self.fc4(out3)
        
        return out


# Architecture CNN-LSTM pour capturer les motifs spatiaux et temporels
class CNNLSTMDQN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size, dropout_rate=0.5):
        super(CNNLSTMDQN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Couches CNN 1D pour extraire des caractéristiques
        self.conv1 = nn.Conv1d(input_dim[1], hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        
        # LSTM pour capturer les dépendances temporelles
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Mécanisme d'attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Tanh()
        )
        
        # Couches fully connected
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        batch_size, seq_len, features = x.size()
        
        # Permutation pour les couches CNN (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # Couches CNN avec skip connections
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = F.relu(self.bn2(self.conv2(conv1_out))) + conv1_out
        conv3_out = F.relu(self.bn3(self.conv3(conv2_out))) + conv2_out
        
        # Permutation pour LSTM (batch, seq_len, hidden_size)
        lstm_in = conv3_out.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(lstm_in)
        
        # Attention
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.transpose(1, 2), lstm_out)
        context = context.squeeze(1)
        
        # Fully connected
        out = F.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Transformer Encoder pour la prédiction de septicémie
class TransformerDQN(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_size, num_heads=4, dropout_rate=0.5):
        super(TransformerDQN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Couche d'embedding pour réduire la dimensionnalité
        self.embedding = nn.Linear(input_dim[1], hidden_size)
        
        # Encodage de position
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_dim[0], hidden_size))
        
        # Couches Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Couches fully connected
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Embedding
        embedded = self.embedding(x)
        
        # Ajout de l'encodage de position
        embedded = embedded + self.pos_encoder
        
        # Transformer Encoder
        transformer_out = self.transformer_encoder(embedded)
        
        # Pooling global (moyenne sur la dimension temporelle)
        pooled = torch.mean(transformer_out, dim=1)
        
        # Fully connected
        out = self.layer_norm(pooled)
        out = F.gelu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Agent DQN Ensemble
class EnsembleDQNAgent:
    def __init__(self, input_dim, action_size, 
                 hidden_size=256, 
                 num_layers=2, 
                 learning_rate=0.0001,
                 gamma=0.99, 
                 epsilon=1.0, 
                 epsilon_min=0.01, 
                 epsilon_decay=0.995, 
                 buffer_size=50000, 
                 batch_size=128, 
                 update_target_every=5):
        self.input_dim = input_dim
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.update_counter = 0
        
        # Création des modèles de l'ensemble
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation de: {self.device}")
        
        # Modèle 1: AdvancedDQN
        self.model1 = AdvancedDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target1 = AdvancedDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target1.load_state_dict(self.model1.state_dict())
        self.target1.eval()
        
        # Modèle 2: CNNLSTMDQN
        self.model2 = CNNLSTMDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target2 = CNNLSTMDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target2.load_state_dict(self.model2.state_dict())
        self.target2.eval()
        
        # Modèle 3: TransformerDQN
        self.model3 = TransformerDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target3 = TransformerDQN(input_dim, hidden_size, num_layers, action_size).to(self.device)
        self.target3.load_state_dict(self.model3.state_dict())
        self.target3.eval()
        
        # Optimiseurs avec learning rate scheduler et weight decay
        self.optimizer1 = optim.AdamW(self.model1.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.optimizer2 = optim.AdamW(self.model2.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.optimizer3 = optim.AdamW(self.model3.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        self.scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, mode='max', factor=0.5, patience=5)
        self.scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, mode='max', factor=0.5, patience=5)
        self.scheduler3 = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer3, mode='max', factor=0.5, patience=5)
        
        # Replay buffer
        self.memory = deque(maxlen=buffer_size)
        
        # Métriques de suivi
        self.training_loss = []
        self.rewards_history = []
        self.epsilon_history = []
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, evaluate=False):
        if not evaluate and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Prédictions des trois modèles
            q_values1 = self.model1(state)
            q_values2 = self.model2(state)
            q_values3 = self.model3(state)
            
            # Moyenne des prédictions (vote majoritaire)
            ensemble_q_values = (q_values1 + q_values2 + q_values3) / 3
        
        return torch.argmax(ensemble_q_values, dim=1).item()
    
    def get_q_values(self, state):
        """
        Obtient les Q-values de chaque modèle et leur moyenne.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values1 = self.model1(state)
            q_values2 = self.model2(state)
            q_values3 = self.model3(state)
            ensemble_q_values = (q_values1 + q_values2 + q_values3) / 3
        
        return {
            'model1': q_values1.cpu().numpy()[0],
            'model2': q_values2.cpu().numpy()[0],
            'model3': q_values3.cpu().numpy()[0],
            'ensemble': ensemble_q_values.cpu().numpy()[0]
        }
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Échantillonnage d'un batch de transitions
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Conversion en tenseurs PyTorch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Calcul des valeurs cibles pour chaque modèle
        with torch.no_grad():
            # Modèle 1
            next_actions1 = self.model1(next_states).argmax(1)
            next_q_values1 = self.target1(next_states).gather(1, next_actions1.unsqueeze(1)).squeeze(1)
            
            # Modèle 2
            next_actions2 = self.model2(next_states).argmax(1)
            next_q_values2 = self.target2(next_states).gather(1, next_actions2.unsqueeze(1)).squeeze(1)
            
            # Modèle 3
            next_actions3 = self.model3(next_states).argmax(1)
            next_q_values3 = self.target3(next_states).gather(1, next_actions3.unsqueeze(1)).squeeze(1)
            
            # Moyenne des valeurs cibles
            next_q_values = (next_q_values1 + next_q_values2 + next_q_values3) / 3
            
            # Calcul des valeurs cibles
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Entraînement de chaque modèle
        # Modèle 1
        q_values1 = self.model1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss1 = F.smooth_l1_loss(q_values1, target_q_values)
        
        self.optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(self.model1.parameters(), max_norm=1.0)
        self.optimizer1.step()
        
        # Modèle 2
        q_values2 = self.model2(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss2 = F.smooth_l1_loss(q_values2, target_q_values)
        
        self.optimizer2.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
        self.optimizer2.step()
        
        # Modèle 3
        q_values3 = self.model3(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss3 = F.smooth_l1_loss(q_values3, target_q_values)
        
        self.optimizer3.zero_grad()
        loss3.backward()
        torch.nn.utils.clip_grad_norm_(self.model3.parameters(), max_norm=1.0)
        self.optimizer3.step()
        
        # Mise à jour des réseaux cibles
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target1.load_state_dict(self.model1.state_dict())
            self.target2.load_state_dict(self.model2.state_dict())
            self.target3.load_state_dict(self.model3.state_dict())
        
        # Décroissance de l'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Perte moyenne
        avg_loss = (loss1.item() + loss2.item() + loss3.item()) / 3
        return avg_loss
    
    def update_lr(self, metric):
        self.scheduler1.step(metric)
        self.scheduler2.step(metric)
        self.scheduler3.step(metric)
    
    def save(self, path):
        torch.save({
            'model1_state_dict': self.model1.state_dict(),
            'model2_state_dict': self.model2.state_dict(),
            'model3_state_dict': self.model3.state_dict(),
            'target1_state_dict': self.target1.state_dict(),
            'target2_state_dict': self.target2.state_dict(),
            'target3_state_dict': self.target3.state_dict(),
            'optimizer1_state_dict': self.optimizer1.state_dict(),
            'optimizer2_state_dict': self.optimizer2.state_dict(),
            'optimizer3_state_dict': self.optimizer3.state_dict(),
            'scheduler1_state_dict': self.scheduler1.state_dict(),
            'scheduler2_state_dict': self.scheduler2.state_dict(),
            'scheduler3_state_dict': self.scheduler3.state_dict(),
            'epsilon': self.epsilon,
            'training_loss': self.training_loss,
            'rewards_history': self.rewards_history,
            'epsilon_history': self.epsilon_history
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model1.load_state_dict(checkpoint['model1_state_dict'])
        self.model2.load_state_dict(checkpoint['model2_state_dict'])
        self.model3.load_state_dict(checkpoint['model3_state_dict'])
        self.target1.load_state_dict(checkpoint['target1_state_dict'])
        self.target2.load_state_dict(checkpoint['target2_state_dict'])
        self.target3.load_state_dict(checkpoint['target3_state_dict'])
        self.optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
        self.optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
        self.optimizer3.load_state_dict(checkpoint['optimizer3_state_dict'])
        self.scheduler1.load_state_dict(checkpoint['scheduler1_state_dict'])
        self.scheduler2.load_state_dict(checkpoint['scheduler2_state_dict'])
        self.scheduler3.load_state_dict(checkpoint['scheduler3_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_loss = checkpoint['training_loss']
        self.rewards_history = checkpoint['rewards_history']
        self.epsilon_history = checkpoint['epsilon_history']
# Fonction d'entraînement avec validation croisée et curriculum learning
def train_ensemble_agent(env, agent, val_env, num_episodes=5000, batch_size=128, print_every=10, early_stopping_patience=100):
    """
    Entraîne l'agent ensemble avec curriculum learning, validation périodique et early stopping.
    """
    print(f"Entraînement de l'agent ensemble sur {num_episodes} épisodes...")
    rewards = []
    losses = []
    epsilons = []
    episode_rewards = []
    best_f1 = 0
    patience_counter = 0
    
    # Barre de progression
    progress_bar = tqdm(range(num_episodes), desc="Entraînement")
    
    for episode in progress_bar:
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Sélection d'une action
            action = agent.act(state)
            
            # Exécution de l'action
            next_state, reward, done, info = env.step(action)
            
            # Mémorisation de l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Mise à jour de l'état et de la récompense
            state = next_state
            episode_reward += reward
            
            # Apprentissage
            if len(agent.memory) > batch_size:
                loss = agent.replay()
                losses.append(loss)
        
        # Suivi des récompenses et de l'exploration
        rewards.append(episode_reward)
        epsilons.append(agent.epsilon)
        episode_rewards.append(episode_reward)
        
        # Validation et affichage des métriques périodiques
        if (episode + 1) % print_every == 0:
            # Évaluation sur l'environnement d'entraînement
            train_metrics = env.get_metrics()
            
            # Évaluation sur l'environnement de validation
            val_metrics = validate_ensemble_agent(val_env, agent, num_episodes=min(1000, len(val_env.X)))
            
            # Mise à jour de la barre de progression
            progress_bar.set_postfix({
                'reward': f"{np.mean(rewards[-print_every:]):.2f}",
                'epsilon': f"{agent.epsilon:.2f}",
                'val_f1': f"{val_metrics['f1']:.4f}",
                'val_acc': f"{val_metrics['accuracy']:.4f}"
            })
            
            # Mise à jour du learning rate basé sur le F1 score de validation
            agent.update_lr(val_metrics['f1'])
            
            # Vérification pour early stopping
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Sauvegarde du meilleur modèle
                agent.save('best_sepsis_ensemble_model.pt')
                
                # Affichage des métriques détaillées
                print(f"\nÉpisode {episode+1}: Nouvelles meilleures métriques!")
                print(f"F1: {val_metrics['f1']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
                
                # Si toutes les métriques sont supérieures à 0.8, on peut s'arrêter
                if (val_metrics['accuracy'] > 0.8 and 
                    val_metrics['precision'] > 0.8 and 
                    val_metrics['recall'] > 0.8 and 
                    val_metrics['f1'] > 0.8):
                    print(f"Toutes les métriques sont supérieures à 0.8! Arrêt de l'entraînement.")
                    break
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping à l'épisode {episode+1}. Meilleur F1: {best_f1:.4f}")
                break
    
    # Sauvegarde des historiques dans l'agent
    agent.training_loss = losses
    agent.rewards_history = rewards
    agent.epsilon_history = epsilons
    
    return rewards, losses, epsilons, best_f1


# Fonction de validation
def validate_ensemble_agent(env, agent, num_episodes=1000):
    """
    Valide l'agent sur un sous-ensemble de données sans modifier l'environnement.
    """
    # Sauvegarde de l'état actuel de l'environnement
    original_tp = env.true_positives
    original_tn = env.true_negatives
    original_fp = env.false_positives
    original_fn = env.false_negatives
    
    # Réinitialisation des compteurs pour la validation
    env.true_positives = 0
    env.true_negatives = 0
    env.false_positives = 0
    env.false_negatives = 0
    
    y_true = []
    y_pred = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Prédiction déterministe (pas d'exploration)
            action = agent.act(state, evaluate=True)
            
            # Exécution de l'action
            _, _, done, info = env.step(action)
            
            # Enregistrement pour les métriques
            y_true.append(info['true_label'])
            y_pred.append(action)
    
    # Calcul des métriques
    accuracy = (env.true_positives + env.true_negatives) / (env.true_positives + env.true_negatives + env.false_positives + env.false_negatives)
    precision = env.true_positives / max(1, env.true_positives + env.false_positives)
    recall = env.true_positives / max(1, env.true_positives + env.false_negatives)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    
    # Restauration de l'état original de l'environnement
    env.true_positives = original_tp
    env.true_negatives = original_tn
    env.false_positives = original_fp
    env.false_negatives = original_fn
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# Fonction d'évaluation complète
def evaluate_ensemble_agent(env, agent, num_episodes=None, threshold=0.5):
    """
    Évalue l'agent ensemble et calcule des métriques détaillées avec seuil de décision ajustable.
    """
    if num_episodes is None:
        num_episodes = len(env.X)
        
    print(f"Évaluation de l'agent sur {num_episodes} épisodes...")
    
    # Réinitialisation des compteurs de l'environnement
    env.true_positives = 0
    env.true_negatives = 0
    env.false_positives = 0
    env.false_negatives = 0
    
    y_true = []
    y_pred = []
    y_prob = []
    
    for _ in tqdm(range(num_episodes), desc="Évaluation"):
        state = env.reset()
        done = False
        
        while not done:
            # Prédiction déterministe (pas d'exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            
            with torch.no_grad():
                # Prédictions des trois modèles
                q_values1 = agent.model1(state_tensor)
                q_values2 = agent.model2(state_tensor)
                q_values3 = agent.model3(state_tensor)
                
                # Moyenne des prédictions
                ensemble_q_values = (q_values1 + q_values2 + q_values3) / 3
                
                # Probabilité de septicémie (softmax des Q-values)
                probabilities = F.softmax(ensemble_q_values, dim=1).cpu().numpy()[0]
                prob_positive = probabilities[1]
            
            # Action avec seuil ajustable
            action = 1 if prob_positive >= threshold else 0
            
            # Exécution de l'action
            _, _, done, info = env.step(action)
            
            # Enregistrement pour les métriques
            y_true.append(info['true_label'])
            y_pred.append(action)
            y_prob.append(prob_positive)
    
    # Calcul des métriques
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    
    # Calcul de l'AUC
    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
    auc = metrics.auc(fpr, tpr)
    
    # Matrice de confusion
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    # Rapport de classification
    classification_report = metrics.classification_report(y_true, y_pred, zero_division=0)
    
    # Affichage des résultats
    print("\nRésultats de l'évaluation:")
    print(f"Seuil de décision: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    print("\nMatrice de confusion:")
    print(cm)
    
    print("\nRapport de classification:")
    print(classification_report)
    
    # Tracé de la courbe ROC
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_ensemble.png')
    
    # Tracé de la courbe Precision-Recall
    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(recall_curve, precision_curve, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.savefig('precision_recall_curve_ensemble.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': classification_report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'threshold': threshold
    }


# Fonction pour trouver le meilleur seuil de décision
def find_optimal_threshold(y_true, y_prob):
    """
    Trouve le seuil optimal qui maximise le F1 score.
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = [1 if p >= threshold else 0 for p in y_prob]
        f1 = metrics.f1_score(y_true, y_pred)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def plot_training_metrics(agent, rewards, losses):
    """
    Trace les métriques d'entraînement.
    """
    plt.figure(figsize=(15, 15))
    
    # Tracé des récompenses
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Récompenses par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Récompense')
    
    # Tracé des pertes
    plt.subplot(3, 1, 2)
    plt.plot(losses)
    plt.title('Perte par batch')
    plt.xlabel('Batch')
    plt.ylabel('Perte')
    
    # Tracé de l'exploration
    plt.subplot(3, 1, 3)
    plt.plot(agent.epsilon_history)
    plt.title('Epsilon par épisode')
    plt.xlabel('Épisode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_metrics_ensemble.png')
    plt.close()


def main():
    """
    Fonction principale pour l'exécution du programme.
    """
    # Paramètres
    data_path = "data/training_setA"
    seq_length = 10
    hidden_size = 256
    num_layers = 2
    batch_size = 128
    num_episodes = 5000
    learning_rate = 0.0001
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # Chargement des données avec équilibrage des classes et augmentation avancée
    X_train, X_test, y_train, y_test, class_weights = load_sepsis_data_advanced(
        data_path=data_path,
        seq_length=seq_length,
        test_size=0.2,
        balance_ratio=0.5,  # 50% de positifs dans l'ensemble d'entraînement
        augment_factor=5    # Multiplier les cas positifs par 5
    )
    
    # Division de l'ensemble de test en validation et test
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42, stratify=y_test
    )
    
    print(f"Forme des données de validation: {X_val.shape}, {y_val.shape}")
    print(f"Forme des données de test final: {X_test.shape}, {y_test.shape}")
    
    # Conversion des poids de classe en tenseur PyTorch
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights)
    
    # Création des environnements
    train_env = SepsisEnvEnsemble(
        X=X_train,
        y=y_train,
        reward_correct_negative=1.0,
        reward_correct_positive=50.0,
        reward_false_positive=-2.0,
        reward_false_negative=-100.0
    )
    
    val_env = SepsisEnvEnsemble(
        X=X_val,
        y=y_val,
        reward_correct_negative=1.0,
        reward_correct_positive=50.0,
        reward_false_positive=-2.0,
        reward_false_negative=-100.0
    )
    
    test_env = SepsisEnvEnsemble(
        X=X_test,
        y=y_test,
        reward_correct_negative=1.0,
        reward_correct_positive=50.0,
        reward_false_positive=-2.0,
        reward_false_negative=-100.0
    )
    
    # Création de l'agent ensemble
    input_dim = (X_train.shape[1], X_train.shape[2])  # [seq_length, feature_dim]
    action_size = 2
    
    agent = EnsembleDQNAgent(
        input_dim=input_dim,
        action_size=action_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        update_target_every=5
    )
    
    # Entraînement de l'agent ensemble
    rewards, losses, epsilons, best_f1 = train_ensemble_agent(
        env=train_env,
        agent=agent,
        val_env=val_env,
        num_episodes=num_episodes,
        batch_size=batch_size,
        print_every=10,
        early_stopping_patience=100
    )
    
    # Tracé des métriques d'entraînement
    plot_training_metrics(agent, rewards, losses)
    
    # Chargement du meilleur modèle
    agent.load('best_sepsis_ensemble_model.pt')
    
    # Évaluation initiale avec seuil par défaut
    print("\nÉvaluation avec seuil par défaut (0.5):")
    evaluation_metrics = evaluate_ensemble_agent(test_env, agent, threshold=0.5)
    
    # Recherche du seuil optimal sur l'ensemble de validation
    print("\nRecherche du seuil optimal sur l'ensemble de validation...")
    val_results = evaluate_ensemble_agent(val_env, agent, threshold=0.5)
    optimal_threshold, _ = find_optimal_threshold(val_results['y_true'], val_results['y_prob'])
    
    print(f"\nSeuil optimal trouvé: {optimal_threshold:.4f}")
    
    # Évaluation finale avec le seuil optimal
    print(f"\nÉvaluation finale avec seuil optimal ({optimal_threshold:.4f}):")
    final_metrics = evaluate_ensemble_agent(test_env, agent, threshold=optimal_threshold)
    
    # Affichage des résultats finaux
    print("\nRésultats finaux (seuil optimal):")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"AUC: {final_metrics['auc']:.4f}")
    
    # Sauvegarde du modèle final
    agent.save('final_sepsis_ensemble_model.pt')
    
    # Sauvegarde des métriques et du seuil optimal
    np.savez(
        'final_ensemble_metrics.npz',
        accuracy=final_metrics['accuracy'],
        precision=final_metrics['precision'],
        recall=final_metrics['recall'],
        f1=final_metrics['f1'],
        auc=final_metrics['auc'],
        optimal_threshold=optimal_threshold
    )
    
    return final_metrics


if __name__ == "__main__":
    main()
