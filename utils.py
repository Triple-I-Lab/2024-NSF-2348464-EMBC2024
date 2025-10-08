"""
Utility functions for ECG data preprocessing and partitioning
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_ecg_data(normal_path, abnormal_path):
    """Load and combine normal and abnormal ECG data"""
    normal_df = pd.read_csv(normal_path, header=None)
    abnormal_df = pd.read_csv(abnormal_path, header=None)
    
    # Combine datasets
    df = pd.concat([normal_df, abnormal_df], axis=0).reset_index(drop=True)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print(f"Loaded {len(X)} ECG samples")
    print(f"Normal: {np.sum(y == 0)}, Abnormal: {np.sum(y == 1)}")
    
    return X, y


def preprocess_data(X, y, test_size=0.3, random_state=42):
    """Preprocess and split data"""
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape for CNN input (samples, timesteps, channels)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def partition_data(X_train, y_train, num_clients, validation_split=0.2):
    """Partition data for federated clients"""
    client_data = []
    samples_per_client = len(X_train) // num_clients
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else len(X_train)
        
        X_client = X_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        
        # Split into train and validation for each client
        split_idx = int(len(X_client) * (1 - validation_split))
        X_train_client = X_client[:split_idx]
        y_train_client = y_client[:split_idx]
        X_val_client = X_client[split_idx:]
        y_val_client = y_client[split_idx:]
        
        client_data.append((X_train_client, y_train_client, X_val_client, y_val_client))
        print(f"Client {i}: Train={len(X_train_client)}, Val={len(X_val_client)}")
    
    return client_data


def calculate_metrics(precision, recall):
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return {'f1_score': 0.0}
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return {'f1_score': f1}