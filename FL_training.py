"""
Federated Learning client and server implementation
"""
import numpy as np
import flwr as fl
from typing import Dict, List, Tuple
from flwr.common import Metrics


class ECGFlowerClient(fl.client.NumPyClient):
    """Flower client for ECG classification"""
    
    def __init__(self, model, x_train, y_train, x_val, y_val, client_id=0):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.client_id = client_id
        
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        print(f"[Client {self.client_id}] Getting parameters")
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        print(f"[Client {self.client_id}] Setting parameters")
        self.model.set_weights(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train model locally"""
        print(f"\n[Client {self.client_id}] Starting local training")
        print(f"  Config: {config}")
        
        self.set_parameters(parameters)
        
        # Training configuration
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        
        # Train model
        history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        print(f"[Client {self.client_id}] Training completed")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Final accuracy: {final_accuracy:.4f}")
        
        return (
            self.get_parameters(config),
            len(self.x_train),
            {
                "loss": final_loss,
                "accuracy": final_accuracy
            }
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate model"""
        print(f"[Client {self.client_id}] Evaluating model")
        
        self.set_parameters(parameters)
        
        # Evaluate model
        results = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        loss = results[0]
        accuracy = results[1]
        precision = results[2]
        recall = results[3]
        
        print(f"  Validation loss: {loss:.4f}")
        print(f"  Validation accuracy: {accuracy:.4f}")
        
        return (
            float(loss),
            len(self.x_val),
            {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall)
            }
        )


def create_client_fn(model_fn, x_train, y_train, x_val, y_val, client_id=0):
    """Factory function to create Flower client"""
    model = model_fn()
    return ECGFlowerClient(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        client_id=client_id
    )


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics using weighted averaging"""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    aggregated = {}
    
    if metrics:
        metric_names = metrics[0][1].keys()
        
        for metric_name in metric_names:
            weighted_sum = sum([
                num_examples * m[metric_name] 
                for num_examples, m in metrics
            ])
            aggregated[metric_name] = weighted_sum / total_examples
    
    return aggregated


def get_fit_config(server_round: int, epochs_per_round: int, batch_size: int) -> Dict:
    """Return training configuration for each round"""
    config = {
        "epochs": epochs_per_round,
        "batch_size": batch_size,
        "round": server_round
    }
    return config


def create_strategy(num_clients: int, epochs_per_round: int, batch_size: int, num_rounds: int):
    """Create federated averaging strategy"""
    
    def fit_config_fn(server_round: int):
        print(f"\n{'='*80}")
        print(f"Round {server_round}/{num_rounds}")
        print(f"{'='*80}")
        return get_fit_config(server_round, epochs_per_round, batch_size)
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config_fn,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    
    return strategy