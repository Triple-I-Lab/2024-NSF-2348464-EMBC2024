"""
Main script for Federated Learning ECG Classification
"""
import os
import argparse
import flwr as fl

from model import create_stacked_cnn
from utils import load_ecg_data, preprocess_data, partition_data, calculate_metrics
from FL_training import ECGFlowerClient, create_strategy


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Federated Learning for ECG Classification'
    )
    parser.add_argument('--normal-data', type=str, default='data/ptbdb_normal.csv')
    parser.add_argument('--abnormal-data', type=str, default='data/ptbdb_abnormal.csv')
    parser.add_argument('--num-clients', type=int, default=3)
    parser.add_argument('--num-rounds', type=int, default=100)
    parser.add_argument('--epochs-per-round', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--output-dir', type=str, default='output')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("FEDERATED LEARNING FOR ECG SIGNAL CLASSIFICATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Clients: {args.num_clients}")
    print(f"  Rounds: {args.num_rounds}")
    print(f"  Epochs per round: {args.epochs_per_round}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Load data
    print("\n" + "-" * 80)
    print("[1/5] Loading Data")
    print("-" * 80)
    X, y = load_ecg_data(args.normal_data, args.abnormal_data)
    
    # Preprocess data
    print("\n" + "-" * 80)
    print("[2/5] Preprocessing Data")
    print("-" * 80)
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=args.test_size)
    
    # Partition data
    print("\n" + "-" * 80)
    print("[3/5] Partitioning Data for Federated Clients")
    print("-" * 80)
    client_data = partition_data(X_train, y_train, args.num_clients)
    
    # Setup FL
    print("\n" + "-" * 80)
    print("[4/5] Setting Up Federated Learning")
    print("-" * 80)
    
    def client_fn(cid: str):
        client_id = int(cid)
        x_train, y_train, x_val, y_val = client_data[client_id]
        
        model = create_stacked_cnn(
            input_shape=(X_train.shape[1], 1),
            learning_rate=args.learning_rate
        )
        
        return ECGFlowerClient(
            model=model,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            client_id=client_id
        )
    
    strategy = create_strategy(
        num_clients=args.num_clients,
        epochs_per_round=args.epochs_per_round,
        batch_size=args.batch_size,
        num_rounds=args.num_rounds
    )
    
    print("Federated learning configuration complete!")
    
    # Start training
    print("\n" + "-" * 80)
    print("[5/5] Starting Federated Learning Training")
    print("-" * 80)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # Evaluate final model
    print("\n" + "=" * 80)
    print("EVALUATING FINAL GLOBAL MODEL")
    print("=" * 80)
    
    final_model = create_stacked_cnn(
        input_shape=(X_train.shape[1], 1),
        learning_rate=args.learning_rate
    )
    
    results = final_model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc, test_prec, test_rec = results[0], results[1], results[2], results[3]
    
    metrics = calculate_metrics(test_prec, test_rec)
    f1_score = metrics['f1_score']
    
    print(f"\nFinal Test Results:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"  Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"  F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"  Loss:      {test_loss:.4f}")
    
    # Save results
    model_path = os.path.join(args.output_dir, 'federated_ecg_model.h5')
    final_model.save(model_path)
    print(f"\n Model saved: {model_path}")
    
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FEDERATED LEARNING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration:\n")
        f.write(f"  Number of clients: {args.num_clients}\n")
        f.write(f"  Number of rounds: {args.num_rounds}\n")
        f.write(f"  Epochs per round: {args.epochs_per_round}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.learning_rate}\n\n")
        f.write("Final Test Results:\n")
        f.write(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)\n")
        f.write(f"  Precision: {test_prec:.4f} ({test_prec*100:.2f}%)\n")
        f.write(f"  Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)\n")
        f.write(f"  F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)\n")
        f.write(f"  Loss:      {test_loss:.4f}\n")
    
    print(f" Results saved: {results_path}")
    print("\n" + "=" * 80)
    print(" FEDERATED LEARNING COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()