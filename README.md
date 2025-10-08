# Federated Learning for ECG Signal Classification

Privacy-preserving ECG signal classification using Federated Learning with Stacked CNN architecture.

## Overview

This project implements a federated learning system for binary ECG classification (Normal vs Abnormal) using the PTB Diagnostic ECG Database. Achieves **98.6% accuracy** while preserving data privacy across distributed clients.

## Architecture

- **Model**: Stacked CNN (256→128→64→32 filters + Dense layers)
- **Framework**: Flower (Federated Learning)
- **Strategy**: FedAvg with weighted averaging
- **Dataset**: PTB Diagnostic ECG Database (14,552 samples)

## Project Structure
├── data/
│   ├── ptbdb_normal.csv
│   └── ptbdb_abnormal.csv
├── output/
├── utils.py              # Data preprocessing & partitioning
├── model.py              # Stacked CNN architecture
├── FL_training.py        # Federated client & server
├── main.py               # Main execution script
└── README.md

## Installation
```bash
pip install -r requirements.txt
```
## Basic usage
```bash
python main.py \
  --num-clients 3 \
  --num-rounds 100 \
  --epochs-per-round 50 \
  --batch-size 32 \
  --learning-rate 0.001
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--normal-data` | `data/ptbdb_normal.csv` | Path to normal ECG data |
| `--abnormal-data` | `data/ptbdb_abnormal.csv` | Path to abnormal ECG data |
| `--num-clients` | `3` | Number of federated clients |
| `--num-rounds` | `100` | FL communication rounds |
| `--epochs-per-round` | `50` | Local training epochs per round |
| `--batch-size` | `32` | Training batch size |
| `--learning-rate` | `0.001` | Optimizer learning rate |
| `--test-size` | `0.3` | Test set proportion |
| `--output-dir` | `output` | Directory for results |

## Citation

Quoc Bao Phan, Linh Nguyen, Ngoc Thang Bui, Dinh C. Nguyen, Lan Zhang, and Tuy Tan Nguyen, "Federated Learning for Enhanced ECG Signal Classification with Privacy Awareness," 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC 2024), Orlando, Florida, USA, 15–19 Jul. 2024, pp. 1–4.

## License
MIT License.