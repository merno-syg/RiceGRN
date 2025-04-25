# Gene Regulatory Network Prediction Using Graph Neural Networks

## Overview

This project presents a novel Graph Neural Network (GNN) framework for predicting gene regulatory interactions between Transcription Factors (TFs) and their target genes. The model integrates multi-omic data sources within a graph-based learning paradigm to accurately infer regulatory relationships.

## Key Features

- **Bipartite Graph Representation**: Models TF-target interactions as a directed bipartite graph
- **Multi-modal Data Integration**:
  - Protein sequence embeddings using ProtBERT (TF nodes)
  - DNA promoter region representations via k-mer encoding (target nodes)
  - 22-dimensional edge attributes capturing regulatory context
- **Advanced Architecture**:
  - Custom Message Passing Neural Network (MPNN) with multiple propagation blocks
  - Hierarchical feature aggregation from sequence and graph data
  - Attention-based feature fusion

## Technical Specifications

### Requirements

- Python 3.7+
- PyTorch 1.10+
- PyTorch Geometric 2.0+
- HuggingFace Transformers
- BioPython
- NetworkX
- scikit-learn
- pandas
- numpy

### Installation

```
pip install torch torch-geometric transformers biopython networkx scikit-learn pandas numpy tqdm
```

## Data Requirements

Input data should be structured as a pandas DataFrame containing:

- **Node Features**:
  - TF protein sequences (TF_pep)
  - Target gene promoter sequences (Target_promoter, 2500bp)
- **Edge Information**:
  - TF-Target pairs with binary labels (1=interaction, 0=no interaction)
  - 22 regulatory context features
- **Metadata**: Additional biological annotations

## Model Architecture

1. **Embedding Layers**:
   - Protein sequences → ProtBERT embeddings (1024D)
   - DNA sequences → k-mer frequency vectors (learned encoding)
2. **Graph Neural Network**:
   - 3 MPNN blocks with residual connections
   - Edge feature gating mechanism
   - Neighborhood aggregation with attention
3. **Prediction Head**:
   - 3-layer MLP with dropout
   - Sigmoid activation for probability output

## Training Protocol

- **Optimization**: AdamW (lr=5e-4, weight_decay=1e-5)
- **Loss Function**: BCEWithLogitsLoss with class weighting
- **Training Regimen**:
  - 300 epochs with early stopping
  - Full-batch training (graph-level)
  - GPU acceleration recommended

## Evaluation Metrics

- Primary Metrics:
  - AUC-ROC
  - Precision-Recall
  - F1-score
- Secondary Metrics:
  - Training/validation loss curves
  - Attention weight analysis

## Usage Pipeline

1. Prepare regulatory network data
2. Configure model parameters in `config.yaml`
3. Execute training script:

```
python train.py --data_path network_data.csv --output_dir results/
```

```
python evaluate.py --model_checkpoint best_model.pt --test_data test_set.csv
```

## Customization Options

- **Architecture**:
  - Adjust MPNN depth (2-5 layers)
  - Modify hidden dimensions (128-1024 units)
- **Training**:
  - Learning rate scheduling
  - Alternative loss functions
- **Data**:
  - Alternative sequence encodings
  - Additional edge features

## Performance Notes

- Achieves state-of-the-art performance on benchmark datasets
- Requires CUDA-enabled GPU for efficient training
- Pretrained ProtBERT model (~420MB) automatically cached
- Implemented TF-centric data splitting to prevent information leakage

This framework provides researchers with a powerful tool for computational regulatory network inference, combining modern deep learning approaches with biological sequence information.