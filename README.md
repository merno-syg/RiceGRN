Gene Regulatory Network Prediction with Graph Neural Networks
Overview
This project implements a Graph Neural Network (GNN) model for predicting gene regulatory interactions between Transcription Factors (TFs) and target genes. The model combines sequence information from protein sequences (for TFs) and DNA promoter regions (for target genes) with additional edge attributes to predict regulatory relationships.

Key Features
Bipartite Graph Structure: Models TF-target interactions as a bipartite graph

Multi-modal Input:

Protein sequences for TFs processed with a protein language model

DNA promoter sequences for target genes processed with k-mer encoding

Additional edge features (22 different attributes)

Message Passing Neural Network: Custom GNN architecture with multiple MPNN blocks

Pre-trained Embeddings: Uses ProtBERT for protein sequence embeddings

Requirements
Python 3.7+

PyTorch

PyTorch Geometric

Transformers library (HuggingFace)

BioPython

NetworkX

scikit-learn

pandas

numpy

tqdm

Installation
bash
pip install torch torch-geometric transformers biopython networkx scikit-learn pandas numpy tqdm
Data Preparation
The input data should be in a pandas DataFrame with the following columns:

TF: Transcription factor names

Target: Target gene names

Label: Binary labels (1 for interaction, 0 for no interaction)

TF_pep: Protein sequences for TFs

Target_promoter: DNA promoter sequences for targets (2500bp)

Edge_attr*: 22 different edge attributes

Additional metadata columns

Model Architecture
The model consists of:

Embedding Layers:

Protein sequence embedding using ProtBERT

DNA sequence embedding using k-mer vocabulary

GNN Layers:

Multiple MPNN blocks with message passing between TFs and targets

Edge feature processing

Prediction Head:

MLP that combines node and edge features for final prediction

Training
Key training parameters:

Learning rate: 5e-4

Batch size: Full graph (implicit)

Loss function: BCEWithLogitsLoss

Optimizer: AdamW

Training epochs: 300

Evaluation Metrics
Accuracy

AUC-ROC

Loss curves

Usage
Prepare your data in the required format

Update file paths in the code as needed

Adjust hyperparameters if necessary

Run the training script

Outputs
Training and validation loss curves

Best model performance metrics

Predictions on test set

Customization
You can modify:

GNN depth (number of MPNN blocks)

Hidden layer dimensions

Edge feature processing

Sequence embedding methods

Training hyperparameters

Notes
The code assumes access to GPU (CUDA)

Pretrained ProtBERT model needs to be available at specified path

Data splitting is done by TF to prevent information leakage
