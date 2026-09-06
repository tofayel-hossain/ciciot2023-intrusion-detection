# Standard GNN Baselines — CICIoT2023

Baseline evaluation of standard **Graph Neural Network (GNN) architectures** for Intrusion Detection on the **CICIoT2023** dataset.

### Models

* **GCN**
* **GNN**
* **GraphSAGE**
* **GIN**
* **GAT**
* **PINN / Constraint-Informed GNN**

### Results

| Model     |   Accuracy |   Macro F1 |        AUC |
| --------- | ---------: | ---------: | ---------: |
| **GAT**   | **72.20%** | **55.24%** |     96.74% |
| GIN       |     74.41% |     54.94% | **97.99%** |
| GraphSAGE |     62.09% |     46.44% |     95.64% |
| GCN       |     66.56% |     40.75% |     89.83% |
| PINN      |     66.51% |     40.68% |     89.83% |
| GNN       |     61.46% |     37.29% |     93.14% |

