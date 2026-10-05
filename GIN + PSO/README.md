# GIN + PSO Optimized CICIoT2023

## Overview

Optimized **Graph Isomorphism Network (GIN)** for intrusion detection on the **CICIoT2023** dataset using PSO-selected hyperparameters.

**Environment**

* Device: CUDA
* GPU: NVIDIA Tesla T4
* Seed: 42
* Features: 46
* Cleaned samples: 299,617
* Removed rows: 383

## Classes

**Total class classification task contains:**
```text
DDoS-ICMP_Flood
DDoS-UDP_Flood
DDoS-TCP_Flood
DDoS-PSHACK_Flood
DDoS-SYN_Flood
DDoS-RSTFINFlood
DDoS-SynonymousIP_Flood
DoS-UDP_Flood
DoS-TCP_Flood
DoS-SYN_Flood
BenignTraffic
Mirai-greeth_flood
Mirai-udpplain
Mirai-greip_flood
DDoS-ICMP_Fragmentation
MITM-ArpSpoofing
DDoS-UDP_Fragmentation
DDoS-ACK_Fragmentation
DNS_Spoofing
Recon-HostDiscovery
```

Classes with fewer than **10,000 samples** were grouped into `Others`.

## Dataset Split

| Split      | Samples |
| ---------- | ------: |
| Train      | 239,693 |
| Validation |  29,962 |
| Test       |  29,962 |

## Model Configuration

| Parameter            |                        Value |
| -------------------- | ---------------------------: |
| Model                |                          GIN |
| k-NN neighbors       |                            3 |
| Hidden Dimension     |                          256 |
| Dropout              |                         0.43 |
| Learning Rate        |        0.0028143775705957007 |
| Weight Decay         |       5.9945168393624304e-05 |
| Epochs               |                          200 |
| Trainable Parameters |                      146,443 |
| Loss                 | Class-weighted Cross-Entropy |
| Selection Metric     |          Validation Macro-F1 |

PSO was used for **hyperparameter selection**.

## Final Results

| Metric          |           Score |
| --------------- | --------------: |
| Accuracy        |      **90.19%** |
| Macro Precision |      **88.55%** |
| Macro Recall    |      **87.12%** |
| Macro F1        |      **87.38%** |
| Weighted F1     |      **89.99%** |
| Macro ROC-AUC   |      **99.29%** |
| Inference Time  | **0.00872 sec** |

**Best Validation Macro-F1:** 87.03% at Epoch 200.

## Graph Construction

Separate k-NN graphs were constructed for train, validation, and test sets using **k = 3**.

```text
Train edges      : 952,688
Validation edges : 118,978
Test edges       : 119,590
```

Graph construction time: **163.9 seconds**.


## Key Result

**GIN + PSO achieved 90.19% accuracy and 87.38% Macro-F1 on the held-out CICIoT2023 test set.**
