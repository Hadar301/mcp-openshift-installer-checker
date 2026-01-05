# NeMo Microservices Installation Analysis

**Repository:** https://github.com/RHEcosystemAppEng/NeMo-Microservices

## Requirements Summary

### Hardware

- GPU nodes with NVIDIA A10G (or similar)
- At least 1 GPU per training workload
- 32-64 Gi memory for training jobs
- 4-8 CPU cores for training jobs
- Storage: 50Gi+ for models, 10Gi+ for workspace

### Software Prerequisites

- OpenShift 4.x cluster
- Helm 3.x
- `oc` CLI
- NGC API key (NVIDIA container registry access)
- Storage class: `gp3-csi` (default)

### Key Components

- PostgreSQL databases (5 instances for various services)
- MLflow, Argo Workflows, Milvus, Volcano scheduler
- NeMo Operator + NIM Operator
- KServe InferenceService support

## Cluster Status

| Resource | Available | Required |
|----------|-----------|----------|
| Nodes | 13 | ✅ Sufficient |
| CPU | 138.5 cores | ✅ Sufficient |
| Memory | 389.5 Gi | ✅ Sufficient |
| GPUs | 4 (NVIDIA-A10G) | ✅ Sufficient |
| Storage Class | gp3-csi (default) | ✅ Available |

## Feasibility

**Yes** — The cluster has sufficient resources (4 GPUs, 13 nodes, adequate CPU/memory) and required storage classes to install NeMo Microservices.

