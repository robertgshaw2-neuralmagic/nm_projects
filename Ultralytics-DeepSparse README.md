# Neural Magic DeepSparse Engine

:woman_student: :man_student: Learn how to deploy YOLOv5 with **realtime latency on CPUs** utilizing Neural Magic's DeepSparse Engine :bangbang: 

UPDATED September 2022

## Overview

DeepSparse is an inference runtime and server (similiar in concept to NVIDIA's TensorRT+Triton) which runs sparse models with GPU-level performance on CPUs.

>:warning: Clarification: When we speak about sparse models, we are describing sparsity in the **weights** of the model. YOLOv5 can be pruned 80% and retain
99% of the accuracy of the dense model. See [here](xxx) for more details on creating a sparse YOLOv5.


DeepSparse achieves realtime performance on CPUs through two main innovations:
- First, it implements sparse convolutions and matrix-multiply operations, skipping the 0s and reducing the number of FLOPs. 
- Second, it uses the CPU’s large fast caches to provide locality of reference, executing the network depthwise and asynchronously.

Using DeepSparse, you can simplify your deployment process by running on commodity hardware and save signficant costs in deployments from cloud to edge.

Checkout a demo of our software running YOLOv5 in realtime on a 4 core laptop on [YouTube](https://www.youtube.com/watch?v=gGErxSqf05o).

# Usage

