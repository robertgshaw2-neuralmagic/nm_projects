# Sparse Transfer Learning

:books: This guide explains how to fine-tune a sparse YOLOv5 :rocket: onto a custom dataset with [SparseML](https://github.com/neuralmagic/sparseml).

## :arrow_heading_down: Installation

We will utilize SparseML, an open-source library that includes tools to create sparse models. SparseML is integrated with
Ultralytics, making it easy to apply SparseML's algorithms to YOLOv5 models.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision]
```

## 💡 Sparse Transfer Learning Conceptual Overview

Sparse transfer learning is the best way to create a sparse YOLOv5 model trained on custom data. 
                                                                                                            
Similiar to typical transfer learning, sparse transfer learning starts with a sparse model pre-trained on a large dataset 
and fine-tunes the weights onto a smaller downstream dataset. Unlike typical fine-tuning, however, sparse transfer learning, enforces
the sparsity structure of the starting model. In other words, the final trained model will have the same level of sparsity as the original model.
                                                                                                            
>:rotating_light: **Clarification:** When we say sparse models, we are describing sparsity in the **weights** of the model. 
With proper pruning algorithms you can set ~75% of YOLOv5-l weights to 0 and retain 95% of the dense model's accuracy. 
See [Sparsifying YOLOv5](Ultralytics-Sparsify-README.md) for more details.

By using Sparse Transfer Learning, you can save the time, GPU-hours, and hyperparemeter-tuning needed to create a sparse YOLOv5 trained
on your dataset.

                                                                                                            
[SparseZoo](https://sparsezoo.neuralmagic.com) is a repository of state-of-the-art pre-sparsified models, including YOLOv5-s and YOLOv5-l.                                                                                                      

[SparseML](https://github.com/neuralmagic/sparseml) is a library containing (among other things) the sparse transfer learning algorithm. 
           
SparseML and SparseZoo are integrated with Ultralytics                                                                                                            


                                                                                                                   

## 🚀 Deploying YOLOv5 on CPUs for Performance

The resulting pruned-quantized YOLOv5-l model is now only 11MB vs the original dense model at 143MB. However, because we pruned in an unstructured manner 
(where any weight can be set to 0, not just groups of weights), inference runtimes (especially GPU-based inference runtimes) will be unlikely to get 
much of a performance speedup. To take advantage of unstructured sparsity, you must deploy on a sparsity-aware inference runtime.

See [YOLOv5's integration with a DeepSparse](Ultralytics-DeepSparse-README.md) for more details on speedups with sparsity-aware runtimes.
