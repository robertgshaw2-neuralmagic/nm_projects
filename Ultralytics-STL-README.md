# Sparse Transfer Learning with YOLOv5 :rocket:

:books: This guide explains how to fine-tune a sparse YOLOv5 :rocket: onto a custom dataset with [SparseML](https://github.com/neuralmagic/sparseml).

## :arrow_heading_down: Installation

We will utilize SparseML, an open-source library that includes tools to create sparse models. SparseML is integrated with
Ultralytics, allowing you to apply sparse transfer learning to YOLOv5 with a single CLI command.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision]
```

## 💡 Conceptual Overview

Sparse transfer learning is the **easiest and best** way to create a sparse YOLOv5 model trained on custom data. 
                                                                                                            
Similiar to typical transfer learning, sparse transfer learning starts with a sparse model pre-trained on a large dataset 
and fine-tunes the weights onto a smaller downstream dataset. Unlike typical fine-tuning, however, sparse transfer learning, enforces
the sparsity structure of the starting model. In other words, the final trained model will have the same level of sparsity as the original model.
                                                                                                            
>:rotating_light: **Clarification:** When we say sparse models, we are describing sparsity in the **weights** of the model. 
With proper pruning algorithms you can set ~75% of YOLOv5-l weights to 0 and retain 95% of the dense model's accuracy. 
See [Sparsifying YOLOv5](Ultralytics-Sparsify-README.md) for more details.

Sparse Transfer Learning saves you the GPU-hours and hyperparemeter expertise needed to create a sparse YOLOv5 from scratch.

## :mag_right: How It Works

There are four simple steps to sparse transfer learning with SparseML:
1. Select a Pre-Sparsified Model
2. Create Dataset
3. Run Sparse Transfer Learning Algorithm
4. Export to ONNX

## 1. Select a Pre-Sparsified Model

[SparseZoo](https://sparsezoo.neuralmagic.com/?domain=cv&sub_domain=detection&page=1)
is an open-source repository of state-of-the-art pre-sparsified models, including YOLOv5-s and YOLOv5-l. 

In this example, we will use the **pruned-quantized YOLOv5-l**. The majority of layers are pruned between 65% and 85% and it 
achieves 95% recovery of the performance for the dense baseline on COCO. 

It is identifed by the following SparseZoo stub:
```
zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
```

## 2. Create Dataset

The [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#1-create-dataset) tutorial has a detailed 
explanation on creating a custom dataset. SparseML is integrated with Ultralytics and accepts data in the same format.

The dataset config file defines (1) the dataset root directory path and relative paths to train / val / test image directories (or \*.txt files with image paths) and (2) a class names dictionary.

SparseML contains an example for VOC ([voc.yaml](https://github.com/neuralmagic/sparseml/blob/ddfe45b6fa2722c9942300af684a618641eceb0d/src/sparseml/yolov5/data/VOC.yaml)), including
a script to download the dataset (which is omitted below):

```
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: datasets/VOC
train: # train images (relative to 'path')  16551 images
  - images/train2012
  - images/train2007
  - images/val2012
  - images/val2007
val: # val images (relative to 'path')  4952 images
  - images/test2007
test: # test images (optional)
  - images/test2007
  
# Classes
nc: 20  # number of classes
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names
```

## 3. Run Sparse Transfer Learning with YOLOv5

SparseML is integrated with Ultralytics, so you can kick off a training run with a simple CLI command (`sparseml.yolov5.train`) that 
accepts a dataset, a base pre-sparsified model, and a transfer learning recipe.

### :cook: Transfer Learning Recipes

SparseML uses **Recipes** to encode the the hyperparameters of the sparse transfer learning process. SparseZoo has pre-made transfer learning recipes for YOLOv5-s and YOLOv5-l off-the-shelf. 
>:rotating_light: **Pro Tip:** Most should use the off-the-shelf recipes for transfer learning (or slightly tweak if needed).

You can see detail on **Recipes** in the [Pruning YOLOv5 Tutorial](Ultralytics-Sparsify-README.md#cook-creating-sparseml-recipes) if interested. 

For sparse transfer learning, the key **Modifiers** in the recipe are:
- `ConstantPruningModifier` which instructs SparseML to maintain the starting sparsity level as it fine-tunes
- `QuantizationModifier` which instructs SparseML to quantize the model 

For example, in the [YOLOv5-l transfer learning recipe](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-l%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned_quant-aggressive_95), the following lines are included in the recipe:

```
pruning_modifiers:
  - !ConstantPruningModifier
    start_epoch: 0.0
    params: __ALL_PRUNABLE__
    
quantization_modifiers:
  - !QuantizationModifier
    start_epoch: eval(quantization_start_epoch)
    submodules: [ 'model.0', 'model.1', 'model.2', 'model.3', 'model.4', 'model.5', 'model.6', 'model.7', 'model.8', 'model.9', 'model.10', 'model.11', 'model.12', 'model.13', 'model.14', 'model.15', 'model.16', 'model.17', 'model.18', 'model.19', 'model.20', 'model.21', 'model.22', 'model.23' ]
```
### 🏋️: Training The Model

As an example, we will sparse transfer learn **pruned-quantized YOLOv5-l** (which was trained on COCO) onto the VOC dataset. The `train`
command downloads the VOC dataset (using the download script from `VOC.yaml`) and kicks off transfer learning using the recipe defined the SparseZoo.
While this example uses SparseZoo stubs as the `weights` and `recipe` parameters, you can also pass paths to a local YOLOv5 model / recipe as needed.

```bash
sparseml.yolov5.train \
    --data VOC.yaml \
    --cfg models_v5.0/yolov5l.yaml \
    --hyp data/hyps/hyp.finetune.yaml \
    --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95?recipe_type=transfer \
    --recipe zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95?recipe_type=transfer
```
  - `--data` is the config file for the dataset
  - `--cfg` / `--hyp` are parameters that do **MARK: XXXXX**
  - `--weights` identifies the base pre-sparsfied model for the transfer learning. It can be a SparseZoo stub or a path to a local model
  - `--recipe` identifies the transfer learning recipe. It can be SparseZoo stub or a path to a local recipe

‼️ Once the training is finished, you will have a pruned-quantized YOLOv5-l fine-tuned on VOC.

Since SparseML is integrated with Ultralytics, you get all of the same [outputs and visualizations](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#4-visualize) from typical YOLOv5 training.

## 4. Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

The SparseML installation provides a `sparseml.yolov5.export_onnx` command that you can use to export. The export process 
ensures that the quantized and pruned models properly tranlated to ONNX. Be sure the `--weights` argument points to your trained model.

```
sparseml.yolov5.export_onnx \
   --weights path/to/weights.pt \
   --dynamic
```

## 🚀 Deploying YOLOv5 on CPUs for Performance

The resulting pruned-quantized YOLOv5-l model is now only 11MB vs the original dense model at 143MB. However, because we pruned in an unstructured manner 
(where any weight can be set to 0, not just groups of weights), inference runtimes (especially GPU-based inference runtimes) will be unlikely to get 
much of a performance speedup. To take advantage of unstructured sparsity, you must deploy on a sparsity-aware inference runtime.

See [YOLOv5's integration with a DeepSparse](Ultralytics-DeepSparse-README.md) for more details on speedups with sparsity-aware runtimes.
