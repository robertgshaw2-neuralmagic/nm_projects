# Model Pruning/Sparsity

:books: This guide explains how to apply **pruning** and **quantization** using [SparseML](https://github.com/neuralmagic/sparseml) to YOLOv5 :rocket: models.

## :arrow_heading_down: Installation

We will utilize SparseML, an open-source library that includes tools to create sparse models. SparseML is integrated with
Ultralytic's YOLOv5, making it easy to apply SparseML's algorithms to YOLOv5 models.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision]
```

## 💡 Sparsity Conceptual Overview

SparseML generally applies two major techniques to create sparse models:
- **Pruning** systematically removes redundant weights from a network
- **Quantization** reduces model precision by converting weights from `FP32` to `INT8`

Pruning and Quantization work best when performed with access to training data that
allows the model to slowly adjust to the new optimization space as the pathways are removed or
become less precise. We descibe the key training-aware sparsity algorithms below. 

#### :scissors: Pruning: GMP

Gradual magnitude pruning or **GMP** is the best algorithm for pruning. With it, 
the weights closest to zero are iteratively removed over several epochs or training steps up to a specified level of sparsity. 
The remaining non-zero weights are then fine-tuned to the objective function. This iterative process enables 
the model to slowly adjust to a new optimization space after pathways are removed before pruning again.

SparseML enables you to run GMP on YOLO-v5 with a single command line call.

#### :black_square_button: Quantization: QAT

Quantization aware training or **QAT** is the best algorithm. With it, fake quantization 
operators are injected into the graph before quantizable nodes for activations, and weights 
are wrapped with fake quantization operators. The fake quantization operators interpolate 
the weights and activations down to `INT8` on the forward pass but enable a full update of 
the weights at `FP32` on the backward pass. This allows the model to adapt to the loss of 
information from quantization on the forward pass. 

SparseML enables you to run QAT on YOLO-v5 with a single command line call.

For more conceputal details checkout this [blog](https://neuralmagic.com/blog/pruning-overview/).

## :cook: Creating SparseML Recipes

Recipes are YAML or YAML front matter markdown files that encode the hyperparameters of the **GMP** 
and **QAT** algorithms. The rest of the SparseML system parses the Recipes to setup the **GMP** and 
**QAT** algorithms.

The easiest way to create a Recipe for usage with SparseML is downloading a pre-made Recipe
from the SparseZoo model repo ([example: YOLOv5-l model card](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-l%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned_quant-aggressive_95)). 
These Recipes were used to create the state-of-the-art sparsified models found in the SparseZoo repo.

However, some users may want to tweak the original Recipe or create a Recipe from scratch. 
As such, we will explain the `Modifiers` used in the recipes for **GMP** and **QAT**.

>:rotating_light: **Pro-Tip:** the pre-made Recipes in the SparseZoo are very good. If a pre-made Recipe
>for a model already exists (e.g. for YOLOv5-s and YOLOv5-l), you should use the pre-made recipes as starting point
>and tweak as needed.

#### :scissors: GMP Modifiers

An example `recipe.yaml` file for GMP is the following:

```yaml
modifiers:
    !GlobalMagnitudePruningModifier
        init_sparsity: 0.05
        final_sparsity: 0.8
        start_epoch: 0.0
        end_epoch: 30.0
        update_frequency: 1.0
        params: __ALL_PRUNABLE__

    !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.05

    !LearningRateFunctionModifier
        start_epoch: 30.0
        end_epoch: 50.0
        lr_func: cosine
        init_lr: 0.05
        final_lr: 0.001

    !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 50.0
```

Each `Modifier` encodes a hyperparameter of the **GMP** algorithm:
  - `GlobalMagnitudePruningModifier` applies gradual magnitude pruning globally across all the prunable parameters/weights in a model. It
  starts at 5% sparsity at epoch 0 and gradually ramps up to 80% sparsity at epoch 30, pruning at the start of each epoch.
  - `SetLearningRateModifier` sets the pruning LR to 0.05 (the midpoint between the original 0.1 and 0.001 LRs used to train YOLO).
  - `LearningRateFunctionModifier` cycles the LR from 0.5 to 0.001 with a cosine curve (0.001 was the final original training LR).
  - `EpochRangeModifier` expands the training time to continue finetuning for an additional `20` epochs after pruning has ended.

30 pruning epochs and 20 finetuning epochs were chosen based on a 50 epoch training schedule -- be sure to adjust based on the number of epochs as needed.

#### 🔲 QAT Modifiers

An example `recipe.yaml` file for QAT is the following:

```yaml
modifiers:
    !QuantizationModifier
        start_epoch: 0.0
        submodules: ['model']
        freeze_bn_stats_epoch: 3.0

    !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 10e-6

    !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 5.0
```

Each `Modifier` encodes a hyperparameter of the **QAT** algorithm:  

  - The `QuantizationModifier` applies QAT to all quantizable modules under the `model` scope.
Note the `model` is used here as a general placeholder; to determine the name of the root module for your model, print out the root module and use that root name.
  - The `QuantizationModifier` starts at epoch 0 and freezes batch normalization statistics at the start of epoch 3.
  - The `SetLearningRateModifier` sets the quantization LR to 10e-6 (0.01 times the example final LR of 0.001).
  - The `EpochRangeModifier` sets the training time to continue training for the desired 5 epochs.

## :fork_and_knife: Applying SparseML Recipes to YOLOv5

Once you have created a Recipe or identifed a Recipe in the SparseZoo, you can use the SparseML-YOLOv5 integration 
to kick off the sparsification process with a single command line call.

In this example, we will use dense YOLOv5-l from the SparseZoo as the starting point and the pre-made sparsification recipe for YOLOv5-l. 
They are identified by the following `sparsezoo_stubs`:

```
zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none
```

The following CLI command kicks off the sparsification process, fine-tuning onto the COCO dataset:

```
sparseml.yolov5.train \
  --weights zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/base-none \
  --data coco.yaml \
  --hyp data/hyps/hyp.scratch.yaml \
  --recipe zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
```

#### MARK: PLEASE ADD COMMENTS HERE IF THERE IS ANY COMPLEXITY WITH SPECIFYING WEIGHTS #####
We have used `sparse_zoo` stubs in this example, but you can also pass `local_path` to `--weights` if you
want to use a different baseline model and to `--recipe` if you want to use a custom recipe.

In general, deep neural networks are overparameterized, meaning we can remove weights and reduce
precision with very little loss of accuracy. In this example, we achieve 95% recovery of the accuracy 
for the dense baseline. The majority of layers are pruned between 65% and 85%, with some more senstive 
layers pruned to 50%. On our training run, final accuracy is 62.3 mAP@0.5 as reported by the Ultralytics training script.

## ⤴️ Exporting the Sparse Model to ONNX

Many inference runtimes accept ONNX as the input format.

The SparseML installation provides a sparseml.yolov5.export_onnx command that you can use to load the training model folder and create a new model.onnx file within. The export process is modified such that the quantized and pruned models are corrected and folded properly. Be sure the `--weights` argument points to your trained model.

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
