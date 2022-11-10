# Pruning YOLOv5 :rocket:

:books: This guide explains how to apply **pruning** and **quantization** using SparseML to YOLOv5 :rocket: models.

## :arrow_heading_down: Installation

SparseML, an open-source library that includes tools to create sparse models. SparseML is integrated with
Ultralytics YOLOv5, making it easy to apply SparseML's algorithms to YOLOv5 models.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision] # [XXX] update with the new pathway
```

## 💡 Sparsity Conceptual Overview

Introducing Sparsity to YOLOv5 can improve inference performance, especially when paired with 
an inference runtime that implements sparsity-aware optimizations.

SparseML uses two major techniques to create sparse models:
- **Pruning** systematically removes redundant weights from a network
- **Quantization** reduces model precision by converting weights from `FP32` to `INT8`

Pruning and Quantization work best when performed with access to training data that
allows the model to slowly adjust to the new optimization space as the pathways are removed or
become less precise. We descibe the key training-aware sparsity algorithms below. 

<details>
    <summary><b>:scissors: Pruning: GMP</b></summary>
    <br>
   
Gradual magnitude pruning or **GMP** is the best algorithm for pruning. With it, 
the weights closest to zero are iteratively removed over several epochs or training steps up to a specified level of sparsity. 
The remaining non-zero weights are then fine-tuned to the objective function. This iterative process enables 
the model to slowly adjust to a new optimization space after pathways are removed before pruning again.

SparseML enables you to run GMP on YOLO-v5 with a single command line call.

</details>
        
<details>
    <summary><b>:black_square_button: Quantization: QAT</b></summary>
    <br>

Quantization aware training or **QAT** is the best algorithm for quantization. With it, fake quantization 
operators are injected into the graph before quantizable nodes for activations, and weights 
are wrapped with fake quantization operators. The fake quantization operators interpolate 
the weights and activations down to `INT8` on the forward pass but enable a full update of 
the weights at `FP32` on the backward pass. This allows the model to adapt to the loss of 
information from quantization on the forward pass. 

SparseML enables you to run QAT on YOLO-v5 with a single command line call.
    
</details>
    
For more conceputal details checkout this [blog](https://neuralmagic.com/blog/pruning-overview/).

## :cook: Creating SparseML Recipes

Recipes are YAML files that encode the hyperparameters of the **GMP** 
and **QAT** algorithms. The rest of the SparseML system parses the Recipes to setup the **GMP** and 
**QAT** algorithms.

The easiest way to create a Recipe for usage with SparseML is downloading a pre-made Recipe
from the open-source SparseZoo model repo. SparseZoo has a recipe available for each version of YOLOv5 and 
YOLOv5p. Checkout the model cards [here **UPDATE LINK**](https://sparsezoo.neuralmagic.com/).

Some users may want to tweak a Recipe or create one from scratch. We will explain the `Modifiers` 
used in the recipes for **GMP** and **QAT**. Checkout SparseML's [Recipe User Guide](https://docs.neuralmagic.com/user-guide/recipes/creating) 
for more details on recipes.

>:rotating_light: **Pro-Tip:** the pre-made Recipes in the SparseZoo are very good. If a pre-made Recipe
>for a model already exists (e.g. for YOLOv5-s and YOLOv5-l), you should use the pre-made recipes as starting point
>and tweak as needed.

>:rotating_light: **Pro-Tip #2:** With YOLOv5, you should consider using [sparse transfer learning **UPDATE LINK**](Ultralytics-STL-README.md). 
>It is an easier way to create a sparse model trained on your data.

<details>
    <summary><b>:scissors: GMP Modifiers</b></summary>
    <br>

An example `recipe.yaml` file for GMP is the following:

```yaml
# gmp-recipe.yaml
   
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

30 pruning epochs and 20 finetuning epochs were chosen based on a 50 epoch training schedule - be sure to adjust based on the number of epochs as needed.

</details>

<details>
    <summary><b>🔲 QAT Modifiers</b></summary>
    <br>
    
An example `recipe.yaml` file for QAT is the following:

```yaml
# qat-recipe.yaml
    
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

</details>

<details>
    <summary><b>:palms_up_together: Compound Sparsity: GMP + QAT</b></summary>
    </br>
    
Pruning and quantization can be applied together. When run in a sparsity-aware runtime, the speedup
from pruning and quantization amplify eachother. Here's what a recipe might look like with both GMP and QAT:

```
# recipe.yaml
    
modifiers:
    - !GlobalMagnitudePruningModifier
        init_sparsity: 0.05
        final_sparsity: 0.8
        start_epoch: 0.0
        end_epoch: 30.0
        update_frequency: 1.0
        params: __ALL_PRUNABLE__

    - !SetLearningRateModifier
        start_epoch: 0.0
        learning_rate: 0.05

    - !LearningRateFunctionModifier
        start_epoch: 30.0
        end_epoch: 50.0
        lr_func: cosine
        init_lr: 0.05
        final_lr: 0.001

    - !QuantizationModifier
        start_epoch: 50.0
        freeze_bn_stats_epoch: 53.0

    - !SetLearningRateModifier
        start_epoch: 50.0
        learning_rate: 10e-6

    - !EpochRangeModifier
        start_epoch: 0.0
        end_epoch: 55.0
```
    
</details>

## :fork_and_knife: Applying Recipes to YOLOv5

Once you have created a Recipe or identifed a Recipe in the SparseZoo, you can use the SparseML-YOLOv5 integration 
to kick off the sparsification process with a single command line call.

SparseZoo has pre-made recipes available for every verion of YOLOv5 and YOLOv5p

In this example, we will use dense [XXX] as the starting point and the pre-made sparsification recipe for [XXX]. 
You can use the `sparsezoo_stub` to identify the sparsification recipe:
```
zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
```

The following CLI command downloads the sparsification recipe from the SparseZoo and 
kicks off the sparsification process, fine-tuning onto the COCO dataset:
```
sparseml.yolov5.train \                 # [XXX] << update with new pathway
  --weights [xxx] \
  --data coco.yaml \
  --hyp data/hyps/hyp.scratch.yaml \
  --recipe [zoo-stub]
```

Used the `sparse_zoo` stub in this example, but you can also pass a local path to a `recipe`.

In general, deep neural networks are overparameterized, meaning we can remove weights and reduce
precision with very little loss of accuracy. In this example, we achieve [**XX**]% recovery of the accuracy 
for the dense baseline. The majority of layers are pruned between [**XX**]% and [**XX**]%, with some more 
senstive layers pruned to [**50**]%. On our training run, final accuracy is [**XX**] mAP@0.5 as 
reported by the Ultralytics training script.

## ⤴️ Exporting to ONNX

Many inference runtimes accept ONNX as the input format.

The SparseML installation provides an export script that you can use to load the training model folder 
and create a new `model.onnx` file within. The export process is modified such that the quantized 
and pruned models are corrected and folded properly. Be sure the `--weights` argument points to your trained model.

```
sparseml.yolov5.export_onnx \           # [XXX] << update with new pathway
   --weights path/to/weights.pt \
   --dynamic
```
