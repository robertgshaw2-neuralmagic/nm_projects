# Model Pruning/Sparsity

:books: This guide explains how to apply **pruning** and **quantization** to YOLOv5 :rocket: models.

## Installation

We will utilize SparseML, an open-source library that includes tools to create sparse models. SparseML is integrated with
Ultralytic's YOLOv5, making it easy to apply SparseML's algorithms to YOLOv5 models.

Install SparseML with the following command. We recommend using a virtual enviornment.
```bash
pip install sparseml[torchvision]
```

## Sparsity Conceptual Overview

SparseML generally applies two major techniques to create sparse models:
- **Pruning** systematically removes redundant weights from a network
- **Quantization** reduces model precision by converting weights from `FP32` to `INT8`

**Pruning** and **Quantization** work best when performed with access to training data that
allows the model to slowly adjust to the new optimization space as the pathways are removed or
become less precise. 

We descibe the key training-aware sparsity algorithms below. For more details on the concepts behind
pruning and quantization, follow along on the following [blog](https://neuralmagic.com/blog/pruning-overview/)

### :scissors: Pruning: GMP

Gradual magnitude pruning or **GMP** is the best algorithm for pruning. With it, 
the weights closest to zero are iteratively removed over several epochs or training steps up to a specified level of sparsity. 
The remaining non-zero weights are then fine-tuned to the objective function. This iterative process enables 
the model to slowly adjust to a new optimization space after pathways are removed before pruning again.

SparseML enables you to run GMP on YOLO-v5 with a single command line call.

### :black_square_button: Quantization: QAT

Quantization aware training or **QAT** is the best algorithm. With it, fake quantization 
operators are injected into the graph before quantizable nodes for activations, and weights 
are wrapped with fake quantization operators. The fake quantization operators interpolate 
the weights and activations down to `INT8` on the forward pass but enable a full update of 
the weights at `FP32` on the backward pass. This allows the model to adapt to the loss of 
information from quantization on the forward pass. 

SparseML enables you to run QAT on YOLO-v5 with a single command line call.

## Sparsifying with SparseML: `Recipes`

SparseML is integrated with Ultralytics YOLOv5 and relies on `Recipes`. `Recipes` are YAML or 
YAML front matter markdown files that encode the hyperparameters of the GMP and QAT algorithms.

The easiest way to create a `Recipe` for usage with SparseML is downloading a pre-made `Recipe`
from the SparseZoo model repo ([example: YOLOv5-l model card](https://sparsezoo.neuralmagic.com/models/cv%2Fdetection%2Fyolov5-l%2Fpytorch%2Fultralytics%2Fcoco%2Fpruned_quant-aggressive_95)). 
These `Recipes` were used to create the state-of-the-art sparsified models found in the SparseZoo repo.

However, some users may want to tweak the original `Recipe` or create a `Recipe` from scratch. 
As such, we will explain the `Modifiers` used in the recipes for **GMP** and **QAT**.

### GMP Modifiers

An example `recipe.yaml` file for pruning could do the following:
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

Each modifier encodes a hyperparameter of the **GMP** algorithm.
  - `GlobalMagnitudePruningModifier` applies gradual magnitude pruning globally across all the prunable parameters/weights in a model. It
  starts at `5%` sparsity at epoch `0` and gradually ramps up to `80%` sparsity at epoch `30`, pruning at the start of each epoch.
  - `SetLearningRateModifier` sets the pruning LR to `0.05` (the midpoint between the original 0.1 and 0.001 LRs used to train dense YOLO).
  - `LearningRateFunctionModifier` cycles the finetuning LR from the pruning LR to 0.001 with a cosine curve (0.001 was the final original training LR).
  - `EpochRangeModifier` expands the training time to continue finetuning for an additional 20 epochs after pruning has ended.

`30` pruning epochs and `20` finetuning epochs were chosen based on a `90` epoch training schedule -- be sure to adjust based on the number of epochs as needed.




## Example

In general, deep neural networks are highly overparameterized, meaning we can remove weights and reduce
precision with very little loss of accuracy. In the example below with YOLOv5-l, we set **80%** of weights 
to 0 and shift to `INT8` precison while retain **95%** of the dense model baseline accuracy.



DeepSparse achieves realtime performance on CPUs through two main innovations:
- First, it implements sparse convolutions and matrix multiply operations, reducing the number of FLOPs by skipping the 0s. 
- Second, it uses the CPU’s large fast caches to provide locality of reference, executing the network depthwise and asynchronously.

<p align="center">
  <img src="figure2c-3.svg" alt="Architecture Diagram" width="60%"/>
</p>

These two ideas enable DeepSparse to achieve suprising speedups and run inference in real-time on CPUs.

# DeepSparse Example

We will walk through an example using YOLOv5-l with DeepSparse, following these steps:
- Install DeepSparse
- Collect ONNX File
- Deploy a Model
- Benchmark Latency/Throughput

## :arrow_heading_down: Installation

Run the following. We reccomend you use a virtual enviornment.

```bash
pip install deepsparse[server]
```

> :warning: DeepSparse is tested on Python 3.6-3.9, ONNX 1.5.0-1.10.1, ONNX opset version 11+ and is manylinux compliant. It is limited to Linux systems running on X86 CPU architectures.

## :monocle_face: Collecting an ONNX File

DeepSparse accepts a model in the ONNX format. ONNX files can be generated through the export pathway for models trained/sparsified with Neural Magic's [SparseML](link_to_export_example) or using PyTorch/Keras native exporting functionality.

The `model_path` argument in the commands below tells DeepSparse where the ONNX file is. It can be one of two options:   
- `local_path` to `[model_name].onnx` in a local filesystem 
- `sparsezoo_stub` which identifies a pre-sparsified model in Neural Magic's [SparseZoo](https://sparsezoo.neuralmagic.com).

We will use the sparsified YOLOv5-l from the SparseZoo, which is identified by the following stub:
```
zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
```
## :rocket: Deploying a Model

The DeepSparse package contains two options for deployment: 
- **Python/C++ API:** run inference on the edge or within an application
- **HTTP Server:** create a model service utilizing REST APIs

Pull down a sample image for testing and save as `basilica.jpg` with the following command:
```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

#### :snake: Python/C++ API 

`Pipelines` wrap image pre-processing and output post-processing around the DeepSparse Engine. The DeepSparse-YOLOv5 integration includes an out-of-the-box `Pipeline` that accepts raw images and outputs the bounding boxes.

Create a `Pipeline` for inference with sparse YOLOv5-l using the following Python code:

```python
from deepsparse import Pipeline

# list of images in local filesystem
images = ["basilica.jpg"]

# create Pipeline containing DeepSparse
model_stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95"
yolo_pipeline = Pipeline.create(
    task="yolo",            # do the YOLO pre-processing + post-processing
    model_path=model_stub,  # if using a local model, can pass the local path here
)

# run inference on images, recieve bounding boxes + classes
pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
```

#### :electric_plug: HTTP Server 

Alternatively, DeepSparse offers a server runs on top of the popular FastAPI web framework and Uvicorn web server such that you can query a model via HTTP. The server supports any task from DeepSparse, such as `Pipelines` for object detection tasks.

Spin up the server with sparse YOLOv5-l by running the following from the command line: 

```bash
deepsparse.server \
    task yolo \
    --model_path "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95"
```

An example request, using Python's `requests` package:
```python
import requests
import json

# list of images for inference (local files on client side)
path = ['basilica.jpg'] 
files = [('request', open(img, 'rb')) for img in path]

# send request over HTTP to /predict/from_files endpoint
url = 'http://0.0.0.0:5543/predict/from_files'
resp = requests.post(url=url, files=files)

# response is returned in JSON
annotations = json.loads(resp.text) # dictionary of annotation results
bounding_boxes = annotations["boxes"]
labels = annotations["labels"]
```

## :bar_chart: Benchmarking 

The mission of Neural Magic is to enable GPU-class inference performance on commodity CPUs. Want to find out how fast our sparse YOLOv5 ONNX models perform inference? You can quickly do benchmarking tests on your own with a single CLI command!

You only need to provide the model path of a SparseZoo ONNX model or your own local ONNX model to get started:

``` bash
deepsparse.benchmark \
    zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95 \
    --scenario sync 

>> Original Model Path: zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
>> Batch Size: 1
>> Scenario: sync
>> Throughput (items/sec): 74.0355
>> Latency Mean (ms/batch): 13.4924
>> Latency Median (ms/batch): 13.4177
>> Latency Std (ms/batch): 0.2166
>> Iterations: 741

```
