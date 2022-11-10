# YOLOv5 :rocket: Deployments on CPUs with DeepSparse

:books: Learn how to deploy YOLOv5 with **realtime latency on CPUs** utilizing Neural Magic's DeepSparse:bangbang: 

## DeepSparse Overview

DeepSparse runs inference-optimized sparse models with GPU-class performance on commodity CPUs.

By deploying perfomantly on CPUs, you can simplify your deployment with the simplicity and scalability of software-defined inference:
- Scale vertically from 2 to 192 cores, tailoring the footprint to an app's exact needs
- Scale horizontally with standard Kubernetes, including using services like EKS/GKE
- Deploy the same model/runtime on any hardware from Intel to AMD to ARM and from cloud to data center to edge, including on pre-existing systems
- No wrestling with drivers, operator support, and compatibility issues

With DeepSparse, you no longer need to pick between the performance of GPUs and the simplicity of software!

See [Sparse Transfer Learning with YOLOv5 - UPDATE LINK](link) or [Sparsifying YOLOv5 - UPDATE LINK](link) for more details.

## Example Deployment

We will walk through an example deploying a sparse version of [XXX] with DeepSparse, following these steps:
- Install DeepSparse
- Collect ONNX File
- Deploy a Model
- Benchmark Latency/Throughput

Pull down a sample image for the example and save as `basilica.jpg` with the following command:
```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

### :arrow_heading_down: Install DeepSparse

Run the following. We recommend you use a virtual enviornment.

```bash
pip install deepsparse[server]
```

### 🔍 Collect an ONNX File

DeepSparse accepts a model in the ONNX format.

The `model_path` argument in the commands below tells DeepSparse where the ONNX file is. It can be one of two options:   
- `sparsezoo_stub` which identifies a pre-sparsified model in [SparseZoo](https://sparsezoo.neuralmagic.com)
- `local_path` to `[model_name].onnx` in a local filesystem. For models trained/sparsified with SparseML, ONNX files can be generated through the [export pathway - UPDATE LINK](link.md#4-exporting-to-onnx)

In the example below, we will use the **pruned-quantized** [XXX] from the SparseZoo, identified by the following stub:
```
zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned_quant-aggressive_95
```
### :rocket: Deploy a Model

DeepSparse contains two options for deployment: 

<details>  
  <summary><b>Python API:</b> run inference on the client side or within an application </summary>
  <br>
  
  `Pipelines` wrap image pre-processing and output post-processing around the DeepSparse Engine. The DeepSparse-Ultralytics integration includes an out-of-the-box `Pipeline` that accepts raw images and outputs the bounding boxes.

  Create a `Pipeline` for inference with sparse [XXX] using the following Python code:

  ```python
  from deepsparse import Pipeline

  # list of images in local filesystem
  images = ["basilica.jpg"]

  # create Pipeline containing DeepSparse
  model_stub = "xxx"
  yolo_pipeline = Pipeline.create(
      task="yolo",            # do the YOLO pre-processing + post-processing
      model_path=model_stub,  # if using a local model, can pass the local path here
  )

  # run inference on images, recieve bounding boxes + classes
  pipeline_outputs = yolo_pipeline(images=images, iou_thres=0.6, conf_thres=0.001)
  ```
</details>

<details>
  <summary><b>HTTP Server:</b> easily setup a model service behind a REST API</summary>
  <br>
  
  DeepSparse offers a server runs on top of the popular FastAPI web framework and Uvicorn web server such that you can query a model via HTTP. 
  The server supports any task from DeepSparse, such as object detection.

  Spin up the server with sparse [XXX] by running the following from the command line: 

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
</details>

### :bar_chart: Benchmark Performance
You can quickly benchmarking performance with a single CLI command!

Let's demonstrate DeepSparse's performance speedup on sparse [XXX] vs dense [XXX].

#### Dense Performance
``` bash
deepsparse.benchmark \
    xxx \
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

#### Sparse Performance
``` bash
deepsparse.benchmark \
    xxx \
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

As you can see, DeepSparse gained an **[XXX]** speedup from running the sparse model!

Run the following to see the full list of scenarios. We recommend playing with the `--engine` option to compare to DeepSparse to 
ONNXRuntime, the `--scenario` to try out multi or single stream inference, and `--batch-size` to test out throughput vs latency.

```bash
deepsparse.benchmark --help

>> usage: deepsparse.benchmark [-h] [-b BATCH_SIZE] [-i INPUT_SHAPES]
>>                            [-ncores NUM_CORES] [-s {async,sync,elastic}]
>>                            [-t TIME] [-w WARMUP_TIME] [-nstreams NUM_STREAMS]
>>                            [-pin {none,core,numa}]
>>                            [-e {deepsparse,onnxruntime}] [-q]
>>                            [-x EXPORT_PATH]
>>                            model_path
```

#### Performance Chart

The chart below demonstrates the performance speedup with DeepSparse comparing the baseline dense model (standard YOLOv5)
to the inference optimized sparse models. We can see that the sparse models consistently enable you to gain a jump in performance
with very limited sacrifice on accuracy.

|Model                    |YOLOv5-n |YOLOv5-s |YOLOv5-m |YOLOv5-l |YOLOv5-x |
|-------------------------|---------|---------|---------|---------|---------|
|Dense Accuracy (mAP@50)  |  
|Dense Throughput         |
|Dense Latency            |
|Sparse Accuracy (mAP@50) |
|Sparse Throughput        |
|Sparse Latency           |
|**Accuracy % Baseline**  |
|**Throughput Increase**  |

*Note: In this example, we used **XXX** machine with **YYY** setting.*

## Get Started With DeepSparse

🆓 **Research or Testing?** DeepSparse Community is free for research and testing. Production deployments require DeepSparse Enterprise.

🧪 **Want to Try DeepSparse Enterprise?** Neural Magic has a [60 day free trail](link_to_trial_page) for DeepSparse Enterprise.

🛒 **Want To Purchase A Subscription?** [See Pricing](pricing_page) or [Contact sales](link_to_contact_sales) to purchase DeepSparse Enterprise.
