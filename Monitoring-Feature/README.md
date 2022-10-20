# DeepSparse Logging

DeepSparse Logging enables users to monitor the health of an ML deployment holistically:
- **System Logging** gives operations teams access to granual performance metrics, diagnosing and isolating deployment system health.
- **Data Logging** gives ML teams access to inputs/outputs (and functions thereof) of each stage of an ML pipeline, supporting downsteam model health monitoring tasks.

DeepSparse provides a simple YAML-based configuration setup with many pre-defined metrics and functions in addition to an extensible interface for adding custom metrics using Python.

## Configuration

Logging is configured through YAML-files. 

<details>
    <summary><b>System Logging Configuration</b></summary>
    </br>

System Logging is *enabled* by default. All metrics are [pre-defined](/README.md#system-logging-metrics). Users can disable System Logging globally or at the Group level by adding the following key-value pairs to a configuration file.

Example disabling all System Logging:

```yaml
system_logging: off
```

Example disabling at the Group Level:

```yaml
system_logging:
    deployment_details: off
    request_details: off
    prediction_latency: on 
    dynamic_batch_latency: off
    # resource_utilization: on      << note: omitted groups are turned on by default
```

</details>

<details>
    <summary><b>Data Logging Configuration</b></summary>
    </br>
    
Data Logging is *disabled* by default. Users can log the raw input / output (and functions thereof) at each stage of a `Pipeline`. Many functions have been [pre-defined](link) and users can provide [custom functions](link). 

A `Pipeline` has 4 stages, each of which can be a `target` for data logging:
|Stage      |Pipeline Inputs    |Engine Inputs  |Engine Outputs     |Pipeline Outputs   |
|-----------|-------------------|---------------|-------------------|-------------------|
|Desciption |Inputs passed by user|Tensors passed to the engine|Outputs from the engine (logits)|Postprocessed output returned to user|
|Target     |`pipeline_inputs`  |`engine_inputs`|`engine_outputs`   |`pipeline_outputs` |
    



In the example YAML snippit below, 

```yaml
pipeline_inputs:
- func: builtins/batch-mean                 # pre-defined function
  target: prometheus                        # NOTE: only logs to prometheus
  frequency: 100
- func: /path/to/your/logging_file.py:my_fn # custom function
  frequency: 10
  # target:                                 # NOTE: not specified, logs to all loggers
engine_inputs:
- func: builtins/channel-mean
  frequency: 100
# engine_outputs:                           # NOTE: not specified, so not logged
# pipeline_outputs:
```


</details>

## Usage

Monitoring is configured though a YAML file which is passed to either a `Server` or a `Pipeline`.

<details> 
    <summary><b>Server Usage</b></summary>
    </br>

The DeepSparse server is launched from the CLI using the `deepsparse.server` command. By default, all system logging is enabled in the Prometheus format and exposed on port `8001`

For example, 
```bash
deepsparse.server --config config.yaml
```
</details>

<details> 
    <summary><b>Pipeline Usage</b></summary>
    </br>

`ManagerLogger` is initialized with the `config` argument, which is a path to a local configuration file, and is passed as the `logger` argument to a `Pipeline`. 

For example, with the QA pipeline:

```python
from deepsparse import Pipeline

# SparseZoo model stub or path to ONNX file
model_path = "zoo:nlp/question_answering/bert-base/pytorch/huggingface/squad/12layer_pruned80_quant-none-vnni"

# logger object referencing the local logging config file
logger = ManagerLogger(config="logging-config.yaml")

# pipeline instantiated with the config file
pipeline = Pipeline.create(
    task="question-answering",
    model_path=model_path,
    config="config.yaml"
)

my_name = qa_pipeline(question="What's my name?", context="My name is Snorlax")
```
</details>
   



## System Logging Overview

DeepSparse provides System Logging to enable out-of-the-box monitoring of system performance and health. The following metric groups are enables by default:
- Deployment Details
- Request Details
- Prediction Latency
- Engine Batch Latency
- Resource Utilizaiton

See [below](/README.md#system-logging-metrics) for the detailed list.


### System Logging Metrics

|Group              |Metric           |Metric Name              |Description                              |Granularity    |Usage  |Frequency      |
|-------------------|---------------- |-------------------------|-----------------------------------------|---------------|-------|---------------|
|Deployment Details |Model Name       |`sl_dd_model_name`       |Name of the model running                |Per Pipeline   |All    |1 hour         |
|Deployment Details |CPU Arch         |`sl_dd_cpu_arch`         |Architecture of the CPU                  |Per Server     |All    |1 hour         |
|Deployment Details |CPU Model        |`sl_dd_cpu_model`        |Model of the CPU                         |Per Server     |All    |1 hour         |
|Deployment Details |Cores Used       |`sl_pl_num_cores`        |Number of cores used by the engine       |Per Server     |All    |1 hour         |
|Prediction Latency |Total Time       |`sl_pl_total_time`       |End-to-end prediction time               |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Preprocess Time  |`sl_pl_preprocess_time`  |Time spent in pre-processing step        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Queue Time       |`sl_pl_queue_time`       |Time spent in queue (waiting for engine) |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Engine Time      |`sl_pl_engine_time`      |Time spent in engine forward pass        |Per Pipeline   |All    |Per Prediction |
|Prediction Latency |Postprocess Time |`sl_pl_postprocess_time` |Time spent in post-processing step       |Per Pipeline   |All    |Per Prediction |



