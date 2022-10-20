# DeepSparse Logging

DeepSparse Logging enables users to monitor the health of an ML deployment holistically:
- **System Logging** gives operations teams access to granual performance metrics, diagnosing and isolating deployment system health.
- **Data Logging** gives ML teams access to inputs/outputs (and functions thereof) of each stage of an ML pipeline, supporting downsteam model health monitoring tasks.

DeepSparse provides a simple YAML-based configuration setup with many pre-defined metrics and functions in addition to an extensible interface for adding custom metrics using Python.

## Configuration

Logging is configured through YAML-files. 
- System Logging is *enabled* by default, and the YAML file is used to disable groups of system metrics
- Data Logging is *disabled* by defualt, and the YAML file is used to specify which data (or functions thereof) should be logged

The configuration file looks like this:
```yaml
loggers
```

<details>
    <summary><b>System Logging Configuration</b></summary>
    </br>

System Logging is *enabled* by default. All metrics are [pre-defined](/README.md#system-logging-metrics). Users can disable System Logging globally or at the Group level by adding the following key-value pairs to a configuration file.

Example YAML snippit disabling all System Logging:

```yaml
system_logging: off
```

Example YAML snippit disabling at the Group Level:

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
    
Data Logging is *disabled* by default. Users can log functions of the inputs/outputs at each of the 4 stages of a `Pipeline`:

|Stage      |Pipeline Inputs    |Engine Inputs  |Engine Outputs     |Pipeline Outputs   |
|-----------|-------------------|---------------|-------------------|-------------------|
|Description|Inputs passed by user|Tensors passed to engine|Outputs from engine (logits)|Postprocessed output returned to user|
|`stage_id` |`pipeline_inputs`  |`engine_inputs`|`engine_outputs`   |`pipeline_outputs` |
    
The following format is used to apply a list of [pre-defined](link) and/or [custom functions](link) to a Pipeline Stage:
 
```yaml
stage_id:
  # first function
  - func:      # [REQUIRED STR] function identifier (built-in or path to custom)
    frequency: # [OPTIONAL INT] logging frequency (default: 1000 - logs once per 1000 predictions)
    target:    # [OPTIONAL STR] logger (default: all)
  # second function
  - func:
    frequency:
    target:
 ...
}
```

A tangible example YAML snippit is below:

```yaml
pipeline_inputs:
  - func: builtins/identity                   # pre-defined function (logs raw data)
    target: prometheus                        # only logs to prometheus
    frequency: 100                            # logs raw data once per 100 predictions
  - func: /path/to/logging_fns.py:my_fn       # custom function
    # frequency:                              # not specified, defaults to once per 1000 predictions
    # target:                                 # not specified, defaults to all loggers

engine_inputs:
  - func: builtins/channel-mean               # pre-defined function (logs per channel mean pixel)
    frequency: 10                             # logs channel-mean once per 10 predictions
    # target:                                 # not specified, defaults to all loggers

# engine_outputs:                             # not specified, so not logged
# pipeline_outputs:                           # not specified, so not logged
```
This configuration does the following at each stage of the Pipeline:
- *Pipeline Inputs*: Raw data (from the `identity` function) is logged to Prometheus once every 100 predictions and a custom function called `my_fn` is applied once every 1000 predictions and is logged to all loggers.
- *Engine Inputs*: The `channel-mean` function is applied once per 10 predictions and is logged to all loggers.
- *Engine Outputs*: No logging occurs at this stage
- *Pipeline Outputs*: No logging occurs at this stage

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



