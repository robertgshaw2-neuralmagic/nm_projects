# DeepSparse Monitoring

DeepSparse monitoring enables users to monitor the health of an ML deployment holistically:
- **System Logging** gives operations teams access to granual performance metrics, diagnosing and isolating deployment system health.
- **Data Logging** gives ML teams access to inputs/outputs (and functions thereof) of each stage of an ML pipeline, supporting downsteam model health monitoring tasks.

DeepSparse provides a simple YAML-based configuration setup with many pre-defined metrics and functions + a simple interface to add custom metrics using Python.

## System Logging Overview

DeepSparse provides System Logging to enable out-of-the-box monitoring of system performance and health. The following metric groups are enables by default:
- Deployment Details
- Request Details
- Prediction Latency
- Engine Batch Latency
- Resource Utilizaiton

See [below](/README.md#system-logging-metrics) for the detailed list.


## System Logging Metrics

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



