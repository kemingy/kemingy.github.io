+++
title = "Deep Learning Serving Benchmark"
[taxonomies]
categories = ["Life", "Technology"]
+++

There is no black magic, everything follows the rules.

<!-- more -->

## What does the deep learning serving frameworks do?

* respond to request (RESTful HTTP or RPC)
* model inference (with runtime)
* preprocessing & postprocessing (optional)
* queries dynamic batching (increase throughput)
* monitoring metrics
* service health check
* versioning
* multiple instances

Actually, when we are trying to deploy the models with kubernetes, we only need part of these features. But we do care about the performance of these frameworks. So let's do a benchmark.

## Benchmark

**Environments**:

* CPU: Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
* GPU: NVIDIA V100
* Memory: 251GiB
* OS: Ubuntu 16.04.6 LTS (Xenial Xerus)

**Docker Images**:
* tensorflow/tensorflow:latest-gpu
* tensorflow/serving:latest-gpu
* nvcr.io/nvidia/tensorrtserver:19.10-py3

The cost of time is recorded after **warmup**. Dynamic batching **disabled**.

All the code can be found in this [gist](https://gist.github.com/kemingy/a382528b29f6e34c47b464cf16806731).

| Framework | Model | Model Type | Images | Batch size | Time(s) |
| :---:  | :---:    |     :---:   |  :---:  |  :---: | :---: |
| Tensorflow | ResNet50 | TF Savedmodel | 32000 | 32 | 83.189 |
| Tensorflow | ResNet50 | TF Savedmodel | 32000 | 10 | 86.897 |
| Tensorflow Serving  |  ResNet50 | TF Savedmodel  | 32000 | 32 | 120.496 |
| Tensorflow Serving  |  ResNet50 | TF Savedmodel  | 32000 | 10 | 116.887 |
| Triton (TensorRT Inference Server)  |  ResNet50 | TF Savedmodel  | 32000 | 32 | 201.855 |
| Triton (TensorRT Inference Server)  |  ResNet50 | TF Savedmodel  | 32000 | 10 | 171.056 |
| Falcon + msgpack + Tensorflow | ResNet50 | TF Savedmodel  | 32000 | 32 | 115.686 |
| Falcon + msgpack + Tensorflow | ResNet50 | TF Savedmodel  | 32000 | 10 | 115.572 |

According to the benchmark, Triton is not ready for production, TF Serving is a good option for TensorFlow models, and self-host service is also quite good (you may need to implement dynamic batching for production).

## Comparing

### Tensorflow Serving

[https://www.tensorflow.org/tfx/serving](https://www.tensorflow.org/tfx/serving)

* coupled with Tensorflow ecosystem (also support other format, not out-of-box)
* A/B testing
* provide both gRPC and HTTP RESTful API
* prometheus integration
* batching
* multiple models
* preprocessing & postprocessing can be implemented with [signatures](https://github.com/tensorflow/tensorflow/issues/31055)

### Triton Inference Server

[https://github.com/NVIDIA/triton-inference-server/](https://github.com/NVIDIA/triton-inference-server/)

* support multiply backends: ONNX, PyTorch, TensorFlow, Caffe2, TensorRT
* both gRPC and HTTP with SDK
* internal health check and prometheus metrics
* batching
* concurrent model execution
* preprocessing & postprocessing can be done with ensemble models
* `shm-size`, `memlock`, `stack` configurations are not available for Kubernetes

### Multi Model Server

[https://github.com/awslabs/multi-model-server](https://github.com/awslabs/multi-model-server)

* require Java 8
* provide HTTP
* Java layer communicates with Python workers through Unix Domain Socket or TCP
* batching (not mature)
* multiple models
* `log4j`
* management API
* need to write model loading and inference code (means can use any runtime you want)
* easy to add preprocessing and postprocessing to the service

### GraphPipe

[https://oracle.github.io/graphpipe](https://oracle.github.io/graphpipe)

* use flatbuffer which is more efficient
* 2 years ago...
* Oracle laid off the whole team

### TorchServe

[https://github.com/pytorch/serve](https://github.com/pytorch/serve)

* fork from Multi Model Server
* developing...