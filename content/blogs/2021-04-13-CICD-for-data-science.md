+++
title = "CI/CD for data science"
[taxonomies]
categories = ["Technology", "Data Science"]
+++

Lots of people are talking about MLOps recently. Continuous Integration (CI) and Continuous Delivery/Deployment (CD) are the basic requirements for MLOps. This post will focus on some details about CI/CD for data science.

<!-- more -->

First of all, I want to declare that data science here means training machine learning (including deep learning) models and serving online requests. Most of the parts may look similar to the traditional backend services.

## Code

Usually, the deep learning models require batch inference to fully utilize the GPU resources. I know there are lots of frameworks, for example, TensorFlow serving, Nvidia triton serving, AWS multi-model serving, etc. I do have built a better one (in my opinion). Or maybe you are still using Flask/FastAPI/Falcon to serve the models. It doesn't matter for the CI/CD part.

Logging is essential for online services. This one needs to be compatible with your log collectors. The most user-friendly way is sending JSON format logs to `stderr`. So the user doesn't need to take care of log format, log file rotation, multiprocessing logging.

Another important part is the metrics monitoring. The ingress or service mesh may already have traffic monitoring. But the services need to collect more detailed metrics for monitoring and debugging. For example, you may want to know some distribution of the user requests and model batch inference time for different batch sizes.

Apart from that, machine learning models may need to collect some feedback from the online services for future training or debugging. This requirement is similar to logging.

To deploy the services into a cluster, the health check is fundamental but doesn't get enough attention. For example, if you are using Kubernetes, usually you will need to provide the liveness and readiness probe. One is for the web service and another is for the model inference part. The health check can help us know the status of all the services so the load balancer can route the requests to the health services.

## Dockerfile

I think there are lots of good materials talking about the best practices for the Dockerfile. I would like to list some of them:

* [Dockerfile best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
* [Production-ready Docker packaging](https://pythonspeed.com/docker/)
* [Intro Guide to Dockerfile Best Practices](https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/)
* [Docker Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

For the GPU container, I prefer to build from the CUDA runtime base image and install all the GPU-related libraries with miniconda. You may run into a lot of `xxx.so not found` errors. Remember, `ldconfig` can solve most of them unless the libraries are not installed.

One thing you need to pay attention to is the first PID in the container. Lots of people are using shell scripts to start the services. In this case, the first PID process will be the shell command instead of the real services. But when we need to terminate the service gracefully, the shell command will receive the `SIGTERM` signal and it won't forward the signal to the child processes. If it doesn't shut down in the pre-defined timeout, all the processes in the container will receive the `SIGKILL` signal.

There are some tools to handle this problem. But none of them are perfect. The most important thing is that your code should handle the graceful shutdown and forward the signal to its children processes.

* tini: https://github.com/krallin/tini
* exec: https://stackoverflow.com/a/18351547


## Kubernetes

In fact, Kubernetes is the container scheduling and orchestration standard. The cluster is maintained by the SRE. But it's good to know how does everything works.

Although some tools (NVIDIA A100 or [vgpu_unlock](https://github.com/DualCoder/vgpu_unlock)) can enable GPU virtualization, most of the time one container will take one GPU. This means the container should consume as much as it can to increase resource utilization.

When you found that you cannot reach the target services, you may need to check the ingress settings. For deep learning models, the request may reach the body size limitation. Getting familiar with the load balancer you are using can reduce a lot of effort for debugging.

For machine learning services, we need to version both the code and the model. The code part is easy, just use Git. But it's a bad idea to use Git Large File Storage (Git LFS) to version the model. The Git storage is not designed to store such kind of large files (hundreds of megabytes or more for each version). The maintainer may send you a warning email : (.

For now, I use special Git tags to trace the model version. When I need to update the corresponding model version with the code commit, I will create a Git tag like `model-v1.0` and CI will help me package the new model from HDFS to the container registry. During the CD, the script will use the latest tag for the model image. This decoupled the code and model.

* [Using Kubernetes Init Containers to decouple the deployment of machine learning applications from their models](https://medium.com/@christianberzi/using-kubernetes-init-containers-to-decouple-the-deployment-of-machine-learning-applications-from-1d557ad52b99)

## Automation

Depends on which tools you are using, for example, Jenkins, GitLab CI Runner, GitHub Actions. I have to say that YAML is annoying. Hope there can be a better config format.

## Summarization

I'm glad to see that there are lots of new tools coming out. Data science is a new and fast-growing area. MLOps is going to be an important part of all the companies that require machine learning solutions. We can still benefit a lot from the backend experiences. But we also need to explore the possibility of new tools to empower the data science development and deployment procedure.

## References

* https://martinfowler.com/articles/cd4ml.html
* https://manjusaka.itscoder.com/posts/2021/02/28/damn-the-init-process/