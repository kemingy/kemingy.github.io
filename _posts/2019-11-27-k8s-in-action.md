---
title: Kubernetes in action note
categories: [Technology]
---

k8s is really complex...

<!-- more -->

### How Linux make container possible

*   Linux namespace
*   Linux control groups (cgroups) [=systemd]

### Notes

*   Container images are composed of layers, which can be shared and reused across multiple images.

**Docker command and arguments**

*   ENTRYPOINT: ["python", "app.py"]
*   CMD: ["-w", "4"]

entrypoint is the default part, cmd can be override.

## Kubernetes

### Components

#### Master

*   API server
*   Scheduler
*   Controller manager
*   etcd

#### Worker

*   Container runtime
*   Kubelet
*   kube-proxy

### Type

*   ClusterIP: internal network
*   LoadBalancer: external access

### Services

*   solve pods' IP problem

### When to use multiple containers in a Pod?

*   Do they need to be run together or can they run on different hosts?
*   Do they represent a single whole or are they independent components?
*   Must they be scaled together or individually?

**Namespace**

Does **not** offer isolation for nodes or network(depends on the networking solution deployed with k8s).

**Liveness Probes**

spec.containers[0].livenessProbe

*   HTTP GET probe
*   TCP Socket probe
*   Exec probe

If liveness probe failed, the pod will be terminated.

**Readiness Probes**

*   Exec
*   HTTP GET
*   TCP Socekt

If Readiness probe failed, the pod will be removed from endpoints.

**ReplicationController**

keep pods running

*   label selector
*   replica count
*   pod template

pods can be removed from the controller by changing the label

**ReplicaSet**

similar to ReplicationController but more powerful in labels matching

Always use ReplicaSet instead of ReplicationController.

**DaemonSet**

To run a pod on all cluster nodes. (even the unschedulable node)

**Job**

Terminate the pod when job is done successfully.

spec.template.spec.restartPolicy:

*   Always (default, need to change)
*   OnFailure
*   Never

sequentially: spec.completions: n (run n jobs)

parallel: spec.parallelism: n (run n jobs parallel)

**CronJob**

cron job for k8s.

spec.schedule: "minute, hour, day of month, month, day of week"

**Service**

Create a single, constant point of entry to a group of pods. ( TCP/UDP level)

redirect by IP: spec.sessionAffinity: ClientIP (default: None) (keep-alive connection will always hit the same pod even it set to None)

Pods start after Service can get the IP:PORT from environment variables.

*   <SERVICE_NAME>_SERVICE_HOST
*   <SERVICE_NAME>_SERVICE_PORT

Dashes in the service name will be converted to underscores and all letters are uppercased.

FQDN (fully qualified domain name):

<pre><span style="text-decoration: underline;"><service_name>.<namespace>.svc.cluster.local</span></pre>

"svc.cluster.local" can be omitted. If they are in the same namespace, it can also be omitted.

<pre>spec.type</pre>

*   ExternalName: pods connect to this service will directly connect to an external endpoint
*   NodePort: each node opens a port and redirects traffic to the underlying service
*   LoadBalancer: provided by cloud infrastructure k8s is running on

spec.externalTrafficPolicy

*   Local: the traffic will only be redirected to the Pod on the Node it hits (if no local pod can be found, it will hang) (also load balance will be node locally)

**EndPoints**

This can expose service to external endpoints.

metadata.name must match service name

IPs are list in subsets.addresses.

**Ingress**

HTTP level (cookie-based or header-based session affinity). (L4 is also planned)

Ingress needs a ingress controller to do the load balance, like Nginx.

The Ingress controller doesn't forward the request to the service. It only use it to select a pod.

**Headless Service**

set "spec.clusterIP: None" to get a headless service.

With headless services, DNS will return the pods' IPs directly. It still provides load balancing across pods, but through the DNS round-robin mechanism instead of through the service proxy.

**Volumes**

types:

*   emptyDir: lifetime is tied to the pod (disk or memory)
*   hostPath: mount directories from the worker node's filesystem (DaemonSet)
*   gitRepo: init by checking out the contents of a Git repo
*   nfs: NFS share mounted into the pod
*   gcePersistentDisk(GCE), awsElasticBlockStore(AWS), azureDisk(Azure)
*   cinder, cephfs, iscsi, flocker, glusterfs, quobyte, rbd, flexVolume, vsphereVolume, photonPersistentDisk, scaleIO: other types of network storage
*   configMap, secret, downwardAPI: used to expose certain K8s resources and cluster information (metadata not data)
*   persistentVolumeClaim: pre- or dynamically provisioned persistent storage

**PersistentVolume**

ask the admin to setup this volume storage.

Still need Volume as a backup.

*   capacity
*   accessModes
*   persistemtVolumeRecalimPolicy (Retain or Delete)

PV don't belong to any namespace.

Mode:

*   RWO: ReadWriteOnce
*   ROX: ReadOnlyMany
*   RWX: ReadWriteMany

**PersistentVolumeClaim**

*   resources
*   accessModes
*   storageClassName

PVC can only be created in a specific namespace.

**StorageClass**

StorageClass resources aren't namespaced. It's dynamic. So it's impossible to run out of PV(but storage space).

**Enveronment Variables**

spec.container[*].image:

*   command: override ENTRYPOINT
*   args: override CMD
*   env[*]{name:value}: environment variables

**ConfigMap**

define: key-value pairs in metadata

usage:

*   spec.containers[*].env[*].valueFrom.configMapKeyRef
*   spec.containers[*].envFrom.configMapRef

This can also be used in volume.

Mounting a directory hides existing files in that directory. (unless you use volumeMount.subPath)

Changes in ConfigMap will be updated in pods without reload. (exclude mounted files in volume)

**Secrets**

The contents of a Secret's entries are shown as base64 encoded strings.

Maximum size is limited to 1MB.

**Downward API for metadata**

*   pod's name, IP, namespace, labels, annotations,
*   name of node, name of service account
*   CPU and memory requests/limits for each container

These can be passed into pods with environment variables or volumes.

**Deployment**

A Deployment is backed by a ReplicaSet.

When rolling update, it will create a new ReplicaSet to handle the new version pods. So it will create a ReplicaSet for each new version. (revisionHistoryLimit is 2 by default)

<pre>kubectl rollout undo deployment <name> --to-revision=1  
kubectl rollout history deployment <name></pre>

*   maxSurge: how many pod instances allow to exist above the desired replica count
*   maxUnavailable: how many pod instances allow to be unavailable relative to the desired replica count

<pre>kubectl rollout pause deployment <name>  
kubectl rollout resume deployment <name></pre>

## Useful CMD

kubelet explain pod  
kubelet explain pod.spec

<pre>kubectl exec <pod> -- <cmd>  
kubectl exec -it <pod> /bin/bash  

kubectl run <name> --image=<> --generator=run-pod/v1 --command -- sleep infinity  

kubectl get endpoints  

kubectl port-forward <name> <port_client>:<port_pod>  

kubectl exec downward env  
kubectl exec downward ls -1L /etc/downward  

kubectl proxy  

kubectl patch deployment <name> -p '{"spec": {"minReadySeconds": 10}}'  

</pre>

## Details

**GPU schedual**

<pre>kubectl label node gpu-node gpu=true  
pod.spec.nodeSelector: gpu: "true"</pre>