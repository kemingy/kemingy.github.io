+++
title = "Note for MapReduce"
[taxonomies]
categories = ["Paper"]
+++

[MapReduce: Simplified Data Processing on Large Clusters](https://pdos.csail.mit.edu/6.824/papers/mapreduce.pdf)

<!-- more -->


* sending code to servers
* tracking tasks
* moving data from Map to Reduce
* load balancing over servers
* recovering from failures

> Limit

* no interaction or state
* no multi-stage pipeline
* no real-time or streaming processing

> Bottleneck

* network
* root switch

> Minimize network use

* input is read from local disk (via GFS)
* Map workers write to local disk
* Reduce workers read directly from Map workers
* intermediate data partitioned into files holding many keys

> load balance

* small tasks
* Master hands out new tasks to workers who finish previous tasks

> fault tolerance

* re-run the failed Maps and Reduces
* Map and Reduce must be pure deterministic functions

> worker crash recovery

* Map
  * Master tells other workers to run those lost tasks
  * omit if Reduce workers already fetched the intermediate data
* Reduce
  * Master re-start worker's unfinished tasks
  
> other failures/problems

* Master gives 2 workers the same Map() task: tell Reduce workers about only one of them
* Master gives 2 workers the same Reduce() task: GFS handle this
* a single worker is very slow: Master starts a 2nd copy of the last few tasks

> conclusion

```diff
- not the most efficient or flexible
+ scales well
+ easy to program
```

### Procedure

1. Splits input files into *M* pieces of typically 16 MB ~ 64 MB per piece.
2. Master picks idle workers and assigns each one a map task or a reduce task.
3. Map worker parses the input data and passes each key/value pair to the user-defined *Map* function. Results are buffered in memory.
4. Periodically write the buffered pairs to local disk, partitioned into *R* regions. Pass the locations back to the master.
5. Reduce worker reads all the data through RPC and sorts it by the intermediate keys. An external sort is used when the intermediate data is too large to fit in memory.
6. Reduce worker passes the keys and the corresponding set of intermediate values to the user-defined *Reduce* function. Results are appended to a final output file.
7. After all map and reduce tasks have been completed, master wakes up the user program.

### Master Data Structure

```rust
enum State {
  idle,
  in_progress,
  completed,
}

struct Task {
  state: State,
  localtion: String,
  size: u32,
}
```

### Fault Tolerance

> worker failure

* master pings every worker periodically
* if no response, marks the worker as falied
* any map tasks completed by the failed worker are reset (because the results are stored locally)
* any map tasks or reduce tasks in progress on the failed worker are reset
* notify the reduce worker

> master failure

* checkpoints for recovery

### Locality

The MapReduce master takes the location information of the input files into account and attempts to schedule a map task on a
machine that contains a replica of the coressponding input data.

### Task Granularity

* scheduling: `O(M+R)`
* states: `O(M*R)`
* each piece of input data size: 16 MB to 64 MB
* M = 200,000
* R = 5,000
* workers = 2,000

### Backup Tasks

When a MapReduce operation is close to completion, the master schedules backup executions of the remaining *in-progress* tasks.
The task is marked as completed whenever either the primary or the backup execution completes.

### Partitioning Function

`hash(func(key)) mod R`

### Ordering Guarantees

Within a given partition, the intermediate key/value pairs are processed in increasing key order.

### Combiner Function

(Optional) Combiner function defined by user will partial merging the data after Map before sending to Reduce worker.

Sometimes this can significantly speeds up the MapReduce operations.

### Input and Output Types

Default *reader* function: the key is the offset in the file and the value is the contents of the line.

### Skipping Bad Records

Each worker process installs a signal handler that catches segementation violations and bus errors. It will send a "last gasp" UDP packet that contains the sequence number to the master. When the master has seen more than one failure on a particular record, it indicates that the record should be skipped.

### Counters

The counter values from individual worker machines are periodically propagated to the master (piggybacked on the ping response).
