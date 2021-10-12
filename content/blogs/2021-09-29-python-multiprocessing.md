+++
title = "Why not multiprocessing"
[taxonomies]
categories = ["Technology", "Python"]
+++

Be careful to use multiprocessing in production.

<!-- more -->

## start from a segment fault

Here is a code snippet that will run well on Darwin but trigger a segment fault on Unix.

```python
import multiprocessing as mp
from time import sleep


def wait_for_event(event):
    while not event.is_set():
        sleep(0.1)


def trigger_segment_fault():
    event = mp.Event()
    p = mp.get_context("spawn").Process(target=wait_for_event, args=(event,))
    p.start()  # this will show the exitcode=-SIGSEGV
    sleep(1)
    print(p)
    event.set()
    p.terminate()


if __name__ == "__main__":
    trigger_segment_fault()
```

Yeah, the pure Python code can trigger a segment fault.

The reason is because of the new process start method. According to the [Python document](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods), `spawn` is the default one on macOS (start from Python 3.8) while `fork` is the default one on Unix. But the start method also affects the `Event` creation. Let's check the source code:

```python
class Event(object):

    def __init__(self, *, ctx):
        self._cond = ctx.Condition(ctx.Lock())
        self._flag = ctx.Semaphore(0)
```

The initialization takes a `ctx` which is related to the start method. So when you try to access a forked event in a spawned process, this segment fault occurs. The way to solve this is simple -- using the same context. (Actually, you can use the *spawn* event in the forked process)

## *fork* or *spawn*

Another question is that, which start method should I use?

> *spawn*: The parent process starts a **fresh** python interpreter process. The child process will only inherit those resources necessary to run the process objects run() method. In particular, unnecessary file descriptors and handles from the parent process will not be inherited.

> *fork*: The parent process uses `os.fork()` to fork the Python interpreter. The child process, when it begins, is effectively identical to the parent process. All resources of the parent are inherited by the child's process. Note that safely forking a multithreaded process is **problematic**.

We can see that *spawn* will create a new Python process and only inherit necessary resources. *fork* will call the underlying `os.fork()`, but the implementation in CPython is problematic.

When you are using *spawn*, accidentally access the main process variables may have some unexpected consequences.

```python
import multiprocessing as mp
import os


class Dummy:
    def __init__(self) -> None:
        print(f"init in pid: {os.getpid()}")


Dummy()
x = None


def task():
    if x is None:
        print("x is None")


if __name__ == "__main__":
    p = mp.get_context("spawn").Process(target=task)
    p.start()
    p.join()
```

In the above code snippet, if the *spawn* process tries to access the variable `x`, it will trigger the initialization of both `Dummy()` and `x = None`. So you can see the terminal will print two "init in pid" with different PIDs.

So what kind of problem can the *fork* cause? Let's take a look at this article: [pythonspeed: Python multiprocessing](https://pythonspeed.com/articles/python-multiprocessing/). 

> This code snippet is copied from the above article and changed to make it clear to explain.

```python
from os import fork
from time import sleep
from threading import Lock

# Lock is acquired in the parent process:
lock = Lock()
lock.acquire()

if fork() == 0:
    # In the child process, try to grab the lock:
    print("Child process: Acquiring lock...")
    lock.acquire()
    print("Lock acquired! (This code will never run)")
else:
    lock.release()
    print("Parent process: release the lock")
    sleep(1)
    print("exit the parent process")
```

In the above example, after the lock is released, the child process still cannot acquire the lock. Why?

The main point is that fork doesn't copy everything.

Let's check the [man page of fork](https://man7.org/linux/man-pages/man2/fork.2.html):

> The child does not inherit its parent's memory locks

> The child does not inherit semaphore adjustments from its parent

So what happens here is that the child process has a lock already been acquired, but no thread will release the lock. These two locks are not the same as we can see in the parent process.

The solution is to use *spawn* instead of *fork*:

```python
from multiprocessing import set_start_method
set_start_method("spawn")
```

The code snippet above may cause some problems when the code is executed more than once.

My suggestion is to use the start method context:

```python
import multiprocessing as mp


context = mp.get_context("spawn")
context.Event()
```

## garbage collection with deadlock

Let's take a look at another article: [The tragic tale of the deadlocking Python queue](https://codewithoutrules.com/2017/08/16/concurrency-python/).

> This code snippet is copied from the above article.

```python
from queue import Queue

q = Queue()


class Circular(object):
    def __init__(self):
        self.circular = self

    def __del__(self):
        print("Adding to queue in GC")
        q.put(1)


for i in range(1000000000):
    print("iteration", i)
    # Create an object that will be garbage collected
    # asynchronously, and therefore have its __del__
    # method called later:
    Circular()
    print("Adding to queue regularly")
    q.put(2)
```

Usually, we believe that Python runs one line at a time. But that's not true.

> Garbage collection can interrupt Python functions at any point, and run arbitrary other Python code: `__del__` methods and [weakref](https://docs.python.org/3/library/weakref.html) callbacks. So can signal handlers, which happen e.g. when you hit Ctrl-C (your process gets the SIGINT signal) or a subprocess dies (your process gets the SIGCHLD signal).

So when we try to `q.put(2)`, the queue needs to acquire the lock. Meanwhile, the GC will try to call the `__del__` which also does the `q.put(1)`. The `q.put(2)` is blocked by the GC, but the GC cannot acquire the lock because `q.put(2)` won't release it. Deadlock happens!

Thanks to the Python-dev team, this has been fixed in Python 3.7.

## Copy on write

When running with multiprocessing, we hope the child process can share some data with the main process instead of copying from it. Especially when they are not used in the child process. This sounds reasonable. However, we missed another important part in Python: reference counting.

CPython contains two kinds of garbage collection methods: reference counting and generational garbage collection. The reference counting is the fundamental one and cannot be disabled. The generational garbage collection is mainly used to solve the reference cycles. Check this article for more details: [Garbage collection in Python: things you need to know](https://rushter.com/blog/python-garbage-collector/) and [Design of CPython’s Garbage Collector](https://devguide.python.org/garbage_collector/).

Let's take a look at the CPython implementation of PyObject:

```c
typedef struct _object {
    _PyObject_HEAD_EXTRA
    Py_ssize_t ob_refcnt;
    PyTypeObject *ob_type;
} PyObject;
```

There is a class member called `ob_refcnt` which is used to track the reference counting. If we call `fork()` in the new process, the reference counting of all the Python objects will increase. This means the object itself has changed although the data accessed by the user is still the same.

<!-- 
```python
import sys
import multiprocessing as mp


data = list(range(1000))

def get_ref_count():
    print("child process", sys.getrefcount(data))


if __name__ == "__main__":
    print("main process:", sys.getrefcount(data))
    p = mp.get_context("spawn").Process(target=get_ref_count)
    p.start()
    p.join()
```

If we run the code above, we will see that the default reference count is 2 (because the function `sys.getrefcount()` will increase it by 1). In the child process, the reference count has changed to 3. -->

To handle this problem, the Instagram Engineering team has come up with a solution: [Copy-on-write friendly Python garbage collection](https://instagram-engineering.com/copy-on-write-friendly-python-garbage-collection-ad6ed5233ddf).

```c
static PyObject *
gc_freeze_impl(PyObject *module)
/*[clinic end generated code: output=502159d9cdc4c139 input=b602b16ac5febbe5]*/
{
    GCState *gcstate = get_gc_state();
    for (int i = 0; i < NUM_GENERATIONS; ++i) {
        gc_list_merge(GEN_HEAD(gcstate, i), &gcstate->permanent_generation.head);
        gcstate->generations[i].count = 0;
    }
    Py_RETURN_NONE;
}
```

Let's check the [Python document for GC](https://docs.python.org/3/library/gc.html#gc.freeze). In Python 3.7, it introduced a new method called `gc.freeze`:

> Freeze all the objects tracked by gc - move them to a permanent generation and ignore all the future collections.

<!-- Let's modify the code to:

```python
import sys
import gc
import multiprocessing as mp


data = list(range(1000))

def get_ref_count():
    print("child process", sys.getrefcount(data))


if __name__ == "__main__":
    gc.freeze()
    print("main process:", sys.getrefcount(data))
    p = mp.get_context("spawn").Process(target=get_ref_count)
    p.start()
    p.join()
```

It works now. The reference count in the child process is the same as the main process. -->

So will this solve the Copy-on-write problem? I'm not sure because I cannot come up with an example to reproduce it.

```python
import time
import psutil
import multiprocessing as mp


def display_memory_usage(msg=""):
    process = psutil.Process()
    print(msg, ">", process.memory_info())


def processing():
    display_memory_usage("child ")


if __name__ == "__main__":
    data = list(range(10000000))

    p = mp.get_context("fork").Process(target=processing)
    p.start()

    time.sleep(0.1)
    display_memory_usage("parent")
    p.join()
```

The code snippet above will print the memory usage of the main process and child process. You may get something like this:

    child  > pmem(rss=414748672, vms=427634688, shared=2969600, text=2035712, lib=0, data=411791360, dirty=0)
    parent > pmem(rss=419000320, vms=427634688, shared=7221248, text=2035712, lib=0, data=411791360, dirty=0)

We can see that they don't share a lot. Although by default, the *fork* process should share the data with the parent process.

But if we change it to *spawn*, we will get something like this:

    child  > pmem(rss=13848576, vms=23044096, shared=7069696, text=2035712, lib=0, data=7163904, dirty=0)
    parent > pmem(rss=419139584, vms=428081152, shared=7196672, text=2035712, lib=0, data=412200960, dirty=0)

Since the `data` is not used by the *spawn* process, so this won't be copied to the new process.

I try to add the `gc.freeze()` before creating a new process, but it doesn't work at all. Not sure what I have missed.

I found that some discussion in the [`gc.freeze()` PR](https://github.com/python/cpython/pull/3705#issuecomment-420191452). It looks that the untouched data should be able to share among processes. Also, it has been 4 years for Gunicorn to process this [support for `gc.freeze()` for apps that use preloading](https://github.com/benoitc/gunicorn/issues/1640). I cannot found a good example to demonstrate that this method works well.

To my understanding, the `gc.freeze()` will disable the generational garbage collection. But the reference counting cannot be disabled. So if we *fork* a new process, everything will be shared with the new process, which means it will change all the reference count.

If we change the start method from *spawn* to *fork*, it doesn't need the `gc.freeze()` to freeze the reference count, which has conflicts with the description in the Instagram blog.

Is there any method to avoid this? Yes. Check another blog written before the Instagram blog: [Python vs Copy on Write](https://llvllatrix.wordpress.com/2016/02/19/python-vs-copy-on-write/). The solution is very straightforward:

* You can just use the [PyPy](https://www.pypy.org/) because it has [a different way for garbage collection](https://doc.pypy.org/en/latest/cpython_differences.html#differences-related-to-garbage-collection-strategies).
* You can use the [Shared `ctypes` Objects](https://docs.python.org/3/library/multiprocessing.html#shared-ctypes-objects).
* You can use the [shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) for Python >= 3.8.
* You can use the [mmap](https://docs.python.org/3/library/mmap.html) to [reduce memory usage of array copies](https://pythonspeed.com/articles/reduce-memory-array-copies/).


## Suggestions

* Try to use Go, Rust, or C++ to do concurrency computing.
* Use *spawn* instead of *fork*.
* Be careful about the garbage collection behavior.
