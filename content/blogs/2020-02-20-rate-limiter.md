+++
title = "Rate Limiter Implementation"
[taxonomies]
categories = ["Technology"]
+++

Rate limiter is used to constrain the request for a certain period, which can reduce the pressure of servers, prevent malicious attacks, offer more stable services.

<!-- more -->

### Basic Knowledge

Usually, the target of the rate limiter is the authorized users, or the API endpoint, or both.

To implement a rate limiter, we have several algorithms:

* Leaky Bucket
* Fixed Window
* Sliding Log
* Sliding Window

The algorithm details can be found in several blogs:

* [Kong: how to design a scalable rate limiting algorithm](https://konghq.com/blog/how-to-design-a-scalable-rate-limiting-algorithm/)
* [Figma: an alternative approach to rate limiting](https://www.figma.com/blog/an-alternative-approach-to-rate-limiting/)
* [ClassDojo: rolling rate limiter](https://engineering.classdojo.com/blog/2015/02/06/rolling-rate-limiter/)

If you are trying to find an out of box Go package, I would recommend [slidingwindow](https://github.com/RussellLuo/slidingwindow).

Next, I'll share some experience in the implementation of a rate limiter in large projects.

### Choice

Among these algorithms, I prefer to use **sliding window**. This is due to my use case. I want to reduce the network calls between rate limiters and centralized data store (Redis). Although that may harm the counter precision. But this is the tradeoff you must do.

### Implementation

To cache the count for the current window and previous window, there should be a local count and a global count in the window. When trying to sync the local records to Redis, there may be new requests comes in. So it's better to not just lock the window during the whole sync procedure. Then we need another cache count that will be used during sync and store the count if sync failed.

```go
type Window struct {
    local uint32
    global uint32
    cache uint32
}
```

Then, to get this window count, just need to add these 3 counter together.

When trying to sync, there will be a pre-sync and sync process. Pre sync will try to move local count to cache, while sync will update global count by Redis returned value and reset cache.

In this implementation, every time you try to do the `increment` to a limiter, it will check if the current window is expired and update the count, then check if the last sync is expired and send this limiter to sync queue. There will be another goroutine watching the sync queue and do the sync in a new goroutine.

The window is created with the limiter and can be reused as it only contains the counters. The time key is yet in the limiter structure.

```go
type Limiter struct {
    mu sync.RWMutex
    key string
    windowKey time.Time
    currWindow *Window
    prevWindow *Window
    windowPeriod time.Duration
    syncInterval time.Duration
    expiryTime time.Time
}
```

### Attentions

* if the limiter doesn't exist, then after you lock and try to create the limiter, you need to re-check if another goroutine already create one before this thread get the lock
* solve the race since there may be multiple threads try to sync the same limiter (this is introduced by separate the sync procedure into pre-sync and sync)
* every time you need update the current window and previous window