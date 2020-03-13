+++
title = "Python Descriptor Short Note"
[taxonomies]
categories = ["Technology"]
+++

**Define**: a descriptor is an object attribute with "binding behavior", one whose attribute access has been overridden by methods in the descriptor protocal.

<!-- more -->

`a.x` -> `a.__dict__['x']` -> `type(a).__dict__['x']`

## Descriptor Protocol

* `descr.__get__(self, obj, type=None) -> value`
* `descr.__set__(self, obj, value) -> None`
* `descr.__delete__(self, obj) -> None`

If only implements `__get__` then it's said to be a **non-data descriptor**. If implements `__set__` and `__del__` then it's said to be a **data descriptor**.

## Lookup Chain

1. `__get__` of the data descriptor
2. object's `__dict__`
3. `__get__` of the non-data descriptor
4. object type's `__dict__`
5. object parent type's `__dict__`
6. repeat for all the parent type's `__dict__`
7. `AttributionError`

## Use case

* lazy properties
* Don't repeat yourself(D.R.Y.) code `__set_name__`

# Decorator

## Wrap 3 layers

1. arguments for decorator `@decorator(*args, **kwargs)`
2. wrapped function `wrapped_func(func)`
3. arguments for wrapped function `@wrap(func)(somefunc(*args, **kwargs))`

```py
def decorator_func(*args, **kwargs):
    def wrapper(func):
        @functools.wraps(func)
        def somefunc(*func_args, **func_kwargs):
            resp = func(*func_args, **func_kwargs)
            return resp
            
        return somefunc
    return wrapper
```