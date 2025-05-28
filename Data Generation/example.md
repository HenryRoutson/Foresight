




## Event types

  - A
  - B {bool}

## Derived states

This corresponds to 3 states. 


```
A         -> 0
B {False} -> 1
B {True}  -> 2
```

## Sequence / attention

To involve attention and sequence data, we will look at all combinations of two events.
For this the context length only needs to see the last two outcomes.

There are now 9 states

```
0 0 > [A, A]
0 1 > [A, B {False}]
0 2 > [A, b {True}]

1 0 > [B {False}, A]
1 1 > [B {False}, B {False}]
1 2 > [B {False}, b {True}]

2 0 > [B {True}, A]
2 1 > [B {True}, B {False}]
2 2 > [B {True}, b {True}]
```

We can then use python to generate a state transition matrix, for example ...

```Python
                                  →0     →1     →2     →3     →4     →5     →6     →7     →8
State 0: [A, A]                0.006  0.001  0.006  0.005  0.004  0.001  0.970  0.004  0.003
State 1: [A, B{False}]         0.006  0.000  0.009  0.007  0.002  0.002  0.002  0.970  0.003
State 2: [A, B{True}]          0.000  0.000  0.007  0.005  0.001  0.970  0.013  0.003  0.001
State 3: [B{False}, A]         0.002  0.006  0.970  0.003  0.005  0.004  0.003  0.000  0.006
State 4: [B{False}, B{False}]  0.004  0.970  0.000  0.002  0.002  0.006  0.006  0.008  0.002
State 5: [B{False}, B{True}]   0.970  0.002  0.005  0.002  0.004  0.004  0.001  0.007  0.005
State 6: [B{True}, A]          0.002  0.970  0.005  0.004  0.002  0.003  0.003  0.005  0.005
State 7: [B{True}, B{False}]   0.005  0.970  0.005  0.008  0.005  0.002  0.003  0.001  0.000
State 8: [B{True}, B{True}]    0.006  0.001  0.000  0.006  0.005  0.005  0.006  0.001  0.970
```


## Summary

Using this method we can create sequences which 
  - are largely predictable but not memorisable
  - involve both sequence (A and B) and event data (such as B {False})
  - very simple to understand

This will be useful in demonstrating the value of a model which can include both
sequence and event data, with this being the simplest example possible. 