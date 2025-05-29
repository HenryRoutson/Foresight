




## Event types

  - A
  - B {bool}

## Derived states

This corresponds to 3 states. 


```
STATES:
A         -> 0
B {False} -> 1
B {True}  -> 2
```

## Sequence / attention

To involve attention and sequence data, we will look at all combinations of two events.
For this the context length only needs to see the last two outcomes.

There are now 9 sequences

```
SEQUENCES

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
State     :                       →1 (A)             →2 (B {False})    →3 (B {True})
Sequence 0: [A, A]                0.900              0.050              0.050
Sequence 1: [A, B{False}]         0.050              0.900              0.050
Sequence 2: [A, B{True}]          0.050              0.050              0.900
Sequence 3: [B{False}, A]         0.900              0.050              0.050
Sequence 4: [B{False}, B{False}]  0.900              0.050              0.050
Sequence 5: [B{False}, B{True}]   0.050              0.900              0.050
Sequence 6: [B{True}, A]          0.050              0.050              0.900
Sequence 7: [B{True}, B{False}]   0.050              0.900              0.050
Sequence 8: [B{True}, B{True}]    0.050              0.050              0.900
```

with xxx replaced with numerical values where each row adds to one so that each sequence samples with each of the new states having the associated probability.

From the above graph we can see that the model will predict for the next state in the seqeunce  ... 

```
[A, A] > A
[A, B{False}] > B {False}
[A, B{True}] > B {True}
[B{False}, A] > A
[B{False}, B{False}] > A
[B{False}, B{True}] > B {False}
[B{True}, A] > B {True}
[B{True}, B{False}] > B {False}
[B{True}, B{True}] > B {True}
```

## Summary

Using this method we can create sequences which 
  - are largely predictable but not memorisable
  TODO is this memorisable if the context length is 2?? 

  - involve both sequence (A and B) and event data (such as B {False})
  - very simple to understand

This will be useful in demonstrating the value of a model which can include both
sequence and event data, with this being the simplest example possible. 