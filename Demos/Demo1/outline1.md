




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
Sequence 0: [A, A]                1.000              0.000              0.000
Sequence 1: [A, B{False}]         0.000              1.000              0.000
Sequence 2: [A, B{True}]          0.000              0.000              1.000
Sequence 3: [B{False}, A]         1.000              0.000              0.000
Sequence 4: [B{False}, B{False}]  1.000              0.000              0.000
Sequence 5: [B{False}, B{True}]   0.000              1.000              0.000
Sequence 6: [B{True}, A]          0.000              0.000              1.000
Sequence 7: [B{True}, B{False}]   0.000              1.000              0.000
Sequence 8: [B{True}, B{True}]    0.000              0.000              1.000
```

with xxx replaced with numerical values where each row adds to one so that each sequence samples with each of the new states having the associated probability.

From the above graph we can see that a good model will predict the following for the next state in the seqeunce  ... 

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


(IMPORTANT >>>) You should use a confusion matrix to test this!!!

Given this is deterministic, 
we should expect 100% accuracy
when predicting both sequence and data
(IE [A, B{True}, B{True}, B{True}])



```


However if we don't have internal data of events ... 


```
[A, A] > A
[A, B] > B
[A, B] > B
[B, A] > A
[B, B] > A
[B, B] > B
[B, A] > B
[B, B] > B
[B, B] > B 


which simplifies to 

[A, A] > A
[A, B] > B
[B, A] > (1/2 A, 1/2 B)
[B, B] > (1/4 A, 3/4 B)

(IMPORTANT >>>) You should use a confusion matrix to test this!!!


Where there is clearly more uncertianty about what to predict next.


We should expect 
1/4 * 1 +
1/4 * 1 +
1/4 * (1/2) +
1/4 * (3/4) 
= 0.8125
= 81% accuracy
when predicting seqeunce (IE [A,B,B,A]) only

```




As a side note you can model this with a state machine.

## models.py should contain

- Confusion matricies to check that theoretical confusions are observed 
- A context length of 2



## Summary

Using this method we can create sequences which 
  - involve both sequence (A and B) and event data (such as B {False})
  - very simple to understand

This will be useful in demonstrating the value of a model which can include both
sequence and event data, with this being the simplest example possible. 