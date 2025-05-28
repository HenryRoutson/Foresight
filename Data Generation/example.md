




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

We can then use python to generate a state transition matrix

```Python
TODO make distribution easier to predict
```




