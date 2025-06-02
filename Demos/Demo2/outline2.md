
This is a more complex example than Demo1.
It will use Event and Data.

The output of the model should be this :

output_vector [
  Probability of A,
  Probability of B
    probability of True if B
]

You should remember to only do backpropogation on 
  probability of True if B
if B occurs.

You should use a neural network for this, 
it can be an RNN or a transformer.

Don't worry about handling any data in events more complicated than the bool in the example for now.

