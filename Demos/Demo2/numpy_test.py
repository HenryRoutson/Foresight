

# using null in numpy


import numpy as np

# create an array with null and 0 and see if this is retained and not coerced into a float



numpy_array = np.array([None, 0, 0.0])
print("numpy_array : ", numpy_array)