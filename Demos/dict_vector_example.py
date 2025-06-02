from sklearn.feature_extraction import DictVectorizer
from typing import Literal
import numpy as np

# AI : json_t represents a dictionary with string keys and values that can be integers or floats.
json_t = dict[str, int | float]


vectorizer : DictVectorizer = DictVectorizer(sparse=False)
list_dicts : list[json_t] = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
# AI: Runtime type assertions for list_dicts
assert isinstance(list_dicts, list), "list_dicts should be a list"
assert all(isinstance(item, dict) for item in list_dicts), "All items in list_dicts should be dictionaries"
assert all(isinstance(k, str) for d in list_dicts for k in d.keys()), "All keys in list_dicts' dictionaries should be strings"
assert all(isinstance(v, int) for d in list_dicts for v in d.values()), "All values in list_dicts' dictionaries should be integers initially"
print("list_dicts ", list_dicts) # D  [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]


# AI : list_dicts_vectorised is a 2x3 numpy array of floats.
list_dicts_vectorised : np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float64]] = vectorizer.fit_transform(list_dicts) # type: ignore
# AI: Runtime type assertions for list_dicts_vectorised
assert isinstance(list_dicts_vectorised, np.ndarray), "list_dicts_vectorised should be an np.ndarray"
assert list_dicts_vectorised.shape == (2, 3), f"Shape of list_dicts_vectorised should be (2, 3), but was {list_dicts_vectorised.shape}"
assert list_dicts_vectorised.dtype == np.float64, f"Dtype of list_dicts_vectorised should be np.float64, but was {list_dicts_vectorised.dtype}"
if list_dicts_vectorised.size > 0: # Check an element type only if array is not empty
    assert isinstance(list_dicts_vectorised[0,0], np.floating), f"Elements of list_dicts_vectorised should be instances of np.floating, but found {type(list_dicts_vectorised[0,0])}"
print("list_dicts_vectorised ", list_dicts_vectorised) # D_vectorised  [[2. 0. 1.] \\n [0. 1. 3.]]
# type(list_dicts_vectorised)  <class 'numpy.ndarray'>
# type(list_dicts_vectorised[0])  <class 'numpy.ndarray'> # This is not strictly true for elements if it's a 2D array, list_dicts_vectorised[0] is a 1D array.
# type(list_dicts_vectorised[0][0])  <class 'numpy.float64'>


# To reverse the transformation:
# AI : list_dicts_un_vectorised is a list of dictionaries with string keys and float values after inverse transformation.
list_dicts_un_vectorised : list[dict[str, float]] = vectorizer.inverse_transform(list_dicts_vectorised) # type: ignore
# AI: Runtime type assertions for list_dicts_un_vectorised
assert isinstance(list_dicts_un_vectorised, list), "list_dicts_un_vectorised should be a list"
assert all(isinstance(item, dict) for item in list_dicts_un_vectorised), "All items in list_dicts_un_vectorised should be dictionaries"
assert all(isinstance(k, str) for d in list_dicts_un_vectorised for k in d.keys()), "All keys in list_dicts_un_vectorised's dictionaries should be strings"
assert all(isinstance(v, float) for d in list_dicts_un_vectorised for v in d.values()), "All values in list_dicts_un_vectorised's dictionaries should be floats"
print("D_un_vectorised ", list_dicts_un_vectorised)


# list_dicts_un_vectorised will be:
# [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]

assert list_dicts_un_vectorised == list_dicts, "Unvectorised list should match original (considering numeric equality)"