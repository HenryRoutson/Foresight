from sklearn.feature_extraction import DictVectorizer
from typing import Any
import numpy as np

json_t = dict[str, Any]


vectorizer : DictVectorizer = DictVectorizer(sparse=False)
list_dicts : list[json_t] = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
print("list_dicts ", list_dicts) # D  [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]


list_dicts_vectorised : np.ndarray = vectorizer.fit_transform(list_dicts)
print("list_dicts_vectorised ", list_dicts_vectorised) # D_vectorised  [[2. 0. 1.] \n [0. 1. 3.]]
# type(list_dicts_vectorised)  <class 'numpy.ndarray'>
# type(list_dicts_vectorised[0])  <class 'numpy.ndarray'>
# type(list_dicts_vectorised[0][0])  <class 'numpy.float64'>


# To reverse the transformation:
list_dicts_un_vectorised : list[json_t] = vectorizer.inverse_transform(list_dicts_vectorised)
print("D_un_vectorised ", list_dicts_un_vectorised)


# list_dicts_un_vectorised will be:
# [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]

assert(list_dicts_un_vectorised == list_dicts)