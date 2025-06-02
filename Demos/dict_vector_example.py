from sklearn.feature_extraction import DictVectorizer
from typing import Any
import numpy as np


v : DictVectorizer = DictVectorizer(sparse=False)
D : dict[str, Any] = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
print("D ", D)


D_vectorised = v.fit_transform(D)
print("D_vectorised ", D_vectorised)

# X is now:
# array([[2., 0., 1.],
#        [0., 1., 3.]])

# To reverse the transformation:
D_un_vectorised : dict[str, Any] = v.inverse_transform(D_vectorised)
print("D_un_vectorised ", D_un_vectorised)


# D_un_vectorised will be:
# [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}]

assert(D_un_vectorised == D)