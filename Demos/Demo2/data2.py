
from sklearn.feature_extraction import DictVectorizer
import numpy as np

#  --------------------------------------------------------




# data has context if B has the internal data including true or false


from typing import TypedDict, Union, Any
json_t = dict[str, Any]

class A(TypedDict):
    A: None
class B_data(TypedDict):
  bool_data : bool # context data
class B_with_context(TypedDict):
  B : B_data
class B_without_context(TypedDict):
  B : None

A_none_dict : A = { "A" : None }

# use b with or without data
B_none_dict : B_without_context = { "B" : None }
# OR
B_false_dict : B_with_context = { "B" : { "bool_data" : False }}
B_true_dict : B_with_context  = { "B" : { "bool_data" : True  }}



DataWithContext = Union[A, B_with_context]
DataWithoutContext = Union[A, B_without_context]


TRAINING_DATA_WITH_CONTEXT : list[list[DataWithContext]] = [
  [A_none_dict, A_none_dict, A_none_dict],
  [A_none_dict, B_false_dict, B_false_dict],
  [A_none_dict, B_true_dict, B_true_dict],
  [B_false_dict, A_none_dict, A_none_dict],
  [B_false_dict, B_false_dict, A_none_dict],
  [B_false_dict, B_true_dict, B_false_dict],
  [B_true_dict, A_none_dict, B_true_dict],
  [B_true_dict, B_false_dict, B_false_dict],
  [B_true_dict, B_true_dict, B_true_dict]
]








#  --------------------------------------------------------





TRAINING_DATA_WITHOUT_CONTEXT : list[list[DataWithoutContext]] = [
  [A_none_dict, A_none_dict, A_none_dict],
  [A_none_dict, B_none_dict, B_none_dict],
  [A_none_dict, B_none_dict, B_none_dict],
  [B_none_dict, A_none_dict, A_none_dict],
  [B_none_dict, B_none_dict, A_none_dict],
  [B_none_dict, B_none_dict, B_none_dict],
  [B_none_dict, A_none_dict, B_none_dict],
  [B_none_dict, B_none_dict, B_none_dict],
  [B_none_dict, B_none_dict, B_none_dict]
]










#  --------------------------------------------------------




with_context_vectorizer = DictVectorizer(sparse=False) # vectorise : [A, B_with_context]
without_context_vectorizer = DictVectorizer(sparse=False) # vectorise : [A, B_without_context]







def forward_vectorize(vectorizer : DictVectorizer, object : list[json_t]) -> np.ndarray[Any, Any] :
  pass




def reverse_vectorise(vectorizer : DictVectorizer, vector : np.ndarray[Any, Any]) -> list[json_t]:
  pass



