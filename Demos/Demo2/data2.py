
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
  # predict 3rd token from first two
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
  # predict 3rd token from first two
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






B_with_context_vectoriser = DictVectorizer(sparse=False)

# because the sequence order changes, we can't vectorise the whole sequence at once, we need to vectorise each event
# TODO this we can use the "B" or "A" key to quickly determine which instance an event is
# TODO To do this properly at run time, you should make a map of event key to vectoriser



def forward_vectorize(vectorizer : DictVectorizer, object : list[json_t]) -> np.ndarray[Any, Any] :

  print("forward_vectorize : ", vectorizer, object)

  arr : np.ndarray[Any, Any] = vectorizer.fit_transform(object) # type: ignore
  assert(isinstance(arr, np.ndarray))
  return arr


def reverse_vectorise(vectorizer : DictVectorizer, vector : np.ndarray[Any, Any]) -> list[json_t]:

  print("forward_vectorize : ", vectorizer, vector)

  arr : list[json_t] = vectorizer.inverse_transform(object) # type: ignore
  assert(isinstance(arr, list))
  return arr









