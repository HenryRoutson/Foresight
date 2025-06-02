

#  --------------------------------------------------------


from typing import TypedDict, Union

class A(TypedDict):
    A: None
class B_data(TypedDict):
  bool_data : bool
class B_with_data(TypedDict):
  B : B_data
class B_without_data(TypedDict):
  B : None

A_none_dict : A = { "A" : None }

# use b with or without data
B_none_dict : B_without_data = { "B" : None }
# OR
B_false_dict : B_with_data = { "B" : { "bool_data" : False }}
B_true_dict : B_with_data  = { "B" : { "bool_data" : True  }}



withDataEvents = Union[A, B_with_data]
withoutDataEvents = Union[A, B_without_data]


TRAINING_DATA_WITH_CONTEXT : list[list[withDataEvents]] = [
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





TRAINING_DATA_WITHOUT_CONTEXT : list[list[withoutDataEvents]] = [
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


