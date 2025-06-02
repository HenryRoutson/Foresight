

#  --------------------------------------------------------


from typing import TypedDict, Union, Any

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




json_t = dict[str, Any]

def mapObjectToNeurons(object : json_t) -> list[float]:
   
  neurons : list[float] = []
  for key, value in object.items() :

    assert(isinstance(key, str))

    if isinstance(value, bool) :

      neurons.extend([float(value)])

    else :
      # throw error if you can't handle type
      raise Exception( 
        f"\nmapObjectToNeurons : cannot handle this value data type. \n key : {key}, value {value}" 
      )

  return neurons



# TODO make a function to reverse mapObjectToNeurons


def reverseMapObjectToNeurons() :

  pass





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


