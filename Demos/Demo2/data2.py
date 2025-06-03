from sklearn.feature_extraction import DictVectorizer
import numpy as np
from typing import NewType

#  --------------------------------------------------------




# data has context if B has the internal data including true or false


from typing import TypedDict, Any
json_t = dict[str, Any]




# These types won't be here in the real world, they are only here to help you understand the structure of the example




event_id_t = NewType("event_id_t", str)

class Event(TypedDict):
    id: event_id_t
    context : dict[str, Any]


# note that the dict structure should be the same if the event id is the same,
# TODO impliment a run time validation of this

A_none_dict : Event = { "id" : event_id_t("A"), "context" : {} }
B_none_dict : Event = { "id" : event_id_t("B_without_context"), "context" : {} }
B_false_dict : Event = { "id" : event_id_t("B_with_context"), "context" : { "bool_data" : False }}
B_true_dict : Event  = { "id" : event_id_t("B_with_context"), "context" : { "bool_data" : True  }}


TRAINING_DATA_WITH_CONTEXT : list[list[Event]] = [
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





TRAINING_DATA_WITHOUT_CONTEXT : list[list[Event]] = [
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






key_to_vectorizer : dict[event_id_t, DictVectorizer] = {}

# because the sequence order changes, 
# we can't vectorise the whole sequence at once, we need to vectorise each event
# and so we need a map from the type of object to the vectorizer



def forward_vectorize(event : Event, event_id : event_id_t) -> np.ndarray[Any, Any]:

  print("forward_vectorize called for ", event_id, " with context ", event["context"])
  
  if not event_id in key_to_vectorizer.keys() : # need to fit vectorizer

    print("creating new vectorizer for ", event_id, " for context ", event["context"])
    
    key_to_vectorizer[event_id] = DictVectorizer(sparse=False)
    key_to_vectorizer[event_id].fit([event["context"]]) # type: ignore

  vectorizer : DictVectorizer = key_to_vectorizer[event_id]


  assert set(event["context"].keys()) == set(vectorizer.get_feature_names_out()) # type: ignore
  
  arr : np.ndarray[Any, Any] = vectorizer.transform([event["context"]]) # type: ignore
  assert(isinstance(arr, np.ndarray))
  return arr


def reverse_vectorise(vector : np.ndarray[Any, Any], event_id : event_id_t) -> list[json_t]:

  assert(event_id in key_to_vectorizer.keys())

  vectorizer : DictVectorizer = key_to_vectorizer[event_id]
  arr : list[json_t] = vectorizer.inverse_transform(vector) # type: ignore
  assert(isinstance(arr, list))
  return arr





# run a basic test


for sequence in TRAINING_DATA_WITH_CONTEXT :
   for event in sequence :
      print()
      
      if event["context"] == {} :
         continue
      
      print(event)


      vec = forward_vectorize(event, event["id"])
      new_context_list = reverse_vectorise(vec,event["id"] )

      print("new_context_list[0] : ", new_context_list[0])
      print("event[\"context\"] : ", event["context"])

      assert new_context_list[0] == event["context"]




