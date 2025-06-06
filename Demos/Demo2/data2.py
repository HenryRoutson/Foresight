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


# This is a function which avoids of Falsey values being ignored by the vectorizer, 
# I may try to merge something into sci kit learn to override this behaviour later
# AI : Helper function to preprocess context for DictVectorizer
def _preprocess_context_for_vectorizer(context: dict[str, Any]) -> dict[str, Any]:
    processed_context: dict[str, Any] = {} # AI : Explicitly type hint the local variable
    for key, value in context.items():
        # AI : Map False, 0, or 0.0 to a low value to ensure key presence after inverse_transform
        if value is False or value == 0 or value == 0.0: # Explicitly check for False, 0, 0.0
            processed_context[key] = 0.0000001
        else:
            processed_context[key] = value
    return processed_context


def forward_vectorize(event : Event, event_id : event_id_t) -> np.ndarray[Any, Any]:

  print("forward_vectorize called for ", event_id, " with context ", event["context"])
  
  # AI : Preprocess the context before fitting or transforming
  processed_event_context = _preprocess_context_for_vectorizer(event["context"])
  # AI : Print the context that will be used by the vectorizer
  print(f"forward_vectorize: using processed context for {event_id}: {processed_event_context}")

  if not event_id in key_to_vectorizer.keys() : # need to fit vectorizer

    print("creating new vectorizer for ", event_id, " for context ", processed_event_context) # AI : Use processed context for fitting message
    
    key_to_vectorizer[event_id] = DictVectorizer(sparse=False)
    # AI : Fit with the processed context
    key_to_vectorizer[event_id].fit([processed_event_context]) # type: ignore

  vectorizer : DictVectorizer = key_to_vectorizer[event_id]


  assert set(event["context"].keys()) == set(vectorizer.get_feature_names_out()) # type: ignore
  
  # AI : Transform with the processed context
  arr : np.ndarray[Any, Any] = vectorizer.transform([processed_event_context]) # type: ignore
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


      """

      Some types will change when they are vectorised 
      ie bool changes to the probability of the bool being true for higher precision 

      bool -> float
      int -> float
      float -> float


      """



      new_context : json_t = new_context_list[0]
      old_context : json_t = event["context"]


      # TODO there is an issue where if values are Falsey ie 0.0 or False they are not included in the new_context
      if new_context.keys() != old_context.keys() :
         print("new_context.keys() != old_context.keys() : ", new_context.keys(), " != ", old_context.keys())
         print("new_context : ", new_context)
         print("old_context : ", old_context)
         assert False


      for (old_key, old_value), (new_key, new_value) in zip(old_context.items(), new_context.items()) :
        print("old_key : ", old_key)
        print("type of old_value : ", str(type(old_value)))
        print("new_key : ", new_key)
        print("type of new_value : ", str(type(new_value)))


        if type(old_value) == bool :
           assert type(new_value) == np.float64, "new_value is not a float : " + str(new_value) + " it is a " + str(type(new_value))
           






         

         # some types change when they are vectorised
         


      




