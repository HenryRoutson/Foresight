from sklearn.feature_extraction import DictVectorizer
import numpy as np
from typing import NewType

#  --------------------------------------------------------




# data has context if B has the internal data including true or false


from typing import TypedDict, Any, Optional
json_t = dict[str, Any]




# These types won't be here in the real world, they are only here to help you understand the structure of the example




event_id_t = NewType("event_id_t", str)

class Event(TypedDict):
    id: event_id_t
    id_debug : str # This is for debugging purposes,
    context : dict[str, Any]


# note that the dict structure should be the same if the event id is the same,
# TODO impliment a run time validation of this

# TODO need to support lists in context

A_none_dict : Event = { "id" : event_id_t("A"), "id_debug" : "A", "context" : {} }
B_none_dict : Event = { "id" : event_id_t("B_without_context"), "id_debug" : "B", "context" : {} }
B_false_dict : Event = { "id" : event_id_t("B_with_context"), "id_debug" : "F", "context" : { "bool_data" : False, "float_data" : 6.3, "int_data" : 2 }}
B_true_dict : Event  = { "id" : event_id_t("B_with_context"), "id_debug" : "T", "context" : { "bool_data" : True, "float_data" : -0.5, "int_data" : 5}}


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
# This section is for vectorising the context of the events





event_id_to_vectorizer : dict[event_id_t, Optional[DictVectorizer]] = {} # each event has a vectorizer to encode its context

# AI : Function to get the output length of a vectorizer for a given event_id
def get_vectorizer_output_length(event_id: event_id_t) -> int:
    if event_id not in event_id_to_vectorizer:
        raise KeyError(f"No vectorizer found for event_id: {event_id}")
    
    vectorizer : Optional[DictVectorizer] = event_id_to_vectorizer[event_id]
    if vectorizer is None :
       return 0
    else :
      return len(vectorizer.get_feature_names_out()) # type: ignore

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


def forward_vectorize(event : Event) -> Optional[np.ndarray[Any, Any]]:
  event_id : event_id_t  = event["id"]


  if event["context"] == {} :
     event_id_to_vectorizer[event_id] = None
     assert get_vectorizer_output_length(event_id) == 0
     return None


  print("forward_vectorize called for ", event_id, " with context ", event["context"])
  
  # AI : Preprocess the context before fitting or transforming
  processed_event_context = _preprocess_context_for_vectorizer(event["context"])
  # AI : Print the context that will be used by the vectorizer
  print(f"forward_vectorize: using processed context for {event_id}: {processed_event_context}")

  if not event_id in event_id_to_vectorizer.keys() : # need to fit vectorizer

    print("creating new vectorizer for ", event_id, " for context ", processed_event_context) # AI : Use processed context for fitting message
    
    event_id_to_vectorizer[event_id] = DictVectorizer(sparse=False)
    # AI : Fit with the processed context
    event_id_to_vectorizer[event_id].fit([processed_event_context]) # type: ignore

  vectorizer : Optional[DictVectorizer] = event_id_to_vectorizer[event_id]


  assert set(event["context"].keys()) == set(vectorizer.get_feature_names_out()) # type: ignore
  
  # AI : Transform with the processed context
  arr : np.ndarray[Any, Any] = vectorizer.transform([processed_event_context]) # type: ignore
  assert(isinstance(arr, np.ndarray))

  print("vectorised arr : ", arr)

  assert arr.shape[0] == 1
  assert arr.shape[1] == get_vectorizer_output_length(event_id)

  return arr


def reverse_vectorise(vector : np.ndarray[Any, Any], event_id : event_id_t) -> Optional[list[json_t]]:

  assert(event_id in event_id_to_vectorizer.keys())

  vectorizer : Optional[DictVectorizer] = event_id_to_vectorizer[event_id]
  if vectorizer is None :
     return None

  arr : list[json_t] = vectorizer.inverse_transform(vector) # type: ignore
  assert(isinstance(arr, list))
  return arr





# run a basic test


for sequence in TRAINING_DATA_WITH_CONTEXT + TRAINING_DATA_WITHOUT_CONTEXT :
   for event in sequence :
      print()
      
      print(event)


      vec = forward_vectorize(event)
      if vec is None :
         continue

      new_context_list = reverse_vectorise(vec,event["id"] )


      if new_context_list is None :
         continue

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




      




#  --------------------------------------------------------
# This section creates the extra data for probability of events ie the one hot encoding of
# and then combines with the event data



events_id_list : list[event_id_t] = list(event_id_to_vectorizer.keys())
assert len(events_id_list) == len(set(events_id_list)), "events_id_list has duplicates : " + str(events_id_list)



def event_id_get_index(event_id : event_id_t) -> int :
   return events_id_list.index(event_id)





# This is a function which creates the one hot encoding of the event id
def create_one_hot_encoding_of_event_id(event_id : event_id_t) -> np.ndarray[Any, Any]:
   one_hot_encoding = np.zeros(len(events_id_list))
   one_hot_encoding[event_id_get_index(event_id)] = 1
   return one_hot_encoding


# Now combine OHE and event data
# [is_event1, is_event2, is_event3, ... , event1_data, event2_data, event3_data, ...]
# where is_event1, is_event2, is_event3 is the one hot encoding of the event id


def create_backprop_target(event : Event) -> np.ndarray[Any, Any]:

  print("\n\n\n\n")
  print("create_backprop_target called for ", event)

  occured_event_id = event["id"]
  assert occured_event_id in events_id_list, "event_id not in events_id_list : " + str(occured_event_id) + " not in " + str(events_id_list)
  one_hot_encoding : np.ndarray[Any, Any] = create_one_hot_encoding_of_event_id(occured_event_id)



  vectors : list[np.ndarray[Any, Any]] = [one_hot_encoding]
  print("events_id_list : ", events_id_list)
  for event_id in events_id_list :
    print(vectors)
    print(f"{event_id} : length {get_vectorizer_output_length(event_id)}")


    if occured_event_id == event_id :
      print("event_id == event_id : ", event_id)

      tmp_forward_vectorize = forward_vectorize(event)
      if tmp_forward_vectorize is None :
         pass # don't need to add anything
      else :
         vectors.append(tmp_forward_vectorize[0])

    else :
      print("event_id != event_id : ", event_id)

      num_zeros = get_vectorizer_output_length(event_id)
      if num_zeros != 0 :
         
        # This list of Nones is intentional, 
        # if a backprop target is none then you should set the error for that neuron to 0
        # becuase the event never occured we have no information to backpropagate
        null_list : list[None] = [None] * num_zeros 
        null_array : np.ndarray[Any, Any] = np.array(null_list)

        vectors.append(null_array)



  print()
   



  print("vectors : ", vectors)

  
  backprop_target = np.concatenate(vectors)

  print("completed create_backprop_target ")
  return backprop_target








print("\n\n\n\n : ")


backprop_length : Optional[int] = None
for sequence in TRAINING_DATA_WITH_CONTEXT :
   for event in sequence :
      
      backprop_target = create_backprop_target(event)


      if backprop_length is None :
         backprop_length = len(backprop_target)
      else :
         assert backprop_length == len(backprop_target), "backprop_length is not the same for all sequences : " + str(backprop_length) + " != " + str(len(backprop_target))


      print("backprop_target : ", backprop_target)

  



def create_backprop_target_for_sequence(sequence : list[Event]) -> list[np.ndarray[Any, Any]]:
   new_sequence : list[np.ndarray[Any, Any]] = []
   for seq_event in sequence :
      new_sequence.append(create_backprop_target(seq_event))
   return new_sequence


def create_backprop_target_for_sequence_lists(sequence_list : list[list[Event]]) -> list[list[np.ndarray[Any, Any]]]:
   new_sequence_list : list[list[np.ndarray[Any, Any]]] = []
   for sequence in sequence_list :
      new_sequence_list.append(create_backprop_target_for_sequence(sequence))
   return new_sequence_list






TRAINING_DATA_WITH_CONTEXT_VECTORISED : list[list[np.ndarray[Any, Any]]] = create_backprop_target_for_sequence_lists(TRAINING_DATA_WITH_CONTEXT)
TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED : list[list[np.ndarray[Any, Any]]] = create_backprop_target_for_sequence_lists(TRAINING_DATA_WITHOUT_CONTEXT)


print("TRAINING_DATA_WITH_CONTEXT_VECTORISED : ", TRAINING_DATA_WITH_CONTEXT_VECTORISED)
print("TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED : ", TRAINING_DATA_WITHOUT_CONTEXT_VECTORISED)






TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION : list[list[np.ndarray[Any, Any]]] = create_backprop_target_for_sequence_lists(TRAINING_DATA_WITH_CONTEXT)

# This is an example to show how performance is worse when we naively set the values for non occuring events to 0.0
for sequence_index, sequence in enumerate(TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION) :
   for vector_index, vector in enumerate(sequence) :
      for value_index, value in enumerate(vector) :
         if value == None :
            TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION[sequence_index][vector_index][value_index] = 0.0 # This replaces the None with 0.0 over the whole vector


print("TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION : ", TRAINING_DATA_WITH_CONTEXT_VECTORISED_COERCION)

         

