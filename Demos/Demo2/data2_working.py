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



# AI : Helper function to convert boolean values to strings for DictVectorizer
def preprocess_context(context: json_t) -> json_t:
    # AI : Convert boolean values to their string representations
    # AI : This ensures that False is treated as a distinct category
    # AI : and not as 0.0, which DictVectorizer might ignore on inverse_transform.
    return {k: str(v) if isinstance(v, bool) else v for k, v in context.items()}

# AI : Helper function to reconstruct original context from DictVectorizer output
def reconstruct_context(processed_context: json_t) -> json_t:
    # AI : Reconstructs the original context dictionary from the
    # AI : format output by DictVectorizer (e.g., {'key=value': 1.0}).
    # AI : It handles stringified booleans and other potential transformations.
    original_context: json_t = {}
    for k_eq_v, val in processed_context.items():
        if val == 0.0: # Skip features that were absent or numerically zero if not part of a 'key=value' structure
            continue
        parts = k_eq_v.split('=', 1)
        key = parts[0]
        # AI : If there was an '=', it means it was a categorical feature that DictVectorizer expanded.
        # AI : The original value is the part after '='.
        # AI : We need to convert stringified booleans back.
        if len(parts) > 1:
            value_str = parts[1]
            if value_str == "True":
                original_context[key] = True
            elif value_str == "False":
                original_context[key] = False
            else:
                # AI : Attempt to convert to float or int if possible, otherwise keep as string
                try:
                    original_context[key] = float(value_str) if '.' in value_str else int(value_str)
                except ValueError:
                    original_context[key] = value_str
        else:
            # AI : If no '=', it might be a numerical feature directly, or a feature name that didn't get expanded.
            # AI : DictVectorizer usually outputs 1.0 for present categorical features after one-hot encoding.
            # AI : For this specific problem, our stringified booleans become "key=True" or "key=False".
            # AI : Direct numerical values would pass through as is, but we are stringifying bools.
            # AI : This branch might not be hit often with the current stringification strategy for bools.
            original_context[key] = val
    return original_context

# AI : Initialize vectorizers by fitting them on all unique contexts for each event_id
def initialize_vectorizers(training_data: list[list[Event]]):
    global key_to_vectorizer
    # AI : Collect all unique contexts for each event_id
    contexts_by_event_id: dict[event_id_t, list[json_t]] = {}
    for sequence in training_data:
        for event in sequence:
            event_id = event["id"]
            # AI : Only process contexts that are not empty
            if event["context"]:
                # AI : Preprocess context before collecting (e.g. stringify booleans)
                processed_event_context = preprocess_context(event["context"])
                if event_id not in contexts_by_event_id:
                    contexts_by_event_id[event_id] = []
                # AI : Add to list if not already present to avoid redundant fitting data for DictVectorizer
                # AI : DictVectorizer's fit method handles unique feature discovery
                contexts_by_event_id[event_id].append(processed_event_context)

    # AI : Fit a DictVectorizer for each event_id
    for event_id, contexts in contexts_by_event_id.items():
        if contexts: # Ensure there are contexts to fit
            print(f"Initializing vectorizer for {event_id} with contexts: {contexts}")
            vectorizer = DictVectorizer(sparse=False)
            vectorizer.fit(contexts) # type: ignore
            key_to_vectorizer[event_id] = vectorizer
            # AI : Add type hint for feature_names to satisfy linter
            feature_names : np.ndarray[Any, np.dtype[np.str_]] = vectorizer.get_feature_names_out() # type: ignore
            print(f"Vectorizer for {event_id} feature names: {feature_names}")


def forward_vectorize(event : Event, event_id : event_id_t) -> np.ndarray[Any, Any]:

  print("forward_vectorize called for ", event_id, " with context ", event["context"])
  
  # AI : Vectorizer should have been initialized by initialize_vectorizers
  if event_id not in key_to_vectorizer:
      # AI : This case should ideally not be hit if initialize_vectorizers is called correctly.
      # AI : However, as a fallback, fit with the current single context.
      # AI : This might lead to issues if this event_id has other context structures not seen yet.
      print(f"Warning: Vectorizer for {event_id} not pre-initialized. Fitting with single instance: {event['context']}")
      key_to_vectorizer[event_id] = DictVectorizer(sparse=False)
      # AI : Preprocess context before fitting and transforming
      processed_context = preprocess_context(event["context"])
      key_to_vectorizer[event_id].fit([processed_context]) # type: ignore
  else:
      processed_context = preprocess_context(event["context"])


  vectorizer : DictVectorizer = key_to_vectorizer[event_id]

  # AI : This assertion is no longer directly applicable because get_feature_names_out()
  # AI : will return names like 'key=value' for stringified booleans,
  # AI : while event["context"].keys() are the original keys.
  # AI : The pre-fitting strategy in initialize_vectorizers ensures vocabulary consistency.
  # assert set(processed_context.keys()) == set(vectorizer.get_feature_names_out()) # type: ignore
  
  arr : np.ndarray[Any, Any] = vectorizer.transform([processed_context]) # type: ignore
  assert(isinstance(arr, np.ndarray))
  return arr


def reverse_vectorise(vector : np.ndarray[Any, Any], event_id : event_id_t) -> list[json_t]:

  assert(event_id in key_to_vectorizer.keys())

  vectorizer : DictVectorizer = key_to_vectorizer[event_id]
  # AI : inverse_transform returns a list of dictionaries with feature names like 'key=value'
  # AI : and values (typically 1.0 for present categorical features).
  processed_contexts_list : list[json_t] = vectorizer.inverse_transform(vector) # type: ignore
  assert(isinstance(processed_contexts_list, list))

  # AI : Reconstruct original context structure from the processed contexts
  reconstructed_contexts: list[json_t] = [reconstruct_context(pc) for pc in processed_contexts_list]
  return reconstructed_contexts


# AI : Call initialize_vectorizers before starting the tests
initialize_vectorizers(TRAINING_DATA_WITH_CONTEXT)


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




