



#  --------------------------------------------------------

# F is short for B{False}
# T is short for B{True}

# TODO expand to object with data

TRAINING_DATA_WITH_CONTEXT = [
  ["A", "A", "A"],
  ["A", "F", "F"],
  ["A", "T", "T"],
  ["F", "A", "A"],
  ["F", "F", "A"],
  ["F", "T", "F"],
  ["T", "A", "T"],
  ["T", "F", "F"],
  ["T", "T", "T"]
]

DATA_WITH_CONTEXT_SET : set[str] = set()
for data in TRAINING_DATA_WITH_CONTEXT:
  for x in data :
    DATA_WITH_CONTEXT_SET.add(x)

assert len(DATA_WITH_CONTEXT_SET) == 3 # A, F, T



#  --------------------------------------------------------


TRAINING_DATA_WITHOUT_CONTEXT = [
  ["A", "A", "A"],
  ["A", "B", "B"],
  ["A", "B", "B"],
  ["B", "A", "A"],
  ["B", "B", "A"],
  ["B", "B", "B"],
  ["B", "A", "B"],
  ["B", "B", "B"],
  ["B", "B", "B"]
]



for without_context, with_context in zip(TRAINING_DATA_WITHOUT_CONTEXT, TRAINING_DATA_WITH_CONTEXT):
  assert without_context == [x.replace("T", "B").replace("F", "B") for x in with_context]




DATA_WITHOUT_CONTEXT_SET : set[str] = set()
for data in TRAINING_DATA_WITHOUT_CONTEXT:
  for x in data :
    DATA_WITHOUT_CONTEXT_SET.add(x)

assert len(DATA_WITHOUT_CONTEXT_SET) == 2 # A and B



#  --------------------------------------------------------


