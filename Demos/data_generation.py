


TRAINING_DATA_WITH_CONTEXT = [
  ["A", "A", "A"],
  ["A", "B{False}", "B{False}"],
  ["A", "B{True}", "B{True}"],
  ["B{False}", "A", "A"],
  ["B{False}", "B{False}", "A"],
  ["B{False}", "B{True}", "B{False}"],
  ["B{True}", "A", "B{True}"],
  ["B{True}", "B{False}", "B{False}"],
  ["B{True}", "B{True}", "B{True}"]
]

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
