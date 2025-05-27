



this will be a simple example of data which might be useful for this model to predict

we will compare using a naieve tranformer with our method

the task is simple



TODO finish


events

  has_surgery
    surgery_duration_hours : float

  has_heart_attack
    severity : float




Example

  has_heart_attack
    severity : 0.1

  has_surgery
    surgery_duration_hours : 5.1

  has_heart_attack
    severity : 0.3




Note that we are not only predicting if someone will have a heart attack,
but also useful data to evaluate the heart attack like the severity



How lets set up a way to generate some plausable data



everyone will start with a heart attack, 
and the surgery is meant to 














TODO need to make sure this isn't simple to just use basic analysis to discover
ie can't just look at a simple line graph and see look here. needs to interfere with multiple variables and also use the attention mechanism of a tranformer.