
There is a difficult problem with sub outcomes.

if i have something like


Delete_google_account
  reason : Enum { slow_results, privacy, bad_results, Other }
  moving_to_competitor : Enum { Bing, DuckDuckGo, Yahoo!, Other }


There are a lot of possible sub outcomes. 
I shouldn't flatten this so that each possible outcome like


  Delete_google_account
  reason : Privacy
  moving_to_competitor : Bing


is it's own event.
This would become sparse as each event might not occur.
This would also be complicated.

However,
if i was to one hot encode each of these outcomes, 
there would need to be some specialised softmax logic to make sure the output adds up to one for each OHE enum group.


I should also note that as mentioned in other areas in my notes i do not want to use specialised heads as this removes transfer learning. 