




### Social Media Platforms

Instagram example


  Open App
    duration_spent : float

  Create post
    number of photos : int
    number of videos : int

  Like
    is_account_following_user : bool

  Comment
    comment_length : int
    comment_sentiment : float

  Share_post


  Delete account
    reason_for_account_deletion : enum

  ect


How might you use this?

  A social media platform wanting to predict if a user might delete an account for mental health reasons,
  might look at how inserting different events might impact the likelyhood of this outcome.

  And might find that users who like people they follow are less likely to report mental health issues.
  The company can then direct user behaviour to interact more with people they follow.

  This would be tested with these seqeunces 

    Open App -> Like { is_account_following_user : 1 } 
    Open App -> Like { is_account_following_user : 0 }

  And then looking at how the probability of account deletion changes.
  reason_for_account_deletion has multiple outcomes and so will be one hot encoded.

  Note that there will be a probabilty for the outcome ie deleting of account and an internal probabilty for each sub outcome.
  TODO need to think about this - sub outcomes will be useful to model, might support later

  However this is not a good way to impliment this as softmax might be difficult to impliment 
  and there isn't the flexibility to add sub data for each reason for account deletion.
  For that reason there should be no enums which aren't broken up into sub outcomes.
  


TODO format this and finish




predicting medical outcomes (take with a grain of salt)

  TODO



  ie surgery -> outcome




predicting consumer pricing

  how does changing the price for a specific user change their probability of purchasing???? TODO



quant trading

  how does changing a ceo affect share price
  how does changing ownership affect share price


and add more