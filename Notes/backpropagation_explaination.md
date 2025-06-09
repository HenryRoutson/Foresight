


# Model prediction

```json
[
    {
        "outcome_name": "instagram_post",
        "probability": 0.35,
        "time_diff_minutes": 45,
        "outcome_data": {
            "num_images": 2,
            "num_videos": 0,
            "has_caption": true,
        }
    },
    {
        "outcome_name": "like_post",
        "probability": 0.42,
        "time_diff_minutes": 12,
        "outcome_data": {
            "post_owner_is_follower": true,
        }
    },
    {
        "outcome_name": "delete_account",
        "probability": 0.05,
        "time_diff_minutes": 2880,
        "outcome_data": {
            
        }
    },
    {
        "outcome_name": "view_story",
        "probability": 0.18,
        "time_diff_minutes": 8,
        "outcome_data": {
            "story_owner_is_follower": true,
            "view_duration_seconds": 3.2
        }
    }
]
```

in a single vector this looks like


```python
expected = [

  # instagram_post outcome -------------------
  0.35, # probability
  45, # time_diff_minutes
  2, # num_images
  0, # num_videos
  1, # has_caption

  # like_post outcome -------------------
  0.42, # probability
  12, # time_diff_minutes
  1, # post_owner_is_follower

  # delete_account outcome -------------------
  0.05, # probability
  2880, # time_diff_minutes

  # view_story outcome -------------------
  0.18, # probability
  8, # time_diff_minutes
  1, # story_owner_is_follower
  3.2 # view_duration_seconds
]
```

# Actual outcome


```json
[
    {
        "outcome_name": "instagram_post",
        "time_diff_minutes": 140,
        "outcome_data": {
            "num_images": 4,
            "num_videos": 1,
            "has_caption": false,
        }
    },
]
```

# Error correction

Error is calculated for each output neuron by the difference between the value for the expected and actual value.

### Naive implimentation

This is an example of a vector containing the values you want the model to output.
If you don't use loss masking the model predict 0 for almost all data and you would have terrible performance.

```python
actual = [

  # instagram_post outcome -------------------
  1.0, # probability
  140, # time_diff_minutes
  4, # num_images
  1, # num_videos
  0, # has_caption

  # like_post outcome -------------------
  0, # probability
  0, # time_diff_minutes
  0, # post_owner_is_follower

  # delete_account outcome -------------------
  0, # probability
  0, # time_diff_minutes

  # view_story outcome -------------------
  0, # probability
  0, # time_diff_minutes
  0, # story_owner_is_follower
  0 # view_duration_seconds
]

expected = [

  # instagram_post outcome -------------------
  0.35, # probability                          # NOT MASKED
  45, # time_diff_minutes                      # NOT MASKED
  2, # num_images                              # NOT MASKED
  0, # num_videos                              # NOT MASKED
  1, # has_caption                             # NOT MASKED

  # like_post outcome -------------------
  0.42, # probability                          # NOT MASKED
  500, # time_diff_minutes                     # NOT MASKED
  2, # post_owner_is_follower                  # NOT MASKED

  # delete_account outcome -------------------
  0.05, # probability                          # NOT MASKED
  5200, # time_diff_minutes                    # NOT MASKED

  # view_story outcome -------------------
  0.18, # probability                          # NOT MASKED
  5, # time_diff_minutes                       # NOT MASKED
  1, # story_owner_is_follower                 # NOT MASKED
  2 # view_duration_seconds                    # NOT MASKED
]

expected # same as above
```

### With loss masking



```python

expected = [

  # instagram_post outcome -------------------
  0.35, # probability                          # NOT MASKED
  45, # time_diff_minutes                      # NOT MASKED
  2, # num_images                              # NOT MASKED
  0, # num_videos                              # NOT MASKED
  1, # has_caption                             # NOT MASKED

  # like_post outcome -------------------
  0.42, # probability                          # NOT MASKED
  500, # time_diff_minutes                     # MASKED
  2, # post_owner_is_follower                  # MASKED

  # delete_account outcome -------------------
  0.05, # probability                          # NOT MASKED
  5200, # time_diff_minutes                    # MASKED

  # view_story outcome -------------------
  0.18, # probability                          # NOT MASKED
  5, # time_diff_minutes                       # MASKED
  1, # story_owner_is_follower                 # MASKED
  2 # view_duration_seconds                    # MASKED
]
```


Note that the probabilities are not masked,
only the data from outcomes which did not occur.