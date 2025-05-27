# Multi-Outcome Event Prediction Library

## Core Concept

This library enables sequence modeling for predicting multiple discrete future outcomes simultaneously, where each outcome has its own probability, timing optionally, and associated structured data. Given a sequence of historical events, the model outputs a comprehensive prediction set containing all possible next outcomes with their respective probabilities and predicted data payloads.

## Key Innovation: Unified Multi-Outcome Prediction

### Problem Statement
Traditional sequence models typically predict either:
- A single most likely next event (classification)
- A probability distribution over event types only (without associated data)
- Continuous values for a single metric (time series forecasting)

This library addresses the gap of predicting **multiple heterogeneous discrete outcomes simultaneously**, each with:
- Event type probability
- Complete structured data payload specific to that event type including time detla optionally


## Architecture Overview

### Model Agnostic Design
While transformers are a natural fit, the library supports multiple model architectures:
- **Transformers**: Self-attention mechanisms for capturing long-range dependencies in event sequences
- **RNNs/LSTMs**: For sequential processing with memory
- **CNNs**: For pattern recognition in temporal sequences

The core requirement is the ability to process sequential data and output structured predictions for multiple outcomes.

#### Dynamic Output Structure
Different outcomes can have completely different data schemas


## Masked Loss Function: Core Innovation

### Problem with Standard Loss Functions
Traditional multi-task learning applies loss to all predicted outputs, which is inappropriate when:
- Only one outcome actually occurs per prediction step
- Predicting data for non-occurring outcomes has no ground truth
- Standard loss would penalize accurate predictions for unobserved outcomes


## Other notes

### Why you shouldn't use specialised heads for each outcome

Each specialized head needs its own data to learn underlying patterns, even if there is lots of overlap. For example ... 
- 'Create post' head learns "user engagement patterns" 
- 'Like' head learns "user engagement patterns"
- 'Comment' head learns "user engagement patterns"

With seperate heads there is no **transfer Learning** Between Outcomes
