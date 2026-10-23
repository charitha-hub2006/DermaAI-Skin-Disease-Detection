# Models Directory

This directory stores trained model files, label mappings, and training configurations.

## Expected Files

- `skin_disease_cnn.keras`: The serialized Keras/TensorFlow model containing weights and architecture.
- `class_indices.json`: Maps class index integers (0-6) to their class acronyms and human-readable names.
- `training_config.json`: Metadata capturing hyperparameter details (image dimensions, batch size, learning rate, epochs trained).

## Notes

- This repository does **NOT** fabricate random model weights.
- When no trained model exists, the Flask web application gracefully informs the user that training has not yet taken place and provides commands to execute it.
- Never deploy a model without running `training/train.py` or running the verified smoke-test training.
