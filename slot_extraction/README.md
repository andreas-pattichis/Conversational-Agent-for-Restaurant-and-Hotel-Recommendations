# Slot Filling Task

This folder contains all the code and results of the slot filling task of the CA.

## Contents:

- Manual Tokenization and BIO tagging/: This folder contains the notebook used to construct the CoLLN 2003 format used as input for the run_ner script.
- ner_approach/: This folder contains the evaluation and results of the BERT NER approach for slot filling. It also contains the training data that can be used to recreate our model.
- similarity_approach/: This folder contains the code, evaluation and best parameters pickle file for developing the similarity system used for slot extraction.
- utilities/slot.py: This is the main module of the task containing all necessary functions and dependencies used by both approaches. It was designed to by importable as a module to have quick access to all the functionalities of the slot extraction systems.
- slot_value_analysis.ods: This a spreadsheet with a complete list of the values for each of the Hotel and Restaurant Frames.