# NLI_Project
This repository contains the code and research developed for the final project of the NLI class of the EMAI's Master's degree. The purpose of the project was to implement the components (most of them fully, a few only partially) of an interactive Conversational Agent (CA) to recommend restaurants and hotels using data from the multiwoz dataset. The objective of the CA is mainly to recommend and book hotels and restaurants as requested per the user.

The code contains three major components:

- **Dialogue act identification**: This task consists on prediction the DA of the written utterances of a user in order to guide the CA in identifying the main ideas the user is communicating.
- **Information extraction** (a.k.a slot filling): This task consists of the extraction of information from the written utterances of a user. To do so it uses a series of domain specific slots which are used to categorize the information provided by the user.
- **Planning**: This part of the assignment consists on the three following subtasks related to planning a response to the user from the information given and requested by the user:
  - **Information retrieval**: This tasks consists on identifying from the previous utterances of a dialogue the slots that need to be retrieved by the agent. Since we didn't have full access to a database of hotel, it was decided to just plan the information that should be looked for (e.g the parameters of the query to a possible database of hotel and/or restaurants).
  - **DA prediction**: This tasks consists on predicting the DA the agent should response to the user with.
  - **To be requested**: This tasks consists of identifying the missing information the agent should require to the user in order to complete its task.

# Contents

The following's a list with the contest of the repository:

- **agent.py**: interactive python application to interact with the conversational agent

- **classification/**: Folder containing all the code and training notebooks of user and agent DA prediction.

- **slot_extraction/**: Folder containing all the code and training notebooks of the two approaches to the slot filling task.

- **info_to_be_retrieved.py**: Module containing the code for the information retrieval task.

- **info_to_be_retrieved/**: Folder containing the training script and results of the information retrieval task.

- **info_to_be_requested.py**: Module containing the code for the task of requesting information to the user.

- **info_to_be_requested/**: Folder containing the training script for the info to be requested task.

- **dataset1.hf**: Multiwoz Dataset.

- **NLI_notebook_for_project_evaluation_EMAI_(task_3_updated).ipynb**: Last version of the evaluation notebook adapted to use all the modules implemented for the task, as well as for removing the question identification slots (slots with the value '?') from the metrics calculation.

- **NLI Final Presentation Report.pdf**: Slide presentation of the whole project.

# Results

The results are contained in the following files, which also contain the training code of the models:

- **User Dialogue Act Prediction**
  - classification/da_prediction.ipynb

- **Slot Filling**:
  - NER approach (NER scores training and validation): slot_extraction/ner_approach/all_resuts.json
  - NER approach (global results): slot_extraction/ner_approach/ner_approach.ipynb
  - Similarity approach: slot_extraction/similarity_approach/similarity_approach.ipynb

- **Information to be Retrieved**:
  - info_to_be_retrieved/Evaluation_Model.ipynb

- **Agent Dialogue Act Prediction**
  - classification/context_classifier.ipynb

- **Information to be Retrieved**
  - info_to_be_requested/train_info_to_be_requested_training.ipynb