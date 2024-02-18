# Conversational Agent for Restaurant and Hotel Recommendations

<p align="center">
  <img src="https://bodyswitch.com.au/wp-content/uploads/2022/08/nlp-logo.png" width="200" alt="Conversational Agent">
</p>

This repository contains the code and research developed for our Conversational Agent (CA) project, as the final project of our coursework in the Natural Language Interaction (NLI) class within the EMAI's Master's degree program. The purpose of the project was to implement the components (most of them fully, a few only partially) of an interactive Conversational Agent (CA) to recommend restaurants and hotels using data from the multiwoz dataset from Hugging Face 🤗. The objective of the CA is mainly to recommend and book hotels and restaurants as requested per the user.

---

## Project Overview

The development of our Conversational Agent focused on three critical components, each contributing to the agent's ability to understand, process, and respond to user requests effectively. These components include:

1. **Dialogue Act Identification**: A predictive model that analyzes user utterances to identify the Dialogue Act (DA), guiding the CA in recognizing the user's intentions and main ideas.

2. **Information Extraction (Slot Filling)**: This process involves extracting pertinent information from user utterances using domain-specific slots. These slots categorize the user-provided data, facilitating accurate response generation.

3. **Planning**: Our planning module is subdivided into three tasks, each designed to optimize the CA's response strategy based on user input:
  - **Information Retrieval**: Identifies the necessary information to be retrieved based on previous dialogue, planning queries for potential databases.
  - **DA Prediction**: Predicts the appropriate Dialogue Act the agent should use in response to the user.
  - **To be Requested**: Determines what additional information is required from the user to fulfill their request effectively.

---

## Repository Contents

This repository is organized to provide a comprehensive overview of our project's components, including code, training notebooks, and evaluation results:

- `agent.py`: An interactive Python application for interfacing with the CA.
- `classification/`: Contains code and training notebooks for both user and agent Dialogue Act prediction.
- `slot_extraction/`: Hosts the code and training notebooks for our innovative slot filling approaches.
- `info_to_be_retrieved.py`: The module dedicated to the information retrieval task.
- `info_to_be_requested.py`: Implements the functionality for requesting additional information from the user.
- `dataset1.hf`: The MultiwoZ Dataset, the backbone of our training and evaluation processes.
- `NLI_notebook_for_project_evaluation_EMAI_(task_3_updated).ipynb`: The latest evaluation notebook, showcasing the integration of our modules and the refinement of metrics calculation.

---

## Results and Evaluations

Our project's outcomes are documented in several key files within this repository, offering insights into the training and performance of our models:

- **User Dialogue Act Prediction**: `classification/da_prediction.ipynb`
- **Slot Filling**:
  - NER approach: `slot_extraction/ner_approach/all_results.json` and `slot_extraction/ner_approach/ner_approach.ipynb`
  - Similarity approach: `slot_extraction/similarity_approach/similarity_approach.ipynb`
- **Information to be Retrieved**: `info_to_be_retrieved/Evaluation_Model.ipynb`
- **Agent Dialogue Act Prediction**: `classification/context_classifier.ipynb`
- **Information to be Requested**: `info_to_be_requested/train_info_to_be_requested_training.ipynb`
