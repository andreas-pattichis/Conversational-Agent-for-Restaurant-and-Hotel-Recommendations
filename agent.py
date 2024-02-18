import argparse

import classification.classifier as da_prediction
import classification.da_agent_classifier as agent_da_prediction
import slot_extraction.utilities.slots as slt
from info_to_be_requested import Info_tb_requested_util
from info_to_be_retrieved import Info_tb_retrieved_util

# I set this variable global to avoid some nltk legends at the beginning
inf_tb_retrieved_util = Info_tb_retrieved_util()
inf_tb_requested_util = Info_tb_requested_util()

"""
Conversational agent of UPF's NLI class.

This a partial implementation of a CLI application implementing an interactive dialogue agent. The agent can recognize the intention
of the user, extract relevant information from its utterances and plan forward what useful information it can deliver to them. The par-
ticular domain this agent was trained for is as a booking service for restaurants and hotels using the multiwoz v2 dataset.
"""

model_path = './models/bilstm.tf'
tokenizer_path = './models/tokenizer.pkl'
model_path2 = './models/da_roRERTa.tf'

bilstm_classifier = da_prediction.BiLSTMClassifier(model_path, tokenizer_path)
agent_classifier = agent_da_prediction.DA_Agent_Classifier(model_path2)


def Dialogue_Act_Prediction(user_utterance, prev_utterances):
    return bilstm_classifier.classify_utterance(user_utterance, prev_utterances)


def Extract_and_Categorize_Spans(user_utterance, user_dialogue_acts, other_features_from_dialogue_history):
    # extracted_information = [('hotel-bookpeople', '2'), ('hotel-bookstay', '2'), ('hotel-bookday', 'sunday'), ('restaurant-phone', '?')]
    return slt.predict_ner(user_utterance)


def Information_to_be_retrieved_Prediction(user_dialogue_acts, extracted_information,
                                           other_features_from_dialogue_history3):
    sentence_preprocessed = inf_tb_retrieved_util.pre_process(other_features_from_dialogue_history3['user_utt'])

    text = other_features_from_dialogue_history3['past_info'] + "|" + sentence_preprocessed
    text_2 = inf_tb_retrieved_util.truncate_from_start(text)

    text_embbedings = inf_tb_retrieved_util.get_embbedings(text_2)
    output = inf_tb_retrieved_util.get_prediction(text_embbedings)
    to_be_retrieved = inf_tb_retrieved_util.from_1_hot_to_labels(output)
    return to_be_retrieved, text_2


def Agent_Move_Prediction(user_dialogue_acts, extracted_information, retrieved_information, utterance,
                          previous_utterances, other_features_from_dialogue_history4):
    # Predict agent's dialogue acts
    agent_dialogue_acts = agent_classifier.classify_utterance(utterance, previous_utterances)

    # Combine relevant information for prediction
    info_processed = inf_tb_requested_util.pre_process(other_features_from_dialogue_history4['user_utt'])
    combined_info_processed = other_features_from_dialogue_history4['past_info'] + "|" + info_processed
    combined_info_truncated = inf_tb_requested_util.truncate_from_start(combined_info_processed)

    # Predict 'to_be_requested' information
    to_be_requested_predictions = inf_tb_requested_util.get_prediction(combined_info_truncated)
    # raw_scores = inf_tb_requested_util.get_prediction(combined_info_truncated, return_raw_scores=True)
    # print(raw_scores)
    to_be_requested = inf_tb_requested_util.from_1_hot_to_labels(to_be_requested_predictions)

    return {"agent_dialogue_acts": agent_dialogue_acts,
            "to_be_requested": to_be_requested}, combined_info_truncated


class Dialogue_History():
    """
    Class implementing a Dialogue history. It's used to record an interactive dialogue with a user, so the Conversational Agent's
    responses are informed of what happened in the past.

    Attributes:
        data: Dictionary containing the information of each turn in the recorded dialogue history.
        turn_counter: Integer that counts the numbers of turns in a dialogue.
    """

    def __init__(self):
        self.data = {}
        self.turn_counter = 0

    def add_turn(self, utt, user_das, extracted_slots, retrieved, agent_das, to_be_requested):
        """
        Adds a turn to the Dialogue history.

        Adds a turn to the data of the Dialogue history by identifying it with the current dialogue counter in the dictionary. The turn 
        is created from the input parameters of the function.
        
        Args:
            utt: String containing the user utterance of the current turn of the history.
            user_das: Predicted dialogue act of the user from the current turn's utterance.
            extracted_slots: List of tuples of strings containing the slots extracted from the current turn's utterance.
            retrieved: List strings containing the slots to be retrieved by the CA in response to the current turn's utterance.
            agent_das: List of strings representing the dialogue acts of the response of the CA to the current user utterance.
            to_be_requested: List of strings representing the slots to be requested from the user in response to the current user intervention. 
        """

        self[self.turn_counter] = {'utterance': utt,
                                   'user_das': user_das,
                                   'extracted_slots': extracted_slots,
                                   'to_be_retrieved': retrieved,
                                   'agent_das': agent_das,
                                   'to_be_requested': to_be_requested}
        self.turn_counter += 1

    def __getitem__(self, t):
        if t >= 0:
            return self.data[t]
        return None

    def __setitem__(self, t, val):
        self.data[t] = val

    def get_previous_values(self, property):
        """
        Returns a list with all values for a given property from all the turns captured by the dialogue history.

        Args:
            property: String representing property of the history whose list of previous values wants to be retrieved.

        Returns:
            List of previous values in the history of the property passed as argument.
        """

        return [self.data[turn][property] for turn in self.data]

    def __repr__(self) -> str:

        str_rpr = ""
        for key in self.data:
            str_rpr += f"-----Turn {key}-----\n"
            str_rpr += f"utterance: {self[key]['utterance']}\n"
            str_rpr += f"user_das: {self[key]['user_das']}\n"
            str_rpr += f"extracted_slots: {self[key]['extracted_slots']}\n"
            str_rpr += f"to_be_retrieved: {self[key]['to_be_retrieved']}\n"
            str_rpr += f"agent_das: {self[key]['agent_das']}\n"
            str_rpr += f"to_be_requested: {self[key]['to_be_requested']}\n"
            # str_rpr+="\n"
        return str_rpr

    def print_turn(self, turn):
        """
        Prints the properties of a specific turn of the Dialogue History in the standard output.

        Args:
            turn: Int representing the turn that wants to be printed
        """

        turn_data = self[turn]
        str_rpr = ""
        str_rpr += f"-----Turn {turn}-----\n"
        str_rpr += f"utterance: {turn_data['utterance']}\n"
        str_rpr += f"user_das: {turn_data['user_das']}\n"
        str_rpr += f"extracted_slots: {turn_data['extracted_slots']}\n"
        str_rpr += f"to_be_retrieved: {turn_data['to_be_retrieved']}\n"
        str_rpr += f"agent_das: {turn_data['agent_das']}\n"
        str_rpr += f"to_be_requested: {turn_data['to_be_requested']}\n"
        print(str_rpr)

    def print_current_turn(self):
        """
        Prints the properties of the current turn of the Dialogue History in the standard output.

        Prints the properties of the current turn of the Dialogue History in the standard output as indicated by the current value of
        the turn_counter attribute of the class.
        """
        self.print_turn(self.turn_counter - 1)

    def get_current_turn(self):
        return self[self.turn_counter - 1]


def main(args):
    if (args['file'] is not None):
        print('Non-interactive mode')
    else:
        print('##########INTERACTIVE MODE BEGINS##########')
        print('Type "exit" or EOF to exit the program.')
        try:
            history = Dialogue_History()
            print(history)
            utt = input("Hello! How can I help you?\n>> ")
            past_info_iftbr = ''
            past_info_iftbreq = ''
            while (utt != 'exit'):
                print("Dummy response.")

                # need to add the turn_id
                user_das = Dialogue_Act_Prediction(utt, history.get_previous_values('utterance'))
                extracted_slots = Extract_and_Categorize_Spans(utt, user_das, other_features_from_dialogue_history={})
                to_be_retrieved, utt_processed = Information_to_be_retrieved_Prediction(user_das, extracted_slots,
                                                                                        other_features_from_dialogue_history3={
                                                                                            'user_utt': utt,
                                                                                            'past_info': past_info_iftbr})
                past_info_iftbr = utt_processed + '|' + ','.join(to_be_retrieved)

                agent_moves, combined_info_truncated = Agent_Move_Prediction(user_das, extracted_slots, {}, utt,
                                                    history.get_previous_values('utterance'),
                                                    other_features_from_dialogue_history4={'user_utt': utt,
                                                                                           'past_info': past_info_iftbreq})
                agent_das = agent_moves['agent_dialogue_acts']
                to_be_requested = agent_moves['to_be_requested']
                utt_processed = combined_info_truncated
                past_info_iftbreq = utt_processed + '|' + ','.join(to_be_requested)

                history.add_turn(utt, user_das, extracted_slots, to_be_retrieved, agent_das, to_be_requested)
                history.print_current_turn()

                utt = input(">> ")
            else:
                print('Good Bye')
        except EOFError:
            print('Good Bye!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='NLI Conversational Agent',
        description='This program is a Conversational Agent\
                        designed to recommend restaurants and hotels. It\'s only\
                        a prototype, so no full answers are given.',
        epilog='Created by Pedro Moreira, Manfred Gonzalez,\
                        Andreas Pattichis and Joaquín Figueira')
    parser.add_argument('--file', type=str, nargs='?',
                        help='File containing utterances separated by a new line character which are to be\
                        processed non-interactively. All utterances will be considered\
                        the same dialogue.')
    args = parser.parse_args()
    print(args)
    main(vars(args))
