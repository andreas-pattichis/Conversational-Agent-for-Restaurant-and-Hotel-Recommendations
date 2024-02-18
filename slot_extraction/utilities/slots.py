import pandas as pd
from collections import Counter
from datetime import datetime,  timedelta
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from transformers import TokenClassificationPipeline, pipeline
import compress_fasttext
import pickle
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import sys

"""
Module containing all the code functions implemented for the slot extraction task of the project.

This module contains all the necessary codes and dependencies imports used for the slot extraction task of the NLI project.
It's comprised mainly of dataset manipulation functions, as well as two different sequence labelling implementation and all
their auxiliary code. Finally, it also contains the initialization of all the pre-trained ML models used for the task.

"""


# Loads the compressed fast text model with 300 dimensions trained on common crawl data
fast_text = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin'
)

# Path to the NER transformer sequence classifier.
NER_MODEL_PATH = "models/slot_extraction/ner_model/"
SIMILARITY_MODEL_PATH = 'models/slot_extraction/similarity_approach/best_similarity_params.pickle'

# Values of the categorical slots.
ini = datetime.strptime('00:00', '%H:%M')
delta=timedelta(minutes=15)
categorical_values = {
    'hotel':
    {
        'type': ['dontcare', 'hotel', 'guesthouse'],
        'bookstay': ['1','2','3','4','5', '6', '7'],
        'bookpeople': ['1','2','3','4','5','6', '7', '8', '9', '10'],
        'stars': ['0','1','2','3','4','5'],
        'pricerange': ['dontcare', 'cheap', 'moderate', 'expensive'],
        'area': ['dontcare','east','south', 'west', 'north', 'centre'],
        'bookday': ['monday', 'thursday', 'wednesday', 'tuesday', 'friday', 'saturday', 'sunday']
    },
    'restaurant':
    {
        'bookpeople': ['1','2','3','4','5','6', '7', '8', '9', '10'],
        #'booktime': [datetime.strftime(ini+i*delta, "%H:%M") for i in range(24*4)],
        'pricerange': ['dontcare', 'cheap', 'moderate', 'expensive'],
        'area': ['dontcare','east','south', 'west', 'north', 'centre'],
        'bookday': ['monday', 'thursday', 'wednesday', 'tuesday', 'friday', 'saturday', 'sunday'] 
    }
}

# Spacey tokenizer used in the similarity approach.
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

# Auxiliary function used to load pickle saved models.
def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# Loads the similarity approach model for sequence classification. It's basically a simple dictionary containing the best 
# representative, sensitivity and ngram size for each slot.
    
similarity_model = load_data(SIMILARITY_MODEL_PATH)

def extract_domain_from_slot(dact):
    return dact.split('-')[0].lower()

def similarity_classifier(dom_slot, val, embedder):

    """
    Function implementing a simple categorical value classifier for a given slot.

    This function implements a simple slot value classifier for a given value and categorical slot, and works by calculating the embedding space
    distance between the provided value and each of the possible categorical values of the slot and returning the slot's categorical value nearest
    to the provided input value. It uses a compressed version of FastText to calculate the embeddings.

    Args:
        dom_slot: Slot of the value to be classified.
        val: Value to be classified.

    Returns:
        String representing the predicted category of the input.
    """    
    if(len(dom_slot.split('-'))!=2):
        print(f"Extrange dom_slot is: {dom_slot} with value: {val}")
        dom = dom_slot
        slot = ""
    else:
        dom, slot = tuple(dom_slot.split('-'))

    if extract_domain_from_slot(dom) in categorical_values:

        if(slot in categorical_values[dom]):
            categories = categorical_values[dom][slot]
        else: return val

        min_distance = float('inf')
        best_category = ''

        for cat in categories:
            cat_dist = embedder.distance(cat, val)
            if(cat_dist < min_distance):
                min_distance = cat_dist
                best_category = cat
        
        return best_category
    
    else: return val


def filter_dataset(split, filter_list=['hotel', 'restaurant']):
    """
    Filters a split of the multiwoz dataset using the given domains.

    Filters a split of the multiwoz dataset using the given domains. It returns only the dialogues that contain any utterance of the domains
    in the filter_list argument. 

    Args:
        split: Split of the huggingface dataset containing multiwoz.
        filter_list: List of strings containing the domains to preserve when filtering the dataset.

    Returns:
        Returns the filtered split where all the turns that don't contain any utterance inside the domains of the filter_list argument.
    """

    filtered = [dial for dial in split 
                if any(set(dial['turns']['frames'][turn_id]['service']).intersection(filter_list) 
                    for turn_id,utt in enumerate(dial['turns']['utterance']))]
    
    return filtered

def extract_gt(row):
    """
    Auxiliary function to extract the list of slot ground truths of the Pandas Dataframe containing the slot info.
    
    Args:
        row: A pd.Series containing a single cell from the dataframe containing the ground truth of a constructed slot's row.

    Returns:
        Returns a list of tuples containing the list of ground truths of slot-value pairs of the current row.
    """

    gt_list = []
    for key,(val,_) in row.items():
        gt_list.append((key,val))
    return gt_list


def construct_slot_extraction_data(dataset):

    """
    Constructs a dataframe with all the relevant information for the slot extraction task from a given multiwoz split.

    This function constructs a Pandas dataframe with all the information necessary for the slot extraction task. It's most important
    task is flattening the dictionary structure of the split, by extracting the spans of each slot in a single column.

    Args:
        dataset: multiwoz split to be transformed.

    Returns:
        Returns a pandas Dataframe where each row contains the information of one utterance. It has the following columns:
            dialogue_id: id (sting) of the dialogue containing the utterance.
            turn_id: id (integer) of the turn of the utterance.
            utterance: String of the utterance.
            intent: List of strings containing the intents of the utterance.
            values: List of tuples containing the slots. Each slot is represented as a tuple containing its name, value and span.
    """

    #Flattening the structure, extracting dialogue IDs, turns IDs, list of DAs and slot name/value pairs.
    base_slot_data = [(dial['dialogue_id'], turn_id, dial['turns']['utterance'][turn_id],
                    #Collecting DAs. 
                    set(act['span_info']['act_type']),
                    #Collecting slot name/value pairs in a dictionary with the slot names (plus the domain) as keys and the value and their span as values.
                    dict(zip([f"{intent.split('-')[0].lower()}-{slot_name}" #Collecting slots' name and joinning them with the DA 
                                    for intent,slot_name
                                    #Combines DA and slot names lists together into a list of tuples. 
                                    in zip(act['span_info']['act_type'], act['span_info']['act_slot_name'])], 
                                [(value,(span_begin,span_end)) #Collecting slots' values and their spans 
                                    for value,span_begin,span_end 
                                    # Combines values, span_start and span_end lists together to create a list of tuples
                                    in zip(act['span_info']['act_slot_value'],act['span_info']['span_start'], act['span_info']['span_end'])])))
                        for dial in dataset
                            for turn_id, act in enumerate(dial['turns']['dialogue_acts'])
                                #Only keeping utterances with at least one DA of Hotel, Restaurant or general 
                                if any([intent.startswith('general') or intent.startswith('Hotel') or intent.startswith('Restaurant') 
                                        for intent in act['dialog_act']['act_type']]) 
                                #Filtering out system utterances
                                and dial['turns']['speaker'][turn_id] == 0
                ]

    #Doing the same but only for slot name/value pairs that don't appear in span_info dict of the turns (with name!='none' and value=='?')
    #Rather, they appear in the dialog_act dictionary.
    base_slot_data_questions = [(d_id, turn_id, utt, intent, slot_values)  for (d_id, turn_id, utt, intent, slot_values) in 
                                [(dial['dialogue_id'], turn_id, dial['turns']['utterance'][turn_id],
                                #Collecting DAs
                                set([act_type  
                                    for (act_type, act_slots) in zip(acts['dialog_act']['act_type'], acts['dialog_act']['act_slots'])
                                    for slot_name, slot_value in zip(act_slots['slot_name'], act_slots['slot_value']) 
                                    #Only keeping DA's for slots whose name is not 'none' and whose value is '?' (this is what the evaluation loop does)
                                    #since the rest of the slot name/value pairs are in the span info dict.
                                    if slot_value =="?" and slot_name != 'none']),
                                #Collecting slot name/value pairs #Collecting slot name/value pairs in a dictionary with the slot names (plus the domain)
                                #as keys and the value and their span as values (the span is always None since the value is not in the utterance).
                                dict([(f"{act_type.split('-')[0].lower()}-{slot_name}",(slot_value,None)) #Creating tuples of slot names and values    
                                    for (act_type, act_slots) in zip(acts['dialog_act']['act_type'], acts['dialog_act']['act_slots'])#for each DA in acts
                                    for slot_name, slot_value in zip(act_slots['slot_name'], act_slots['slot_value']) #for each pair of name and values of the DA
                                    #Only keeping name/value pairs whose name is not 'none' and whose value is '?' (this is what the evaluation loop does)
                                    #since the rest of the slot name/value pairs are in the span info dict.
                                    if slot_value =="?" and slot_name != 'none']))
                                    
                        for dial in dataset 
                            for turn_id, acts in enumerate(dial['turns']['dialogue_acts']) 
                            #Only keeping utterances with at least one DA of Hotel, Restaurant or general
                            if any([intent.startswith('general') or intent.startswith('Hotel') or intent.startswith('Restaurant') 
                                                                            for intent in acts['dialog_act']['act_type']]) 
                                #Filtering out system utterances
                                and dial['turns']['speaker'][turn_id] == 0
                                ]if bool(slot_values)]
    
    #Creating DataFrames from the base data
    slot_data_span = pd.DataFrame(base_slot_data, columns=['dialogue_id', 'turn_id', 'utterance', 'intent' ,'values'])
    slot_data_questions = pd.DataFrame(base_slot_data_questions, columns=['dialogue_id', 'turn_id', 'utterance', 'intent', 'values'])

    #Joinning the base data DataFrames
    slot_data = slot_data_span.set_index(['dialogue_id','turn_id']).join(slot_data_questions.set_index(['dialogue_id','turn_id']), lsuffix="span", rsuffix="questions")

    #Copying the joinned data to combine it all together into the final structure
    slot_data_final = slot_data[['utterancespan']].copy()
    slot_data_final = slot_data_final.rename(columns={'utterancespan':'utterance'})

    #Combines the values and intent columns from the two DataFrames to unify them
    slot_data_final['values'] = slot_data[['valuesspan', 'valuesquestions']].apply(lambda v: dict(list(v[0].items()) + list(({} if type(v[1])!= dict else v[1]).items())), axis=1)
    slot_data_final['intent'] = slot_data[['intentspan', 'intentquestions']].apply(lambda v: v[0].union((set() if type(v[1])!= set else v[1])), axis=1)
    slot_data_final['ground_truths'] = slot_data_final['values'].apply(extract_gt)
    slot_data_final.to_csv('./slot_data_final_train.csv')

    return slot_data_final



def get_slots_from_dataset(dataset):
    """
    Returns the set of possible slots which are related to the Hotel and Restaurant domains from a split of the multiwoz dataset.

    Args:
        dataset: A split of the huggingface dataset containing multiwoz.

    Return:
        The set of possible slots which are related to the Hotel and Restaurant domains from a split of the multiwoz dataset.
    """

    slot_values = set([f"{act['span_info']['act_type'][span_ind].lower()}-{span_name}" 
                    for dial in dataset 
                        for turn_id, act in enumerate(dial['turns']['dialogue_acts']) if any([intent.startswith('general') or intent.startswith('Hotel') or intent.startswith('Restaurant') 
                                                                        for intent in act['dialog_act']['act_type']]) and dial['turns']['speaker'][turn_id] == 0
                        for span_ind, span_name in enumerate(act['span_info']['act_slot_name']) 
                        ])

    return slot_values

def get_slot_values_from_dataset(dataset:pd.DataFrame):
    """
    Returns the set of possible slot-value pairs which are related to the Hotel and Restaurant domains from a split of the multiwoz dataset.

    Args:
        dataset: A split of the huggingface dataset containing multiwoz.

    Returns:
        The set of possible slot-value pairs which are related to the Hotel and Restaurant domains from a split of the multiwoz dataset.
    """

    slot_name_values = set([f"{act['span_info']['act_type'][span_ind].split('-')[0].lower()}-{span_name}: {act['span_info']['act_slot_value'][span_ind]}" 
                        for dial in dataset 
                            for turn_id, act in enumerate(dial['turns']['dialogue_acts']) if any([intent.startswith('general') or intent.startswith('Hotel') or intent.startswith('Restaurant') 
                                                                            for intent in act['dialog_act']['act_type']]) and dial['turns']['speaker'][turn_id] == 0
                            for span_ind, span_name in enumerate(act['span_info']['act_slot_name'])])
    return slot_name_values

def get_slot_values_from_categorical_slot(dataset:pd.DataFrame, name_categorical):
    slot_name_values = get_slot_values_from_dataset(dataset)
    return [v for v in slot_name_values if name_categorical in v]

def get_slot_values_from_non_categorical_slot(dataset, slot_name):
    slot_name_values = get_slot_values_from_dataset(dataset)
    return [v for v in slot_name_values if slot_name in v]

def get_ngrams(tokens, max_size):
    """
    Function to construct all possible ngrams with size atmost "maxsize" from a list of tokens.

    The function constructs all possible ngrams with size at most "maxsize" of contiguous tokens in the "tokens" list.

    Args:
        tokens: List of tokens from which ngrams want to be constructed.
        max_size: Maximum size of the ngrams to be constructed.

    Returns:
        Returns a list of strings containing the ngrams. 
    """

    ngrams = []
    tokens = [t.text for t in tokens]

    # Looping through the substrings
    for i in range(len(tokens)):
        j = min(len(tokens), i+max_size)
        while(i < j):
            tok = (" ").join(tokens[i:j])
            j = j-1
            ngrams.append(tok)

    return ngrams

def get_fix_size_ngrams(tokens, size):
    """
    Function to construct all possible ngrams with size "size".

    The function constructs all possible ngrams with size at most "size" of contiguous tokens in the "tokens" list.

    Args:
        tokens: List of tokens from which ngrams want to be constructed.
        size: Size of the ngrams to be constructed.

    Returns:
        Returns a list of strings containing the ngrams.
    """
    

    ngrams = []
    tokens = [t.text for t in tokens]
    # Looping through the substrings of size "size"
    j = 0
    i = 0
    while j < len(tokens):
        j = min(len(tokens), i+size)
        tok = (" ").join(tokens[i:j])
        ngrams.append(tok)
        i+=1

    return ngrams

def get_single_representatives(base_dict, selections, excluded_domain='', excluded_slot=''):
    """
    Auxiliary function for the similarity_sequence_labeling function
    """
    dict_hotels = {
            key: base_dict['hotel'][key][selections['hotel'][key]]  for key in base_dict['hotel'] if (key!=excluded_slot or excluded_domain!='hotel') 
    }
    dict_restaurants = {
            key: base_dict['restaurant'][key][selections['restaurant'][key]]  for key in base_dict['restaurant'] if (key!=excluded_slot or excluded_domain!='restaurant')  
    }
    return{
        'hotel': dict_hotels,
        'restaurant': dict_restaurants
    }


def similarity_sequence_labeling(utt, das, tokenizer, embedder, representatives, sensitivity, ngram_size):

    """
    Extracts the tokens of a utterance using the similarity approach.

    This is the main function we implemented for our similarity approach. It uses a list of representatives to search for relevant ngrams
    of the utterance for each slot, and then it extracts this ngrams from the utterance and passes them through the similarity_classifier to
    obtain its categorical values, if it applies. To find the relevant ngrams the embedding distance between the representatives passed as
    input for each slot, and the ngrams of the utterance of a given size, their embedding distance is calculated. An ngram is then considered
    relevant if it passes a certain threshold contained in the sensitivity parameter.

    Args:
        utt: String representing the utterance to be labeled.
        das: List of strings representing dialogue acts that appear in the utterance. This is used so which domains the function can expect.
        tokenizer: Spacey tokenizer used to tokenize the utterance when searching for relevant ngrams.
        representatives: List of candidates to search for in each slot. They help i
    """
   
    extracted_info = []

    extract_domain = lambda dact: dact.split('-')[0].lower()

    filter_das = [da for da in das if extract_domain(da) in representatives or extract_domain(da) in representatives]

    for da in filter_das:

        ngrams = get_fix_size_ngrams(tokenizer(utt), ngram_size)
        domain = da.split('-')[0].lower()
        da_representatives = representatives[domain]
        distances = {}
        for ngram in ngrams:
            for slot, repr in da_representatives.items():
                dist = embedder.distance(ngram, repr)
                if(dist <= sensitivity):
                    if not slot in distances:
                        distances[slot] = [(dist, ngram)]
                    else:
                        distances[slot].append((dist, ngram))
        
        extracted_info.extend([('-'.join([domain, slot]), min(info_list, key=lambda l:l[0])[1]) for (slot, info_list) in distances.items()])
    final_info = [(extraction[0], similarity_classifier(extraction[0], extraction[1], embedder)) for extraction in extracted_info]
    return set(final_info)


def df_similarity_sequence_labeling(row, tokenizer, embedder, representatives, sensitivity, ngram_size):
    """
    Wrapper used to pass the similarity_sequence_labeling function to a Dataframe.
    """
    utt = row['utterance']
    da = row['intent']
   
    return similarity_sequence_labeling(utt, da, tokenizer, embedder, representatives, sensitivity, ngram_size)


def load_ner_model(path):

    """
    Auxiliary function used to load the NER BERT model into memory using a huggingface pipeline.

    Args:
        path: path to the model to be loaded.
    
    Returns:
        Returns a huggingface pipeline which automatically tokenizes its input before passing it through the model.
    """

    nlp = pipeline(
        "ner",
        model=path,
        tokenizer=path,
    )

    return nlp



# Loads the NER BERT model into memory
bert_pipeline = load_ner_model(NER_MODEL_PATH) 



def check_no_space(tok):
    return tok == "'" or tok == ',' or tok == "-" or tok==":"

def extract_predictions(row):
    """
    Auxiliary function to extract the slot-value pairs from the NER BERT model's output.
    """
    curr_value = ""
    curr_slot = ""

    val_list = []

    row_len = len(row[0])

    for ind, (tok, lab) in enumerate(zip(row[0], row[1])):

        last_elem = (ind == (row_len-1))

        if lab[0] == 'B':
            if(curr_slot != ""):
                val_list.append((curr_slot,curr_value))
            curr_slot = lab[2:]
            curr_value = tok
        elif lab[0] == 'I':
            if('##' in tok):
                curr_value = curr_value+(tok.replace('##',''))
            else:
                if(check_no_space(tok) or (ind>0 and check_no_space(row[0][ind-1]))):
                    curr_value += (tok)
                else:
                    curr_value += (" "+tok)
        elif lab[0] == 'O':
            if(curr_slot != ""):
                val_list.append((curr_slot,curr_value))
            curr_slot = ""
        if(last_elem):
                val_list.append((curr_slot,curr_value))
    return val_list

def predict_ner(utt, categorize=True):
    """
    Interface to the NER BERT model for slot extraction.

    This function implements a simple interface to extract from a given string/utterance the slot-value pairs contained within it.

    Args:
        utt: String representing the input utterance from which the slots'll be extracted.

    Returns:
        The set of slot-value pairs extracted from the utterance.
    """

    entities = bert_pipeline(utt)
    slots = [ent['entity'] for ent in entities]
    values = [ent['word'] for ent in entities]

    extracted = extract_predictions([values, slots])

    if(categorize):
        categorized = [(ext[0],similarity_classifier(ext[0], ext[1], fast_text)) for ext in extracted]
        return categorized
    
    return extracted


def predict_similarity(utt, das):
    """
    Interface for the similarity approach model used for slot extraction.

    This function works as a wrapper for the similarity_sequence_labeling function and passes to it the best parameters 
    contained in the similarity_model for each slot.
    
    Args:
        utt: String representing the input utterance from which the slots'll be extracted.

    Returns:
        The set of slot-value pairs extracted from the utterance.

    """

    pred_slots = set()
    doms = [extract_domain_from_slot(da) for da in das]
    doms = [dom for dom in doms if dom == "restaurant" or dom == "hotel"]
    for dom in doms:
        for slot in similarity_model[dom]:
            repr = similarity_model[dom][slot]['representative']
            sens = similarity_model[dom][slot]['sensitivity']
            ngr_size = similarity_model[dom][slot]['ngram']
            curr_slots = similarity_sequence_labeling(utt, das, tokenizer,fast_text, {dom:{slot:repr}}, sens, ngr_size)
            pred_slots = pred_slots.union(curr_slots)
    return pred_slots

def predict_similarity_df(row):
    utt = row['utterance']
    das = row['intent']

    return predict_similarity(utt, das)


##### Evaluation metrics implemented by the professor ######

def count_matches(ground_truth_list, predicted_list):
    no_gt = len(ground_truth_list)
    no_predicted = len(predicted_list)
    no_correct = no_gt - sum((Counter(ground_truth_list) - Counter(predicted_list)).values())
    return no_gt, no_predicted, no_correct

def get_metrics(no_gt_global, no_predicted_global, no_correct_global):
    precision =  1.0*no_correct_global/no_predicted_global if no_predicted_global > 0 else 1.0
    recall = 1.0*no_correct_global/no_gt_global if no_gt_global > 0 else 1.0
    f1_score = 2.0*precision*recall/(precision+recall) if precision+recall > 0 else 0

    #print(f"Number of predicted slots: {no_predicted_global}")
    #print(f"Number of ground truth global: {no_gt_global}")

    return precision, recall, f1_score

def get_evaluation_metrics(dataset, ground_truth_column='ground_truths', predicition_column='predictions', filter_slots=None):

    gt = [elem for col in dataset[ground_truth_column] for elem in col if elem[1] != '?']
    if filter_slots != None:
        gt = [elem for elem in gt if elem[0] in filter_slots]

    extracted_info = [elem for col in dataset[predicition_column] for elem in col]
    no_gt, no_predicted, no_correct = count_matches(gt, extracted_info)

    return get_metrics(no_gt, no_predicted, no_correct)

#####################