import torch
import spacy
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import re

class Info_tb_retrieved_util():
    """For this you need to have installed in your environment the following:
        1. ! python -m textblob.download_corpora
        2. ! pip install nltk
        3. ! pip install textblob
        4. ! pip install spacy
        5. PYTORCH
    """
    def __init__(self,labels_itr_path = 'models/info_to_be_retrieved/labels.txt',model_itr_path = 'models/info_to_be_retrieved/best_model.pt'):
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        nltk.download('wordnet', quiet=False)
        nltk.download('averaged_perceptron_tagger', quiet=False)

        self.stemmer = stemmer = PorterStemmer()
        self.lemmatizer = lemmatizer = WordNetLemmatizer()
        self.nlp_spacy_tokenizer = spacy.load("en_core_web_md")
        with open(labels_itr_path, 'r') as file:
            self.labels = [line.strip() for line in file]
        self.model = SpacyEmbeddingClassifier(300,66)
        self.model, epoch, valid_loss_min = self.load_ckp(model_itr_path, self.model)
    def from_1_hot_to_labels(self,one_hot):
        output_idx = np.where(one_hot == 1)[0].tolist()
        return [self.labels[i] for i in output_idx]
    def get_embbedings(self,text):
        # Average the token vectors to get a sentence vector
        doc = self.nlp_spacy_tokenizer(text)
        vector = np.mean([token.vector for token in doc], axis=0)

        if len(vector) < 300:
            padding = np.zeros((300 - len(vector), vector.shape[0]))
            vector = np.concatenate((vector, padding), axis=0)
        return torch.tensor(vector, dtype=torch.float)
    def load_ckp(self,checkpoint_fpath, model):
        """
        checkpoint_path: path to save checkpoint
        model: model that we want to load checkpoint parameters into       
        optimizer: optimizer we defined in previous training
        """
        # load check point
        checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))
        # initialize state_dict from checkpoint to model
        model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        #optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        # return model, optimizer, epoch value, min validation loss 
        return model, checkpoint['epoch'], valid_loss_min
    def remove_stop_words(self,sentence):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
        filtered_sentence = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_sentence)
    def correct_spelling(self,text):
        corrected_text = []
        for word in text.split():
            # Correct the spelling and convert the TextBlob object back to a string
            corrected_text.append(str(TextBlob(word).correct()))
        return " ".join(corrected_text)
    def remove_punctuation(self,text):
        # Removing punctuation and special characters
        return re.sub(r'[^\w\s]', '', text)
    def stem_and_lemmatize(self,text, stemmer, lemmatizer):
        # Tokenize the text
        tokens = word_tokenize(text)
        
        stemmed_and_lemmatized_tokens = []
        for word in tokens:
            # Apply stemming
            stemmed_word = stemmer.stem(word)
            # Apply lemmatization
            lemmatized_word = lemmatizer.lemmatize(stemmed_word)
            stemmed_and_lemmatized_tokens.append(lemmatized_word)

        # Reconstruct the sentence
        return ' '.join(stemmed_and_lemmatized_tokens)
    def truncate_from_start(self,text, sep = '|'):
        max_length = 300

        # If the text is already within the limit, return as is
        if len(text) <= max_length:
            return text

        # Split the text into chunks based on [SEP]
        chunks = text.split(sep)

        # Remove chunks from the beginning until the length is within the limit
        while len(chunks) > 1 and len(sep.join(chunks)) > max_length:
            chunks.pop(1)  # Remove the first chunk after [CLS] (index 1)

        # Reassemble the text
        truncated_text = sep.join(chunks)
        return truncated_text
    def pre_process(self,sentence):
        current_sentence = self.remove_stop_words(sentence)
        current_sentence_1 = self.correct_spelling(current_sentence)
        current_sentence_2 = self.remove_punctuation(current_sentence_1)
        current_sentence_3 = self.stem_and_lemmatize(current_sentence_2, self.stemmer, self.lemmatizer)
        return current_sentence_3
    def get_prediction(self,sentence_embbedings):
        self.model.eval()
        with torch.no_grad():
            output = self.model(sentence_embbedings)
            output = output.cpu().detach().numpy()
            output = (output > 0.5).astype(int)
        return output



class SpacyEmbeddingClassifier(torch.nn.Module):
    def __init__(self, embedding_dim, output_len):
        super(SpacyEmbeddingClassifier, self).__init__()
        self.fc = torch.nn.Linear(embedding_dim, output_len)
 
    def forward(self, features):
        x = self.fc(features)
        return torch.sigmoid(x)  # Use sigmoid for multilabel classification
    
'''def Information_to_be_retrieved_Prediction(user_dialogue_acts, extracted_information, other_features_from_dialogue_history3):
    inf_tb_retrieved_util = Info_tb_retrieved_util()

    sentence_preprocessed = inf_tb_retrieved_util.pre_process(other_features_from_dialogue_history3['user_utt'])

    text = other_features_from_dialogue_history3['past_info'] + "|" + sentence_preprocessed
    text_2 = inf_tb_retrieved_util.truncate_from_start(text)

    text_embbedings = inf_tb_retrieved_util.get_embbedings(text_2)
    output = inf_tb_retrieved_util.get_prediction(text_embbedings)
    to_be_retrieved = inf_tb_retrieved_util.from_1_hot_to_labels(output)
    return to_be_retrieved, text_2

past_info = ""
user_utt = "Hi, I'm looking for a hotel to stay in that includes free wifi. I'm looking to stay in a hotel, not a guesthouse."
Information_to_be_retrieved_Prediction(None, None, other_features_from_dialogue_history3={'user_utt':user_utt,'past_info':past_info})'''