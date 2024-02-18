import re

import nltk
import numpy as np
import torch
import transformers
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from transformers import BertTokenizer, BertModel


class Info_tb_requested_util():
    """For this you need to have installed in your environment the following:
        1. ! python -m textblob.download_corpora
        2. ! pip install nltk
        3. ! pip install textblob
        4. ! pip install transformers
        5. PYTORCH
    """

    def __init__(self, labels_itr_path='models/info_to_be_requested/labels.txt',
                 model_itr_path='models/info_to_be_requested/best_model.pt'):
        nltk.download('punkt', quiet=False)
        nltk.download('stopwords', quiet=False)
        nltk.download('wordnet', quiet=False)
        nltk.download('averaged_perceptron_tagger', quiet=False)

        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.stemmer = stemmer = PorterStemmer()
        self.lemmatizer = lemmatizer = WordNetLemmatizer()

        # Load labels and model
        with open(labels_itr_path, 'r') as file:
            self.labels = [line.strip() for line in file]
        self.model = BERTClass(len(self.labels))
        self.model, epoch, valid_loss_min = self.load_ckp(model_itr_path, self.model)

    def from_1_hot_to_labels(self, one_hot):
        # Find indices where prediction is 1
        output_idx = np.where(one_hot == 1)[1].tolist()
        # print("Indices with prediction 1:", output_idx)
        labels_predicted = [self.labels[i] for i in output_idx]
        # print("Labels predicted:", labels_predicted)
        return labels_predicted

    def get_bert_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = self.bert_model(input_ids, attention_mask=attention_mask)

        embeddings = outputs.last_hidden_state.mean(1)
        return embeddings

    def load_ckp(self, checkpoint_fpath, model):
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
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']
        # return model, optimizer, epoch value, min validation loss
        return model, checkpoint['epoch'], valid_loss_min

    def remove_stop_words(self, sentence):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
        filtered_sentence = [word for word in word_tokens if word not in stop_words]
        return ' '.join(filtered_sentence)

    def correct_spelling(self, text):
        corrected_text = []
        for word in text.split():
            # Correct the spelling and convert the TextBlob object back to a string
            corrected_text.append(str(TextBlob(word).correct()))
        return " ".join(corrected_text)

    def remove_punctuation(self, text):
        # Removing punctuation and special characters
        return re.sub(r'[^\w\s]', '', text)

    def stem_and_lemmatize(self, text, stemmer, lemmatizer):
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

    def truncate_from_start(self, text, sep='|'):
        max_length = 300
        # max_length = 512
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

    def pre_process(self, sentence):
        current_sentence = self.remove_stop_words(sentence)
        current_sentence_1 = self.correct_spelling(current_sentence)
        current_sentence_2 = self.remove_punctuation(current_sentence_1)
        current_sentence_3 = self.stem_and_lemmatize(current_sentence_2, self.stemmer, self.lemmatizer)
        return current_sentence_3

    def get_prediction(self, text, return_raw_scores=False):
        # Tokenize the text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        # Predict using the model
        self.model.eval()
        with torch.no_grad():
            # Pass token_type_ids to the model as well
            output = self.model(input_ids, attention_mask, token_type_ids)
            output = output.cpu().detach().numpy()

            # If return_raw_scores is True, return the raw model output
            if return_raw_scores:
                return output

            # Else, apply threshold and return binary predictions
            output_binary = (output > 0.5).astype(int)
            return output_binary


class BERTClass(torch.nn.Module):
    def __init__(self, output_len):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, output_len)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # Extract the pooled output (or you can use output_1.last_hidden_state)
        pooled_output = output_1.pooler_output
        output_2 = self.l2(pooled_output)
        pre_output = self.l3(output_2)
        # Apply sigmoid to the output
        output = torch.sigmoid(pre_output)
        return output
