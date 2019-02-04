import numpy as np

from local_perturbations.black_box.rnn.helpers import get_word_model, get_model


class RNN_Scorer(object):
    """
    Streaming scorer based on a trained RNN model from keras. 

    Currently, streaming scoring is done in a clumsy, non-scalable way, by maintaining a record of words seen so far, 
        and then refeeding that into the "predict" function.
    This is because the current RNN model happened to have been trained in a "stateless" way (the Keras default), 
        and switching over to stateful for prediction mode (without changing the training) may cause problems; see 
            https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
    We accept this for now for prototyping purposes. 

    Main Attributes:
        word_model: Instance of gensim.models.word2vec.Word2Vec
        model: Instance of keras.models.Sequential 
    """

    def __init__(self, model_path):
        self.word_model = get_word_model(model_path)
        self.pretrained_weights = self.word_model.wv.syn0
        self.vocab_size, self.embedding_size = self.pretrained_weights.shape
        self.model = get_model(model_path)
        self.processed_word_idxs = []  # store the word indices processed thus far

    def _word2idx(self, word):
        if word in self.word_model.wv.vocab:
            return self.word_model.wv.vocab[word].index
        else:
            return self.vocab_size - 1

    def _idx2word(self, idx):
        if idx < self.vocab_size - 1:
            return self.word_model.wv.index2word[idx]
        else:
            return "OOV"

    def predict_probabilities(self, text):
        """
        output the result from a softmax layer, which can be used to represent the probabilities of a categorical distribution

        :param text:
        :return:
        """
        word_idxs = [self._word2idx(word) for word in text.lower().split()]
        if len(self.processed_word_idxs) != 0:
            word_idxs = self.processed_word_idxs + word_idxs
        self.processed_word_idxs = word_idxs
        return self.model.predict(x=np.array(word_idxs))[-1][-1]
