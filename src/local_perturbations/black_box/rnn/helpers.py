import os
from keras.models import model_from_json
import joblib as jlib


def get_word_model(model_path):
    word_model_path = os.path.join(model_path, "word.model")
    if os.path.exists(word_model_path):
        word_model = jlib.load(word_model_path)
        return word_model
    else:
        raise ValueError("Path to word model %s does not exist" % (word_model_path))


def get_model(model_path):
    model_structure_path = os.path.join(model_path, "model_structure.json")
    if os.path.exists(model_structure_path):
        model = model_from_json(open(model_structure_path).read())
        model.load_weights(os.path.join(model_path, "model_weights.h5"))
        return model
    else:
        raise ValueError(
            "Path to model structure %s does not exist" % (model_structure_path)
        )
