from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import Preprocessing
import test
import numpy as np

reference = Preprocessing.test_dataset.values()
bleu = []
for img in tqdm(Preprocessing.test_dataset.keys()):
    caption = test.predict_captions(img).split()
    bleu1 = sentence_bleu(reference, caption)
    bleu.append(bleu1)

print("Mean For Full Predicted Captions BLEU {:4.3f}".format(np.mean(bleu)))

bleu = []
for img in tqdm(Preprocessing.test_dataset.keys()):
    caption = test.beam_search_predictions(img, beam_index=5).split()
    bleu1 = sentence_bleu(reference, caption)
    bleu.append(bleu1)

print("Mean For Beam Search Captions BLEU {:4.3f}".format(np.mean(bleu)))
