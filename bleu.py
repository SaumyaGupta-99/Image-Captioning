from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import Preprocessing
import test

reference = Preprocessing.test_dataset
bleu =[]
bleu_beam =[]
actual, predicted, beam_predicted = list(), list(), list()
for key, desc_list in reference.items():
    caption = test.predict_captions(key)
    beam_caption = test.beam_search_predictions(key, beam_index=5)
    references = [d.split()[1:-1] for d in desc_list]
    actual.append(references)
    predicted.append(caption.split())
    beam_predicted.append(beam_caption.split())
    bleu.append(corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    bleu_beam.append(corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0)))

print("Greedy Search Predicted Captions BLEU-2: %f" % np.mean(bleu))
print("Beam Search K=5 Predicted Captions BLEU-2: %f" % np.mean(bleu_beam))

