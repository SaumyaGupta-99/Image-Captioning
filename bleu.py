from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import Preprocessing
import test

reference = Preprocessing.test_dataset
for img in tqdm(reference.keys()):
    actual, predicted, beam_predicted = list(), list(), list()
    for key, desc_list in reference.items():
        caption = test.predict_captions(key)
        beam_caption = test.beam_search_predictions(key, beam_index=5)
        references = [d.split()[1:-1] for d in desc_list]
        actual.append(references)
        predicted.append(caption.split())
        beam_predicted.append(beam_caption.split())

    print("Greedy Search Predicted Captions BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print("Greedy Search Predicted Captions BLEU-3: %f" % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print(
        "Beam Search K=5 Predicted Captions BLEU-2: %f" % corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0)))
    print("Beam Search K=5 Predicted Captions BLEU-3: %f" % corpus_bleu(actual, beam_predicted,
                                                                        weights=(0.3, 0.3, 0.3, 0)))
