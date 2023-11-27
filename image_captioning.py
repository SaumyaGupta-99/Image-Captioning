# -*- coding: utf-8 -*-
"""Image captioning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Pgnmg3pnnWlDjA4Wa7MdX1oZ4ocs_JZr
"""

import os
import cv2
from keras.applications import InceptionV3
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import string
import warnings
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from itertools import chain
import random
from keras import Sequential, Input, Model
from keras.src.layers import Dense, RepeatVector, Embedding, Bidirectional, Dropout, BatchNormalization, \
    TimeDistributed, LSTM, Concatenate
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import numpy as np

model = InceptionV3(include_top=False, pooling='avg', weights='imagenet')

image_features = {}

current_working_directory = os.getcwd()
image_path = current_working_directory + "/flickr/Images/"


def get_image_features(images):
    for img in tqdm(images):
        image = io.imread(image_path + "/" + img)
        if image.ndim != 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize each image size 299 x 299
        image = cv2.resize(image, (299, 299))
        image = np.expand_dims(image, axis=0)

        # Normalize image pixels
        image = image / 127.5
        image = image - 1.0

        # Extract features from image
        feature = model.predict(image)
        image_features[img] = feature
    return image_features


warnings.filterwarnings("ignore")
current_working_directory = os.getcwd()
image_path = current_working_directory + "/flickr/Images"
annotations_file = current_working_directory + "/flickr/captions.txt"

jpgs = os.listdir(image_path)

print("Total Images in Dataset = {}".format(len(jpgs)))

dataset = dict()
max_length_caption = 0
captions = open(annotations_file, 'r', encoding="utf8").read().split("\n")
captions = captions[1:]
for read_line in captions:
    caption_col_data = read_line.split(',')
    if len(caption_col_data) <= 1:
        break
    w = caption_col_data[0].split("#")
    cap = caption_col_data[1].translate(str.maketrans('', '', string.punctuation))
    # Replace - to blank
    cap = cap.replace("-", " ")

    # Split string into word list and Convert each word into lower case
    cap = cap.split()
    cap = [word.lower() for word in cap]

    # join word list into sentence and <start> and <end> tag to each sentence which helps
    # LSTM encoder-decoder model while training.
    cap = '<start> ' + " ".join(cap) + ' <end>'
    if w[0] not in dataset.keys():
        dataset[w[0]] = []
    max_length_caption = max(max_length_caption, len(cap.split()))
    dataset[w[0]].append(cap.lower())

print(max_length_caption)
print("Length of Dataset: ", len(dataset))

flatten_list = list(chain.from_iterable(dataset.values()))  # [[1,3],[4,8]] = [1,3,4,8]
tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', oov_token='<unk>')  # For those
# words which are
# not found in word_index
tokenizer.fit_on_texts(flatten_list)
total_words = len(tokenizer.word_index) + 1

print("Vocabulary length: ", total_words)
print("Bicycle ID: ", tokenizer.word_index['bicycle'])
print("Airplane ID: ", tokenizer.word_index['airplane'])
print('<start>:', tokenizer.word_index['<start>'])
print('<end>:', tokenizer.word_index['<end>'])

# for getting image features using InceptionV3 CNN model

# img_encodings = get_image_features(dataset.keys())
# print("Image features length: ", len(img_encodings))

# with open(current_working_directory + "/drive/MyDrive/flickr/encoded_train_images_inceptionV3-8k.p", "wb") as encoded_pickle:
#     pickle.dump(img_encodings, encoded_pickle)

img_encodings = pickle.load(open(current_working_directory + "/encoded_train_images_inceptionV3-8k.p", 'rb'))

word_idx = {val: index for index, val in enumerate(tokenizer.word_index)}
idx_word = {index: val for index, val in enumerate(tokenizer.word_index)}


def split_dict(dictionary, n):
    # Get the keys from the dictionary
    keys = list(dictionary.keys())

    # Get the first n random values
    a_keys = random.sample(keys, n)
    a = {key: dictionary[key] for key in a_keys}

    # Get the rest (n-1) values
    b_keys = [key for key in keys if key not in a_keys]
    b = {key: dictionary[key] for key in b_keys}

    return a, b


training_dataset, test_dataset = split_dict(dataset, 7091)
captionz = []
img_id = []

for img in training_dataset.keys():
    for caption in training_dataset[img]:
        captionz.append(caption)
        img_id.append(img)

print(len(captionz), len(img_id))
no_samples = 0
for caption in captionz:
    no_samples += len(caption.split()) - 1
print(no_samples)

caption_length = [len(caption.split()) for caption in captionz]
max_length_caption = max(caption_length)


def data_process(batch_size):
    partial_captions = []
    next_words = []
    images = []
    total_count = 0
    while 1:

        for image_counter, caption in enumerate(captionz):
            current_image = img_encodings[img_id[image_counter]][0]
            for i in range(len(caption.split()) - 1):
                total_count += 1
                partial = [word_idx[txt] for txt in caption.split()[:i + 1]]
                partial_captions.append(partial)
                next = np.zeros(total_words)
                next[word_idx[caption.split()[i + 1]]] = 1
                next_words.append(next)
                images.append(current_image)

                if total_count >= batch_size:
                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    partial_captions = pad_sequences(partial_captions, maxlen=max_length_caption, padding='post')
                    total_count = 0
                    yield [[images, partial_captions], next_words]
                    partial_captions = []
                    next_words = []
                    images = []


EMBEDDING_DIM = 300
image_input = Input(shape=(2048,))
image_embedding = Dense(EMBEDDING_DIM, activation='relu')(image_input)
image_repeat = RepeatVector(max_length_caption)(image_embedding)

image_model = Model(inputs=image_input, outputs=image_repeat)

lang_model = Sequential()
lang_model.add(Embedding(total_words, EMBEDDING_DIM, input_length=max_length_caption))
lang_model.add(Bidirectional(LSTM(256, return_sequences=True)))
lang_model.add(Dropout(0.5))
lang_model.add(BatchNormalization())

lang_input = Input(shape=(max_length_caption,))
lang_embedding = (Embedding(total_words, EMBEDDING_DIM, input_length=max_length_caption)
                  (lang_input))
lang_bidirectional = Bidirectional(LSTM(256, return_sequences=True))(lang_embedding)
lang_dropout = Dropout(0.5)(lang_bidirectional)
lang_batchNorm = BatchNormalization()(lang_dropout)
lang_timeDistributed = TimeDistributed(Dense(EMBEDDING_DIM))(lang_batchNorm)

lang_model = Model(inputs=lang_input, outputs=lang_timeDistributed)

concatenated = Concatenate()([image_model.output, lang_model.output])
dropout_concatenated = Dropout(0.5)(concatenated)
batchNorm_concatenated = BatchNormalization()(dropout_concatenated)

# Final model
lstm_bidirectional = Bidirectional(LSTM(1000, return_sequences=False))(batchNorm_concatenated)
output_layer = Dense(total_words, activation='softmax')(lstm_bidirectional)

fin_model = Model(inputs=[image_input, lang_input], outputs=output_layer)

# Compile the final model
fin_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print("Image Model!")
print(image_model.summary())
print("Language Model!")
print(lang_model.summary())
print("Final Model!")
print(fin_model.summary())

epoch = 20
batch_size = 128
checkpoint_path = current_working_directory + '/weights-model3.{epoch:02d}-{loss:.2f}.h5'
# Create a ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='accuracy',
    save_weights_only=True,
    save_best_only=False,
    period=5  # Save weights every 10 iterations
)

# fin_model.fit_generator(data_process(batch_size=batch_size), steps_per_epoch=no_samples/batch_size, epochs=epoch,
# verbose=1,callbacks=[checkpoint_callback]) fin_model.save(
# current_working_directory+"/Weights_Bidirectional_LSTM-model3.h5")

# After model is trained , we simply just load the weights
fin_model.load_weights(current_working_directory + "/weights-model3.20-1.13.h5")


# for getting image features of validation images
def predict_captions(img_file):
    start_word = ["<start>"]
    while 1:
        now_caps = [word_idx[i] for i in start_word]
        now_caps = pad_sequences([now_caps], maxlen=max_length_caption, padding='post')
        e = img_encodings[img_file][0]
        predictions = fin_model.predict([np.array([e]), np.array(now_caps)])
        word_pred = idx_word[np.argmax(predictions[0])]
        start_word.append(word_pred)

        if word_pred == "<end>" or len(start_word) > max_length_caption:
            # keep on predicting next word until word predicted is <end> or caption lengths is greater than
            # max_length(40)
            break

    return ' '.join(start_word[1:-1])


def beam_search_predictions(img_file, beam_index=3):
    start = [word_idx["<start>"]]

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length_caption:
        temp = []
        for s in start_word:
            now_caps = pad_sequences([s[0]], maxlen=max_length_caption, padding='post')
            e = img_encodings[img_file][0]
            predictions = fin_model.predict([np.array([e]), np.array(now_caps)])

            word_predictions = np.argsort(predictions[0])[-beam_index:]

            # Getting the top Beam index = 3  predictions and creating a
            # new list to put them via the model again
            for w in word_predictions:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [idx_word[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


test_dataset = test_dataset

for i in range(5):
    image_file = list(test_dataset.keys())[random.randint(0, 1000)]
    test_image = image_path + str(image_file)
    # Read the image using matplotlib
    image = mpimg.imread(test_image)

    caption1 = predict_captions(image_file)
    caption2 = beam_search_predictions(image_file, beam_index=3)
    caption3 = beam_search_predictions(image_file, beam_index=5)

    actual, predicted, beam_predicted, beam_predicted2 = list(), list(), list(), list()
    references = [d.split()[1:-1] for d in test_dataset[image_file]]
    actual.append(references)
    predicted.append(caption1.split())
    beam_predicted.append(caption2.split())
    beam_predicted2.append(caption3.split())
    bleu = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_beam = corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0))
    bleu_beam2 = corpus_bleu(actual, beam_predicted2, weights=(0.5, 0.5, 0, 0))

    # Display the image
    plt.imshow(image)
    plt.title("Your Image Title")
    plt.axis('off')  # Turn off axis labels
    plt.show()

    print('Greedy search:', caption1, 'Bleu is:', str(bleu))
    print('Beam Search, k=3:', caption2, 'Bleu is:', str(bleu_beam))
    print('Beam Search, k=5:', caption3, 'Bleu is:', str(bleu_beam2))

reference = test_dataset
actual, predicted, beam_predicted = list(), list(), list()
for key, desc_list in tqdm(reference.items()):
    caption = predict_captions(key)
    beam_caption = beam_search_predictions(key, beam_index=5)
    references = [d.split()[1:-1] for d in desc_list]
    actual.append(references)
    predicted.append(caption.split())
    beam_predicted.append(beam_caption.split())

print("Greedy Search Predicted Captions BLEU-2: %f" % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print("Beam Search K=5 Predicted Captions BLEU-2: %f" % corpus_bleu(actual, beam_predicted, weights=(0.5, 0.5, 0, 0)))
