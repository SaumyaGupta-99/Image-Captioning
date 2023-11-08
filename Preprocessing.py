import os
import string
import warnings

from keras.src.applications import InceptionV3
from keras.src.preprocessing.text import Tokenizer
from pycocotools.coco import COCO
from itertools import chain
import numpy as np
import skimage.io as io
import cv2
from tqdm import tqdm

warnings.filterwarnings("ignore")

image_path = "C:\\Users\\ASUS\\Downloads\\coco\\coco2017\\train2017"
annotations_file = "C:\\Users\\ASUS\\Downloads\\coco\\coco2017\\annotations\\captions_train2017.json"
jpgs = os.listdir(image_path)

print("Total Images in Dataset = {}".format(len(jpgs)))

coco = COCO(annotations_file)
imgIds = coco.getImgIds()
dataset = dict()
imgcaptions = []

for imgid in imgIds:
    img = coco.loadImgs(imgid)[0]
    annIds = coco.getAnnIds(imgIds=img['id'])
    anns = coco.loadAnns(annIds)
    imgcaptions = []
    for cap in anns:
        # Remove punctuation
        cap = cap['caption'].translate(str.maketrans('', '', string.punctuation))

        # Replace - to blank
        cap = cap.replace("-", " ")

        # Split string into word list and Convert each word into lower case
        cap = cap.split()
        cap = [word.lower() for word in cap]

        # join word list into sentence and <start> and <end> tag to each sentence which helps
        # LSTM encoder-decoder model while training.

        cap = '<start> ' + " ".join(cap) + ' <end>'
        imgcaptions.append(cap)

    dataset[img['coco_url']] = imgcaptions

print("Length of Dataset: ", len(dataset))

flatten_list = list(chain.from_iterable(dataset.values()))  # [[1,3],[4,8]] = [1,3,4,8]

tokenizer = Tokenizer(oov_token='<unk>')  # For those words which are not found in word_index
tokenizer.fit_on_texts(flatten_list)
total_words = len(tokenizer.word_index) + 1

print("Vocabulary length: ", total_words)
print("Bicycle ID: ", tokenizer.word_index['bicycle'])
print("Airplane ID: ", tokenizer.word_index['airplane'])

model = InceptionV3(include_top=False, pooling='avg',weights='imagenet')

image_features = {}

for img in tqdm(dataset.keys()):
    image = io.imread(img)
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

print("Image features length: ", len(image_features))

