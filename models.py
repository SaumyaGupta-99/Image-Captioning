from keras import Sequential, Input, Model
from keras.src.layers import Dense, RepeatVector, Embedding, Bidirectional, Dropout, BatchNormalization, \
    TimeDistributed, LSTM, Concatenate, Activation
from keras import layers
from keras.regularizers import l2
import Preprocessing

EMBEDDING_DIM = 300
max_length = Preprocessing.max_length_caption
vocab_size = Preprocessing.total_words

image_input = Input(shape=(2048,))
image_embedding = Dense(EMBEDDING_DIM, activation='relu')(image_input)
image_repeat = RepeatVector(Preprocessing.max_length_caption)(image_embedding)

image_model = Model(inputs=image_input, outputs=image_repeat)

lang_model = Sequential()
lang_model.add(Embedding(Preprocessing.total_words, EMBEDDING_DIM, input_length=Preprocessing.max_length_caption))
lang_model.add(Bidirectional(LSTM(256, return_sequences=True)))
lang_model.add(Dropout(0.5))
lang_model.add(BatchNormalization())
# lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))

lang_input = Input(shape=(Preprocessing.max_length_caption,))
lang_embedding = (Embedding(Preprocessing.total_words, EMBEDDING_DIM, input_length=Preprocessing.max_length_caption)
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
output_layer = Dense(Preprocessing.total_words, activation='softmax')(lstm_bidirectional)

fin_model = Model(inputs=[image_input, lang_input], outputs=output_layer)

# Compile the final model
fin_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
print("Image Model!")
print(image_model.summary())
print("Language Model!")
print(lang_model.summary())
print("Final Model!")
print(fin_model.summary())

# v_size = vocab_size
# input_image = layers.Input(shape=(2048,))
# img_d = layers.Dense(128, activation='relu')(input_image)
# inp = layers.Input(shape=(30,))
# lang = layers.Embedding(v_size, 64)(inp)
# lang = layers.LSTM(128, return_sequences=True, recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))(lang)
# new_m = Dropout(0.5)(lang)
# lang = layers.Bidirectional(LSTM(64, recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))(new_m)
# conca = layers.add([lang, img_d])
# conca = layers.Dense(256, activation='relu')(conca)
# output = layers.Dense(v_size, activation='softmax')(conca)
# fin_model = Model(inputs=[input_image, inp], outputs=output)
# fin_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(fin_model.summary())
