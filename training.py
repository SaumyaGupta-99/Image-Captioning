import Preprocessing
import models

epoch = 20
batch_size = 128
no_samples = len(Preprocessing.training_dataset)
models.fin_model.fit_generator(Preprocessing.process_captions(Preprocessing.training_dataset, batch_size=batch_size),
                               steps_per_epoch=no_samples / batch_size, epochs=epoch, verbose=1, callbacks=None)
models.fin_model.save(Preprocessing.current_working_directory + "Weights_Bidirectional_LSTM.h5")
# After model is trained , we simply just load the weights
# models.fin_model.load_weights(Preprocessing.current_working_directory+"Weights_Bidirectional_LSTM.h5")
