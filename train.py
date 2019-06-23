import json
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint

# model config
BATCH_SIZE = 64
EPOCHS = 30
VAL_SPLIT = 0.20


def train_model(model, X_train, y_train, epochs=200):

    model.compile(optimizer='adam', loss='mse')

    # add call backs
    early_stopper = EarlyStopping(monitor='val_rmse', patience=10, verbose=1)

    model_json = model.to_json()
    with open("checkpoint/%s.json" % model.name, "w") as json_file:
        json.dump(model_json, json_file)
    model_saver = ModelCheckpoint(filepath=("checkpoint/%s.h5" % model.name),
                                  monitor='val_rmse',
                                  save_best_only=True,
                                  save_weights_only=True)
    # train model
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=epochs,
                        validation_split=VAL_SPLIT,
                        callbacks=[early_stopper, model_saver])

    model_json = model.to_json()
    with open("checkpoint/%s.json" % model.name, "w") as json_file:
        json.dump(model_json, json_file)
    model.save_weights("checkpoint/%s.h5" % model.name)

    return model, history