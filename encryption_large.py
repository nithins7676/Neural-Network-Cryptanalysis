import warnings
import config
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import model_from_json
from utils import (accuracy, change_output, create_input_array, create_labels,
                   generate_hashmap, random_paragraph_generator)

warnings.filterwarnings("ignore")

def build_model(layer_sizes, input_shape):
    """Helper function to build a sequential model with specified layer sizes."""
    model = Sequential()
    model.add(layers.Dense(layer_sizes[0], input_shape=(input_shape,)))
    model.add(layers.LeakyReLU())
    
    for size in layer_sizes[1:]:
        model.add(layers.Dense(size))
        model.add(layers.LeakyReLU())
    
    return model

def setup_callbacks(checkpoint_path):
    """Setup callbacks for model checkpointing and learning rate reduction."""
    checkpoint = ModelCheckpoint(
        f"{checkpoint_path}.weights.h5",  # Ensure the correct extension is added
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    
    reducelr = ReduceLROnPlateau(
        monitor="val_loss", verbose=1, patience=5, factor=0.05, min_lr=0.003
    )
    
    return [checkpoint, reducelr]

def main():
    hashmap = generate_hashmap(2 ** 55, 2 ** 56)
    
    train_string = random_paragraph_generator()
    print("Training string generated.")

    test_string = """ abcdefghigjklmnopqrstuvx ysx d go abcdef hidsog """

    # Check for invalid characters in both training and test strings
    for string, label in [(train_string, "train_string"), (test_string, "test_string")]:
        for i in string:
            if ord(i) < 32 or ord(i) > 122:
                print(f"Invalid char in {label}: {i}, ord: {ord(i)}, index: {string.find(i)}")

    # Create input arrays
    X_train = create_input_array(train_string)
    print("X_train shape:", X_train.shape)

    X_test = create_input_array(test_string)
    print("X_test shape:", X_test.shape)

    # Create labels for training and test sets
    Y_train = create_labels(train_string, hashmap)
    print("Y_train shape:", Y_train.shape)

    Y_test = create_labels(test_string, hashmap)
    print("Y_test shape:", Y_test.shape)

    # Define encrypter model
    encrypter_layer_sizes = [91, 82, 74, 68, 64, 56, 56]
    encrypter = build_model(encrypter_layer_sizes, input_shape=91)

    learning_rate = config.ENC_LEARNING_RATE
    epochs = config.ENC_EPOCHS
    batch_size = config.ENC_BATCH_SIZE

    optim = optimizers.Adam(learning_rate=learning_rate)

    # Setup callbacks for encrypter
    callbacks = setup_callbacks(config.ENC_LARGE_CHK)
    
    encrypter.compile(optimizer=optim, loss="mean_squared_error", metrics=["acc"])
    encrypter.summary()

    # Train encrypter model
    history = encrypter.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
        validation_split=config.ENC_VALIDATION_SPLIT,
    )

    # Predictions and accuracy
    output = encrypter.predict(X_train)
    print("Sample output:", output[0])
    print("Expected output:", Y_train[0])

    Y_pred = encrypter.predict(X_train)
    accuracy(Y_pred, Y_train)

    Y_test_pred = encrypter.predict(X_test)
    accuracy(Y_test_pred, Y_test)

    # Serialize model to JSON
    model_json = encrypter.to_json()
    with open(config.ENC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)

    # Serialize weights to HDF5
    encrypter.save_weights(config.ENC_LARGE_MODEL)
    print("Saved encrypter model to disk.")

    # Load JSON and create model
    with open(config.ENC_LARGE_JSON, "r") as json_file:
        loaded_model_json = json_file.read()
    
    loaded_model = model_from_json(loaded_model_json)

    # Load weights into new model
    loaded_model.load_weights(config.ENC_LARGE_MODEL)
    print("Loaded encrypter model from disk.")

    print("Encrypter can be loaded and run.")

    # Define decrypter model
    decrypter_layer_sizes = [56, 64, 72, 80, 85, 88, 91, 91]
    decrypter = build_model(decrypter_layer_sizes, input_shape=56)

    learning_rate = config.DEC_LEARNING_RATE
    decrypter_optimizer = optimizers.Adam(learning_rate=learning_rate)

    decrypter.compile(optimizer=decrypter_optimizer, loss="mean_squared_error", metrics=["acc"])

    # Prepare data for decrypter training
    decrypter_X_train = Y_train
    decrypter_Y_train = X_train

    decrypter_X_test = Y_test_pred
    decrypter_Y_test = X_test

    print("Decrypter X_train shape:", decrypter_X_train.shape)
    print("Decrypter Y_train shape:", decrypter_Y_train.shape)
    print("Decrypter X_test shape:", decrypter_X_test.shape)
    print("Decrypter Y_test shape:", decrypter_Y_test.shape)

    # Setup callbacks for decrypter
    callbacks_decrypter = setup_callbacks(config.DEC_LARGE_CHK)

    # Train decrypter model
    decrypter_history = decrypter.fit(
        decrypter_X_train,
        decrypter_Y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks_decrypter,
        validation_split=config.DEC_VALIDATION_SPLIT,
    )

    decrypted_text = decrypter.predict(decrypter_X_train)
    decrypted_int_text = change_output(decrypted_text)

    print("Decrypted text sample:", decrypted_text[0])
    print("Decrypted int text sample:", decrypted_int_text[0])
    print("Expected Y_train sample:", decrypter_Y_train[0])

    accuracy(decrypted_int_text, decrypter_Y_train)

    decrypted_Y_test_pred = decrypter.predict(decrypter_X_test)
    decrypted_Y_test_pred = change_output(decrypted_Y_test_pred)
    accuracy(decrypted_Y_test_pred, decrypter_Y_test)

    # Serialize decrypter model to JSON
    model_json = decrypter.to_json()
    with open(config.DEC_LARGE_JSON, "w") as json_file:
        json_file.write(model_json)

    # Serialize decrypter weights to HDF5
    decrypter.save_weights(config.DEC_LARGE_MODEL)
    print("Saved decrypter model to disk.")

    # Load JSON and create decrypter
    with open(config.DEC_LARGE_JSON, "r") as json_file:
        loaded_model_json = json_file.read()
    
    loaded_decrypter = model_from_json(loaded_model_json)

    # Load weights into new decrypter
    loaded_decrypter.load_weights(config.DEC_LARGE_MODEL)
    print("Loaded decrypter model from disk.")

    print("Decrypter can be loaded and run.")

if __name__ == "__main__":
    main()
