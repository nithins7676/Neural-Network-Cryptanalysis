import warnings
from secrets import randbelow

import config
import numpy as np
from tensorflow.keras.models import model_from_json
from utils import create_input_array

warnings.filterwarnings("ignore")

__all__ = ["allocate_encrypt_packet"]


# For every dataPacket allocate NNs and encrypt in parallel.
def allocate_encrypt_packet(packet, nets, filename, public_key_f):
    public_key = []
    encrypted_text = []
    with open(filename, "wb") as f:
        for bit in packet:
            net = randbelow(2)  # Choose the neural network randomly (0 or 1)
            public_key.append(net)  # Store the choice of network in the public key
            bit_arr = create_input_array(bit)  # Convert the bit to input array format
            encoded = nets[net].predict(bit_arr)  # Encrypt the bit using the chosen network
            np.save(f, encoded)  # Save the encrypted bit to file
            encrypted_text.append(encoded)  # Append the encoded bit to the list

    # Save the public key to file
    np_public_key = np.array(public_key)
    np.save(public_key_f, np_public_key)

    # Print public key and encrypted text
    print("Public Key (Network Indices):", public_key)
    print("Encrypted Text:", encrypted_text)

    return (f, public_key_f)


if __name__ == "__main__":

    # Load json and create the small encrypter model
    with open(config.ENC_SMALL_JSON, "r") as json_file:
        read_model_json = json_file.read()

    encrypter_small = model_from_json(read_model_json)
    encrypter_small.load_weights(config.ENC_SMALL_MODEL)
    print("Loaded small encrypter model from disk")

    # Load json and create the large encrypter model
    with open(config.ENC_LARGE_JSON, "r") as json_file:
        read_model_json = json_file.read()

    encrypter_large = model_from_json(read_model_json)
    encrypter_large.load_weights(config.ENC_LARGE_MODEL)
    print("Loaded large encrypter model from disk")

    # Compile models
    encrypter_small.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])
    encrypter_large.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

    # Input text to encrypt
    packet = input("Enter the Text you want to encrypt: ")
    nets = [encrypter_small, encrypter_large]

    # Perform encryption and print public key and encrypted text
    encrypted_file, public_key = allocate_encrypt_packet(
        packet, nets, config.ENCRYPTED_FILE_PATH, config.PUBLIC_KEY_PATH
    )
