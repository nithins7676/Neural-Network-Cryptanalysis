# Model save path
SAVE_PATH = "models/"

# Checkpoint for encrypter
ENC_LARGE_CHK = SAVE_PATH + "encrypter_large_chk"
DEC_LARGE_CHK = SAVE_PATH + "decrypter_large_chk"

# Actual models go here!
ENC_LARGE_MODEL = SAVE_PATH + "encrypter_large.weights.h5"  # Adjusted to match save
ENC_LARGE_JSON = SAVE_PATH + "encrypter_large.json"

DEC_LARGE_MODEL = SAVE_PATH + "decrypter_large.weights.h5"  # Adjusted to match save
DEC_LARGE_JSON = SAVE_PATH + "decrypter_large.json"

# Checkpoint for small models
ENC_SMALL_CHK = SAVE_PATH + "encrypter_small_chk.weights.h5"  
DEC_SMALL_CHK = SAVE_PATH + "decrypter_small_chk.weights.h5"  

# Actual models go here!
ENC_SMALL_MODEL = SAVE_PATH + "encrypter_small.weights.h5"  # Updated to .weights.h5
ENC_SMALL_JSON = SAVE_PATH + "encrypter_small.json"

DEC_SMALL_MODEL = SAVE_PATH + "decrypter_small.weights.h5"
DEC_SMALL_JSON = SAVE_PATH + "decrypter_small.json"

# Encryption Files
ENCRYPTED_FILE_PATH = SAVE_PATH + "encrypted_file.npy"
PUBLIC_KEY_PATH = SAVE_PATH + "public_key.npy"

# Training Hyperparameters
ENC_EPOCHS = 50
ENC_BATCH_SIZE = None
ENC_LEARNING_RATE = 0.0015
ENC_VALIDATION_SPLIT = 0.1

DEC_EPOCHS = 100
DEC_BATCH_SIZE = None
DEC_LEARNING_RATE = 0.0015
DEC_VALIDATION_SPLIT = 0.1

# # Model save path
# SAVE_PATH = "models/"

# # Check point for encrypter
# ENC_LARGE_CHK = SAVE_PATH + "encrypter_large_chk"
# DEC_LARGE_CHK = SAVE_PATH + "decrypter_large_chk"

# # Actual models go here !
# ENC_LARGE_MODEL = SAVE_PATH + "encrypter_large.h5"
# ENC_LARGE_JSON = SAVE_PATH + "encrypter_large.json"

# DEC_LARGE_MODEL = SAVE_PATH + "decrypter_large.h5"
# DEC_LARGE_JSON = SAVE_PATH + "decrypter_large.json"

# # Model save path
# SAVE_PATH = "models/"

# # Checkpoint for encrypter
# ENC_SMALL_CHK = SAVE_PATH + "encrypter_small_chk.weights.h5"  # Change here
# DEC_SMALL_CHK = SAVE_PATH + "decrypter_small_chk.weights.h5"  # Change here

# # Actual models go here!
# ENC_SMALL_MODEL = SAVE_PATH + "encrypter_small.h5"
# ENC_SMALL_JSON = SAVE_PATH + "encrypter_small.json"

# DEC_SMALL_MODEL = SAVE_PATH + "decrypter_small.weights.h5"
# DEC_SMALL_JSON = SAVE_PATH + "decrypter_small.json"


# # Encrpytion Files
# ENCRYPTED_FILE_PATH = SAVE_PATH + "encrypted_file.npy"
# PUBLIC_KEY_PATH = SAVE_PATH + "public_key.npy"

# # Training Hyperparameters

# ENC_EPOCHS = 50
# ENC_BATCH_SIZE = None
# ENC_LEARNING_RATE = 0.0015
# ENC_VALIDATION_SPLIT = 0.1

# DEC_EPOCHS = 100
# DEC_BATCH_SIZE = None
# DEC_LEARNING_RATE = 0.0015
# DEC_VALIDATION_SPLIT = 0.1
