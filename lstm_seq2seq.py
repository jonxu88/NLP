#testing git

import numpy as np
import tensorflow as tf
from tensorflow import keras

batch_size = 64
epochs = 100
latent_dim = 256  # Latent dimensionality of the encoding space
num_samples = 10000

data_path = "/Users/jonxu/code/KerasExamples/fra.txt"

# Vectorize the data

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[ : min(num_samples, len(lines) - 1) ]:
    input_text, target_text, _ = line.split("\t")
    target_text = "\t" + target_text + "\n"
    # Use "tab" as the "start sequence" character in the RNN
    # and "\n" as the "end sequence" character
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([ len(txt) for txt in input_texts ])
max_decoder_seq_length = max([ len(txt) for txt in target_texts ])
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# tensor of zeroes with dimensions #input_texts x max_encoder_seq_length x num_encoder_tokens
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    encoder_input_data[i, t+1 :, input_token_index[" "]] = 1.0
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_target_data[i, t-1, target_token_index[char]] = 1.0
    decoder_input_data[i, t+1:, target_token_index[" "]] = 1.0
    decoder_target_data[i, t:, target_token_index[" "]] = 1.0
# stuff only to run when not called via 'import' here
if __name__ == "__main__":
    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)
    print("Max sequence length for outputs:", max_decoder_seq_length)
    # Define an input sequence and process it
    # None elements represent dimensions where the shape is not known
    encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
    # latent_dim is the Latent dimensionality of the encoding space,
    # so it will be the dimension of the output space
    # the return_state argument returns a list where the first entry
    # is the outputs and the next entries are the internal RNN states
    encoder = keras.layers.LSTM(latent_dim, return_state=True)
    # the LSTM layer returns a whole sequence output, a memory state, and a carry state
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    # discard 'encoder_outputs' and only keep the states
    encoder_states = [state_h, state_c]

    # set up the decoder, using 'encoder_states' as initial state.
    decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return state states in the training model, but we wil use them in inference.
    decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Train the model
    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    # Save model
    model.save("s2s")
