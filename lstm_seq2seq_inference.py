import numpy as np
import tensorflow as tf
from lstm_seq2seq import input_token_index, target_token_index, latent_dim, encoder_input_data, input_texts, \
    max_decoder_seq_length, num_decoder_tokens
from tensorflow import keras

# Define sampling models
# Restore the model and construct the encoder and decoder.

# This function is necessary so that Keras doesn't error out with the
# 'every layer must have a unique name' bug.
def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary.
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''

    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config[ 'layers' ]:
        new_name = prefix + layer[ 'name' ]
        old_to_new[ layer[ 'name' ] ], new_to_old[ new_name ] = new_name, layer[ 'name' ]
        layer[ 'name' ] = new_name
        layer[ 'config' ][ 'name' ] = new_name

        if len(layer[ 'inbound_nodes' ]) > 0:
            for in_node in layer[ 'inbound_nodes' ][ 0 ]:
                in_node[ 0 ] = old_to_new[ in_node[ 0 ] ]

    for input_layer in config[ 'input_layers' ]:
        input_layer[ 0 ] = old_to_new[ input_layer[ 0 ] ]

    for output_layer in config[ 'output_layers' ]:
        output_layer[ 0 ] = old_to_new[ output_layer[ 0 ] ]

    config[ 'name' ] = prefix + config[ 'name' ]
    new_model = tf.keras.Model().from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[ layer.name ]).get_weights())

    return new_model


model = keras.models.load_model("s2s")
model = add_prefix(model, prefix="bugfixprefix")

print("Model from s2s:", model.summary())

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)
print("New encoder model:", encoder_model.summary())

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

print(decoder_model.summary())

# Reverse-lookup token index to decode sequences back to something readable
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0,0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1, num_decoder_tokens))
        target_seq[0,0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(20):
    # Take one sequence for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
