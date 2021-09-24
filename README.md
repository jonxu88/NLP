# NLP

Self-learning NLP.

## Seq2Seq

Used https://keras.io/examples/nlp/lstm_seq2seq/ to learn seq2seq. Tried separating into two files (training and inference). This caused a bug involving using the same name for multiple layers, which is fixed by using the add_prefix function (this adds a prefix to the names of the layers that are initiated for the new model). Also note the "if __name__ == "__main__":" idiom to isolate code that is to be run only when the script itself is run, and *not* when the module is called from another file via the "import" command.
