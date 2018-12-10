#!/bin/bash

mkdir example_embeddings
cd example_embeddings
curl -o glove_embeddings.zip https://nlp.stanford.edu/data/glove.6B.zip
unzip glove_embeddings.zip
rm glove_embeddings.zip glove.6B.50d.txt glove.6B.300d.txt glove.6B.200d.txt
python example.py
