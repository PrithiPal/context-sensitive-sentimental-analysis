
# Phrasal Chunking

## Setup

    python3 -m venv venv
    source venv/bin/activate
    pip3 install -r requirements.txt

## Training phase

    python3 chunk.py > default.model

## Testing and Evaluation phase

    python3 perc.py -m default.model > output
    python3 score_chunks.py < output

OR

    python3 perc.py -m default.model | python3 score_chunks.py

## Options

    python3 chunk.py -h

This shows the different options you can use in your training
algorithm implementation.  In particular the -n option will let you
run your algorithm for less or more iterations to let your code run
faster with less accuracy or slower with more accuracy. You must
implement the -n option in your code so that we are able to run
your code with different number of iterations.

## Baseline

    $  python3 chunk.py -e 10 -m data/baseline.model
    reading data ...
    done.
    epoch 0
    epoch 1
    epoch 2
    epoch 3
    epoch 4
    epoch 5
    epoch 6
    epoch 7
    epoch 8
    epoch 9
    Training finished in  934.457836151123

    Note: The score below is based on ((e +1)/2) weight assignment
    $ python3 perc.py -m baseline.model | python3 score_chunks.py 
    reading data ...
    done.
    processed 500 sentences with 10375 tokens and 5783 phrases; found phrases: 5876; correct phrases: 5231
                 ADJP: precision:  54.55%; recall:  54.55%; F1:  54.55; found:     99; correct:     99
                 ADVP: precision:  65.02%; recall:  78.22%; F1:  71.01; found:    243; correct:    202
                CONJP: precision:  33.33%; recall:  40.00%; F1:  36.36; found:      6; correct:      5
                 INTJ: precision:   0.00%; recall:   0.00%; F1:   0.00; found:      0; correct:      1
                   NP: precision:  89.43%; recall:  91.11%; F1:  90.26; found:   3083; correct:   3026
                   PP: precision:  96.93%; recall:  95.82%; F1:  96.38; found:   1207; correct:   1221
                  PRT: precision:  71.43%; recall:  45.45%; F1:  55.56; found:     14; correct:     22
                 SBAR: precision:  73.33%; recall:  82.24%; F1:  77.53; found:    120; correct:    107
                   VP: precision:  89.86%; recall:  90.18%; F1:  90.02; found:   1104; correct:   1100
    accuracy:  94.18%; precision:  89.02%; recall:  90.45%; F1:  89.73
    Score: 89.73  

