1. **What do you know? A topic-model approach to authority identification (****passos****)**
  1. [Intuition : Good topics or reviews have good content words]

-
  1. Identifies the documents that have authority (which sounds more bjective and meaninful review overall).

Three approaches used :

-
  1.
    - For each product
      - Heuristic based : unique : Rank document d with number of words unique to the document collection.
      - Summarization based : Rank documents d with the highest frequency word w.r.t to the product p&#39;s high frequency word vector
      - Discriminative : Logistical regression for ranking documents d   to pick best review for each product p. Feature : word identity, punctuation, sent structure(word length, average sent len, num words, num sents etc..)



- **Incoporating****  Phrase-level Sentiment Analysis on Textual reviews for Personalized Recommendation**
  1. Just contained the idea of using SA to the collaborative filtering. Good for CF based systems but do not concentrates the SA
- **Deep learning for Sentimental Analysis : A Survey**
  1. Give generalized introduction to all the improtant NN frameworks in NLP : encoder/decoder (for the seq2seq models), word embeddings (input to the seq2seq models), CNN, RNN , LSTM.
  2. Scopes :
    - **Document-level ** : targets : polarity, objectivity/subjectivity, positive/neutral/negative in NN
    - **Sentence-level**  : same targets as the above. Uses NN with word-embeddings
    - **Aspect based with NN**  (automatic feature dev.) :
      - Represent the context of a target.
      - Target representation to interact with the context.
      - Third task is to identify the important sentiment context (words) for the specific target.
      - [ATTENTION MODELS]
  3. Look into the papers in the Aspect SA.

1. **Exploiting Expectation Maximization Algorithm for SA for Product Reviews.**
  1. The paper proposed the EM algorithm for product review .
  2. Methodology :
    1. Divide the corpus into three parts : product features, sentiment features and everything else is SW.
    2. Have to carefully read it though, but may seems achievable.
    3. Results :
    4. Precision Recall FM Accuracy
    5. 0.9   0.875   0.88   0.83

1. **Building a Sentiment Summarizer for Local Service **** Revies****[GOOGLE PAPER]**
  1. Build majorly two things :  **sentiment classifier**  +  **aspect extractor**.
  2. **Sentiment Classifier [Hybrid]**:
    1. **Lexicon-based**  : tag the positive/negative and expand to the synonyms (same) and antonyms (opposite.) + confidence (label-propagation algorithm)
    2. **Algorithm** : start with prior score, re-iterate set weights and in the end of 5 iterations, you will have more lexicons with the weights.
  3. **Aspect Extractor[algorithm to combine dynamic and static extractions.] : **
    1. **Dynamic Extractor[extracts aspects automatically]: **identifies the sentimentic sentences, extract the aspects which are nouns(Nouns Compounds upto three words),high-frequent.
    2. **Static Extractor[scores the aspect sentiment with provided aspects] :  **mapper from sentences to the fine-grained aspects(aspects are hand-labelled this time.) Chooses aspects such as food, décor,service,value.
  4. **Summarizer, takes : **
    1. **S : ** setnences which contains the sentiment
    2. **P**  : polarity for all sentences  **S**
    3. **A : ** ranked aspects
    4. **C**  : scored aspects  **A ** for all sentences  **S ** [a matrix]
    5. **L ** : length of summary needed(num of sentences)

1.
  1. **Results ** :

**Precision Recall F1**

Service 86.9 / 82.3 66.9 / 66.6 75.6 / 73.6

Value 90.3 / 94.1 55.6 / 65.6 68.9 / 77.4

1.
  1. The  **future expansion**  talks about the UI study to see the best way to deliver the SA and doing the same study on the product reviews (if few products have the most reviews)

1. **Clustering Product Features for Opinion Mining[supervised except the soft-constraints]**
  1. The paper tackles the problem of grouping all the features together that inherently corresponds to the same target (or aspect) of a produce/service.
  2. Steps :
    1. **Generating Labeled Data L [from given soft-constraints]** :
      1. **Sharing words**  : attributes in the same sentences are likely to belong to same group.
      2. **Lexical similarity**  : lexical similar words (wordnet) also fall into same group
      3. **Same context**  : two features in the same emotional sentence cannot  be synonyms (likely talking about different attributes.)

1.
  1.
    1. **EM algorithm with Naïve Bayes (labels comes from the soft-constraints above)**
      1. **Input : **
    2. **Context Extraction  : ** A vector representation of the context of each feature fi. For example : di-\&lt;ldc,scnree,gives,clear\&gt;, dk=\&lt;gives,clear\&gt;
      1. Distributional context extraction algorithm is given.
2. **Automated Rule Selection for Aspect Extraction in Opinion Mining**



1. **Combining Lexicon-based and Learning-based Methods for Twitter Sentiment Analysis [unsupervised method for SA except Lexicons] but entity is given by user.**

**[Paper makes good use of grammar]**

1. Steps :
  1. **Preprocessing**
  2. **Classify sentences** : Declarative, Imperative and Interrogative. Interrogative does not give info on Sentiment.
  3. For sentence s, identify opinion words(sentiment words), score each word using lexicon and aggregate the score for whole sentence using a score(e) formula
  4. **Comparative sentences ** : not consider comparative sentences. Identified by POS tags of words: JJR, RBR, JJS and RBS
  5. **Chi-square**  : if words are more likely to occur in emotional sentences(positive or negative), those words are more likely to be sentiment words. Helps to identify positive/negative indicators ( **emotional words.** )
  6. **Entity words**  : NNP or NN occuring in a emotional sentence.
  7. **Context-dependent opinion words**  : Co-occurrence of positive words.
2.