CMPT 413 HW2




TASKS : Decipher the code.




1. Implement Beam Search
   1.  SCORE
  





1. NGRAM MODEL [ PRITHI WORKING ON] : Use to do score


   1. Default ngram score : For this, study the ngrams.py and use the scoring function in it.
   2. Improved ngram score : Some changes has to be made to improve the scoring system. 


       2. NEURAL MODEL [EKAM WORKING ON] : 
* And also the research paper to implement
http://anoopsarkar.github.io/nlp-class/assets/cached/decipherment_neural_lm_emnlp2018.pdf
  
       3. MIXTURE MODEL: 
* This is possible only if both ngrams and neural models are developed. So these both will be used to combine the scores.w
 TOTAL_MMSCORE=  


B. EXT_ORDER 
* This is the input to the beam search algorithm. Consists of the unique cipher symbols. The order is important for scoring, so the ordering should be thought of as stated in the research paper.
1. Old : sorted out by frequency of cipher characters. The first symbol is highest frequency and last in the list is lower frequency.   
2. New : Choose through maximum context at every step. Not understood everything though, but should follow the discussion on the canvas (https://coursys.sfu.ca/2018fa-cmpt-413-x1/discussion/topic/maximum-context/) : discussion about maximum context on coursys
   1. 



2. EVALUATION METHOD : 


   * Use already deciphered ciphers and test using held-out data. There are two famous ciphers as discussed in the research papers : 
   * Zodiac-408
   * Beale Cipher
   * Provided with our code, take some training data and train on it (Let that train be our cipher.txt)
   * See how algorithm deciphers it.
   * Compare with actual decipherment.