Natural Language Processing with Disaster Tweets
------------------------------------------------
A Kaggle "Getting Started" competition solution: classifying tweets as real disaster reports (1) or not (0).

* Competition: NLP with Disaster Tweets
* Task: Binary text classification
* Evaluation metric: F1 Score
* Final Kaggle Score: 0.82133

1. Dataset
    File	    | Rows	| Columns
    ------------|-------|------------------------------------
    train.csv	| 7,613	| id, keyword, location, text, target
    test.csv	| 3,263	| id, keyword, location, text

    Target distribution (train):
    Class	              | Count
    ----------------------|------
    0 (not disaster)	  | 4,342
    1 (disaster)	      | 3,271
    
    Missing values:
    Column	   | Train missing	| Test missing
    -----------|----------------|-------------
    keyword	   |  61	        | 26
    location   |  2,533	        | 1,105
    text	   |  0	            | 0

2. Data Preprocessing
   * Missing value handling:
      1. keyword -> filled with 'none'
      2. location -> filled with 'unknown' (dropped from modeling due to high noise/missing ratio)
   * Feature construction: keyword and text were combined into a single text_combined column, since single-column vectorizers (TF-IDF) and tokenizers (BERT) work on one text field.
   * Text normalization (lowercasing, URL/mention removal, punctuation stripping) was applied only for the TF-IDF pipeline. For BERT-based models, raw text was used intentionally — pretrained 
     transformers already understand punctuation, casing, and sentence structure, and heavy cleaning removes useful signal (e.g. emphasis, tone).

3. Experiments
   All experiments used an 80/20 stratified train/validation split (random_state=42) for fair comparison, with F1 Score as the primary metric (matching Kaggle's evaluation).

   #	Approach	                             Details	                                                                                                        Validation F1	     Validation Accuracy
   1	TF-IDF + Logistic Regression (baseline)	 max_features=10000, unigrams only	                                                                                0.7357	             79%
   2	TF-IDF + Logistic Regression (tuned)	 max_features=20000, ngram_range=(1,2), min_df=2	                                                                0.7449	             79%
   3    DistilBERT fine-tuned	                 distilbert-base-uncased, 3 epochs, lr=2e-5	                                                                        0.7988	             83%
   4	Twitter-RoBERTa fine-tuned	             cardiffnlp/twitter-roberta-base-sentiment-latest, 5 epochs, lr=3e-5, classifier head reinitialized (3->2 classes)	0.7788	             82%
   5	DistilBERT fine-tuned (final)	         distilbert-base-uncased, 5 epochs, lr=2e-5	                                                                        0.8016	             84%

# Why DistilBERT outperformed Twitter-RoBERTa
-> Although Twitter-RoBERTa was pretrained specifically on tweet data, its original classification head (built for 3-class sentiment) 
had to be discarded and reinitialized for this 2-class task. Combined with RoBERTa-base being a larger model (~125M params vs DistilBERT's ~66M), 
it didn't converge as well within the given epoch budget on this relatively small dataset (~6k training tweets).

4. Final Model — DistilBERT (5 epochs)
Architecture: AutoModelForSequenceClassification (DistilBERT base, uncased) with a 2-class classification head, fine-tuned end-to-end.

Training setup:
Framework: PyTorch Lightning
Tokenizer: AutoTokenizer (distilbert-base-uncased), max sequence length 128
Batch size: 16
Learning rate: 2e-5
Epochs: 5
Optimizer: AdamW
Hardware: Google Colab (GPU)

Validation classification report:
              precision    recall  f1-score   support

           0       0.83      0.90      0.86       869
           1       0.85      0.76      0.80       654

    accuracy                           0.84      1523
   macro avg       0.84      0.83      0.83      1523
weighted avg       0.84      0.84      0.84      1523

5. Kaggle Submission
Metric	                       |Value
-------------------------------|-------------------------
Submission file	               |submission_DistilBERT.csv
Validation F1 (local)          |0.8016
Kaggle Public F1 (official)	   |0.82133

The Kaggle score came in slightly higher than the local validation score, suggesting the model generalized well and did not overfit to the training/validation split.

6. Pipeline Summary
Data          ->  Load CSVs → handle missing values → combine keyword+text
Model         ->  Train/val split → tokenize (BERT tokenizer) → load pretrained model
Training      ->  Fine-tune with PyTorch Lightning → validate (F1) → predict on test set
Submission    ->  Format predictions as id,target → submit to Kaggle

7. Key Learnings
   1. Preprocessing depends on the model. TF-IDF needs aggressive text cleaning; transformer models perform better with raw, natural text.
   2. Pretrained domain relevance doesn't guarantee better results. A Twitter-specific model underperformed a general-purpose model here, largely due to the classifier head being reinitialized and model size/training budget tradeoffs.
   3. Fine-tuning a pretrained transformer gave a ~9% F1 improvement over a classical TF-IDF + Logistic Regression baseline (0.7357 -> 0.8213).
   4. Validation score closely tracked the actual Kaggle leaderboard score, confirming a leak-free, properly stratified train/validation split.