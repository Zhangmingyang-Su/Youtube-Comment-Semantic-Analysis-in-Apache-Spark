# Youtube Comment Semantic Analysis

In this notebook, we have a dataset of user comments for youtube videos related to animals or pets. We will attempt to identify cat or dog owners based on these comments, find out the topics important to them, and then identify video creators with the most viewers that are cat or dog owners.

## Project Summary:   
Overview : This project aims to build up classification models for the cat and dog owners from text comments. There are around 6,000,000 comments recorded in the dataset and the dataset itself is unlabeled. Each data row will have an userid, an text comment and belongs to one channel. 

## Details:
1. Searched specific terms which a cat/dog owner might have, and label those users as dog&cat owners.  Also labeled users who don't have pets as those whose comments don't contain any specific terms.  By this way, we turn our dataset into labeled ones and convert problem from unsupervised to supervised. 

2. Use RegexTokenizer to tokenize text comments.RegexTokenizer allows more advanced tokenization based on regular expression (regex) matching. By default, the parameter “pattern” is used as delimiters to split the input text. Alternatively, users can set parameter “gaps” to false indicating the regex “pattern” denotes “tokens” rather than splitting gaps, and find all matching occurrences as the tokenization result.  

3. Use Word2Vec represent text features as vectors. 

4. Train Logistic regression model and random forest model to classify audiences. Evaluation metrics are include: precision, recall, accuracy, AUC. Based on my analysis (I only randomly pick a small portion of dataset for training in order to speed process up)  random forest outperforms LR model.

5. Use our trained model to predict all users in the dataset. And get statistical sense of how dog&cat owners' distribution.  Extract word frequency to see related topics regards to those owner.

## Future Work:
According to the work (using only 35% of data because the databricks crashed often with the whole data), around 13% of the total user who commented on Youtube in this dataset are dog or cat owners. The potential topics they are interested in are include vet, white, eye, boo, half, lion, zoo, sugar, ferre, etc. Videos related to these topics could be promoted to these cat or dog owners. Also, 'The Dodo', 'brave wilderness' and 'Robin Seplut' are the top three creators with largest distinct cat or dog owner audience population (based only on the 5% data). Ads targeting cat or dog owners will potentially have the biggest payback cooperating with these creators.

For future work, this work could be improved in the following aspects: 
  1. Use all of the data.
  2. Specify the dog and cat owners based on more information.
