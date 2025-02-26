# movie_rec_system

## About Dataset
This dataset is a Kaggle Dataset named **Full TMDB Movies Dataset 2024** 
<br><br>
The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres, and much more.
<br><br>
This dataset contains a collection of 1,000,000 movies from the TMDB database, and it is updated daily.

## Leanrning objectives:
1. Learn about one types of candidate generation: content-based filtering
2. Learn typical recommendation model building workflow
3. Learn several sickit learn packages that help preprocess data for content-based filtering
4. Learn the typical word embedding process (tokenization, remove stopwords, word lemmatizaion, compare euclidean distance)
5. learn to peform unsupervised learning - k-means clustering, and how to reduce dimensions using PCA
6. Learn to use cosine similarity as similarity measure
   
## Project Workflow
Model Development: 
1. data cleaning
    eliminating null values & duplicates
    pay attention to value counts and find out if each category is unique
    pay attention to the data types, and whether more relevant data can be extracted
2. feature selection
   determine which features are relevant for our model
3. feature engineering
    define eras, runtime, andlanguages, then convert categorical data to numeric using dummy variables (only major categories)
4. multi-label encoding
    encode all genres using multilabel binarizer 
5. keyword clustering
   preprocess keywords and vevorize them into dense metrix using word embedding, then clusrer keywords into 10 different labels
   map the resulting label to each processed words, so that each processed words will have dummy variables from label 0 to label 9
   * for every word found in the 'keywords' column, if this word is found in one of the label, increment 1 under the corresponding label, so that for every row in the original dataframe, value under each label will show how many words in this row belong to each label
  
7. assign weighting & normalization 
   assign weights to different features, and normalize all numetic features using standard scaler

Model Deployment<br>
* candidate generation
    * get user input on which movie they want to find similar
    * compare cosine simlarity matrix using preprocessed datasets
    * combine similarity scores from eras, runtimes, adult, genres, overview, and keywords with adjustable weights


### Dependencies
* The process of Analysis is stored in the Jupyter Notebook **'recommendation_system.ipynb'**.
* Pre-req Python Libraries include: numpy, pandas, maplotlib, sklearn, nltk, gensim.

### Installing
* You can download this project [here](https://github.com/kkrit-tinna/movie_rec_system).

### Executing Program
* To begin, locate **'recommendation_system.ipynb'**, and run the last cell, which performs function get_recommendation()
* **DATASETS ARE TOO LARGE FOR GITHUB STORAGE, CANNOT EXECUTE CODE**. <br>You can download this notebook and find the original datasets on [kaggles](https://www.kaggle.com/code/shikristin/movie-recommendation-system)).


### Potential Issues
* no collaborative content filtering -- due to the lack of data accuracy in columns such as poularity, rating, etc.
* only display candidate generation stage in recommendation system -- scoring & re-ranking is necessary in future
  
## Resources
Resources:
content-based filtering:<br>
@https://developers.google.com/machine-learning/recommendation/content-based/basics<br>
history of films:<br>
@https://en.wikipedia.org/wiki/History_of_film<br>
different movie eras:<br>
@https://www.videomaker.com/how-to/directing/film-history/film-history-the-evolution-of-film-and-television/<br>
word embedding using FastText<br>
@https://github.com/facebookresearch/fastText<br>
principal component analysis<br>
@https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html<br>
k means clustering<br>
@https://www.geeksforgeeks.org/k-means-clustering-introduction/<br>
elbow method<br>
@https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/<br>
k means inertia<br>
@https://www.codecademy.com/learn/dspath-unsupervised/modules/dspath-clustering/cheatsheet<br>


## Author
Yifei Shi


## Version History
* 0.1
    * Initial Release
