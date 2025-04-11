# movie_rec_system

## About Dataset
This dataset is a Kaggle Dataset named **Full TMDB Movies Dataset 2024** 

The TMDb (The Movie Database) is a comprehensive movie database that provides information about movies, including details like titles, ratings, release dates, revenue, genres, and much more.

This dataset contains a collection of 1,000,000 movies from the TMDB database, and it is updated daily.

## Leanrning objectives:
1. Learn about one type of candidate generation: content-based filtering
2. Learn typical recommendation model building workflow
3. Learn how to preprocess different types of data: numerical & binary & multi-label using different scikit learn packages
4. Learn the typical word embedding process (tokenization, remove stopwords, word lemmatizaion, compare euclidean distance) using tfidf vectorizating
5. learn to reduce vectors dimensions using PCA
6. Learn to use cosine similarity to combine all movies and recommend top movies using title searchup
   
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
5. keyword vectorization
   preprocess keywords using redex and find top 1000 most frequeent keywords using tiidf, which measures movies based on their similarity
  
7. assign weighting & normalization 
   change all to numeric features, assign weights based on emperical knowledge, and normalize all features using standard scaler.

Model Deployment: 
* candidate generation testing on local terminal
    * get user input on which movie they want to find similar
    * compare cosine simlarity matrix using preprocessed datasets
    * combine similarity scores from eras, runtimes, adult, genres and keywords with fixed weights
* get user inputs via UI(html)
    * create a movie recommender website with 
    * create a back-end python file using to generate top n candidates from embedded moviesets and receive user feedback
    * create a front-end html file 

### Potential Issues
* no collaborative content filtering -- due to the lac of user feedback --> need UI
* only display candidate generation stage in recommendation system -- scoring & re-ranking is necessary in future
  
## Resources
Resources:
content-based filtering:<br>
@https://developers.google.com/machine-learning/recommendation/content-based/basics<br>
imdb popularity meaning
https://community-imdb.sprinklr.com/conversations/imdbcom/popularity-rating/61ffe22823c1a32f12e18c7f
history of films:<br>
@https://en.wikipedia.org/wiki/History_of_film<br>
different movie eras:<br>
@https://www.videomaker.com/how-to/directing/film-history/film-history-the-evolution-of-film-and-television/<br>
word embedding using FastText<br>
@https://github.com/facebookresearch/fastText<br>
principal component analysis<br>
@https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html<br>
@https://stackoverflow.com/questions/77309231/reduce-an-embeddings-dimensions-using-pca
web dev using Flask
@https://code.visualstudio.com/docs/python/tutorial-flask



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
 
* 0.2
    * Upgraded word embedding to find most frequent keywords using tf-idf (file too large for github free storage)
