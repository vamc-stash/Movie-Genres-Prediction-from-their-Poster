# Movie-Genres-Prediction-from-their-Poster

## Introduction

This project objective is to classify movie genre based only on movie poster images. Movie posters are one of the first impressions used to get an idea about the movie content and its genre based on things like color, objects, facial expressions of actors etc. 

## Proposed Approach
In order to classify movie posters by genre, I utilized the concept of multi-label multi-class classification with Convolution neural networks.
### Part 1. Dataset
We used this [dataset](https://www.kaggle.com/neha1703/movie-genre-from-its-poster).The dataset consists of around 38,000 movie posters scraped from IMDB. Each poster can have multiple genres associated with it. 

### Part 2. Pre-Processing 
Extract all genres available in the dataset and label them with indexes and store [here](https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/blob/master/code/label.json). Using the IMDB link of each movie (available in this dataset) we use a Web Scraping approach in order to retrieve its poster image from the IMDB movie page and save it locally [here](https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/tree/master/movie-genre-from-its-poster/MoviePosters).Also reshape poster images so that all of them has the same size, that will match the input size of our CNN. our Final Dataset which has as X variable "poster images" numpy arrays (obtained processing each image) and as Y variable the target variable "genres". For this project, I used only 17,000 samples of images for training and testing.

### Part 3. CNN
We can finally build our Convolutional Neural Network in order to classify movie genre basing on poster characteristics. For this purpose, we use Keras, a Python framework which allows to build Machine Learning models.</br>
The Keras model type that we will be using is Sequential, which is the easiest way to build a model, since it allows to to build a model layer by layer.</br>
Our first 4 layers are Conv2D layers. The first layer also takes an input shape, which is is the shape of each input image. The first two has 32 nodes, the second and the third have 64 nodes, while the last layer has 128 nodes.</br>
In our case the size of the filter matrix is 3, which means we will have a 3x3 filter matrix for each Conv2D layer.</br>
In between the Conv2D layers and the Dense layer, there is a Flatten layer, used as a connection between the convolution and dense layers.</br>
The activation for the last layer is "sigmoid" and loss function is "binary_crossentropy" since we are dealing with a multi-label classication problem.</br>
Train the model by running cnn file
```
 python3 /code/cnn.py
```

check out [model summary](https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/blob/master/model/train_summary.txt) trained on my system.
### Part 4. Testing the model
save movie posters to be tested inside [test-images](https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/tree/master/test-images) folder.</br>
To test, run following script</br>
```
python3 /code/test.py <image-file-name>
```

## Results
```
python3 /code/test.py seven.jpg
``` 
<img src="https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/blob/master/test-images/seven.jpg" alt="se7en" width="200" height="250"> </br>
#### Prediction
Drama 0.5605113 |
Thriller 0.29830286 |
Action 0.2179066 |
Horror 0.20707247 |
Crime 0.19378987 </br>

```
python3 /code/test.py enemy.jpg
```
<img src="https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/blob/master/test-images/enemy.jpg" alt="Enemy" width="200" height="250"> </br>
#### Prediction
Drama 0.50983024 |
Documentary 0.33969897 |
Comedy 0.1932565 |
Action 0.107409 |
Biography 0.10224339 </br>

```
python3 /code/test.py sr.jpg
```
<img src="https://github.com/vamc-stash/Movie-Genres-Prediction-from-their-Poster/blob/master/test-images/sr.jpg" alt="The Shawshank Redemption" width="200" height="250"> </br>
#### Prediction
Drama 0.61487967 |
Comedy 0.14339986 |
Documentary 0.12381813 |
Romance 0.10465199 |
Thriller 0.09669298 </br>
