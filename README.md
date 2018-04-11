# README

## Pre-requisites

### Libraries
Please ensure that these python libraries are installed:
 - [Gensim](https://radimrehurek.com/gensim/install.html)
 - [scikit-learn](http://scikit-learn.org/stable/install.html)
 - [NumPy](http://www.numpy.org/)
 - [Pandas](https://pandas.pydata.org/)
 - [Node.js](https://nodejs.org/en/)
 - [MongoDB](https://docs.mongodb.com/manual/administration/install-community/)

### Usage

#### Installation
From the root of the project directory, run the following commands to install the application dependencies:

 - `npm install`
 - `cd client && npm install`

#### Training
To train the model:
 - `cd sentiment_controller`
 - `python train.py 60000 1 10 yelp_model.d2v`

#### Classification
To classify the Yelp corpus after training:
 - Start the MongoDB database, ensure that it is available at       `localhost:27017`
 - `cd sentiment_controller`
 - `python database.py 60000 10 60000`

#### Running
Simply run `npm start` from the root of the project. The application will be visible in a web browser at: http://localhost:3000
