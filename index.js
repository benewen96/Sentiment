require('dotenv').config();

const express = require('express');
const exphbs = require('express-handlebars');
const bodyParser = require('body-parser');
const cookieParser = require('cookie-parser');
const atob = require('atob');

const sentiment = require('./sentiment_controller/sentiment');


const app = express();

const hbs = exphbs.create({
  defaultLayout: 'main',
  extname: '.hbs',
});


app.use(bodyParser.json());
app.use(cookieParser());

// serve our static stuff like index.css
app.use(express.static('public'));


// set our app's view engine to use the handlebars template engine
// default layout is main.hbs
app.engine('.hbs', hbs.engine);

app.set('view engine', '.hbs');


app.get('/', (req, res) => {
  res.render('home');
});

app.get('/data', (req, res) => {
  sentiment.initialiseData()
    .then((data) => {
      res.json(data);
    });
});

app.get('/review', (req, res) => {
  sentiment.findReviews(req.query)
    .then((review) => {
      res.send(review);
    })
    .catch((e) => {
      throw new Error(e);
    });
});

app.get('/business', (req, res) => {
  sentiment.getBusiness(req.query)
    .then((business) => {
      res.send(business);
    })
    .catch((e) => {
      throw new Error(e);
    });
});

app.get('/reviewIds', (req, res) => {
  sentiment.getReviewIds(req.query)
    .then((review) => {
      res.send(review);
    })
    .catch((e) => {
      throw new Error(e);
    });
});

app.get('/accuracy', (req, res) => {
  sentiment.getAccuracy()
    .then((accuracy) => {
      res.json(accuracy);
    })
    .catch((e) => {
      throw new Error(e);
    });
});

app.post('/similar', (req, res) => {
  sentiment.getSimilarReviews(req.body)
    .then((similar) => {
      res.json(similar);
    })
    .catch((e) => {
      console.log(e);
    });
});

app.listen(process.env.PORT);
