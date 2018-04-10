import React from 'react';
import { BrowserRouter as Router, Route, Link } from 'react-router-dom';
import { withStyles } from 'material-ui/styles';
import AppBar from 'material-ui/AppBar';
import Toolbar from 'material-ui/Toolbar';
import Typography from 'material-ui/Typography';
import Grid from 'material-ui/Grid';
import Paper from 'material-ui/Paper';
import List, { ListItem, ListItemText } from 'material-ui/List';
import Avatar from 'material-ui/Avatar';
import { DateRange, Mood, MoodBad, Assessment } from 'material-ui-icons';
import Collapse from 'material-ui/transitions/Collapse';
import { LinearProgress } from 'material-ui/Progress';
import TextField from 'material-ui/TextField';

import Review from './Review';

const pageHeight = window.innerHeight - 64;

const styles = theme => ({
  root: {
    flexGrow: 1,
  },
  flex: {
    flex: 1,
  },
  category: {
    margin: theme.spacing.unit * 2,
  },
  reviewPane: {
  },
  accuracy: {
    marginRight: '5px',
  },
  detail: {
    padding: theme.spacing.unit * 2,
    // margin: theme.spacing.unit * 2,
  },
  typography: {
    fontSize: '9.5pt',
  },
  chip: {
    margin: theme.spacing.unit,
  },
  searchPane: {
    overflowY: 'scroll',
    overflowX: 'hidden',
    height: pageHeight,
  },
  good: {

    overflowY: 'scroll',
    overflowX: 'hidden',
    height: pageHeight,
  },
  bad: {
    overflowY: 'scroll',
    overflowX: 'hidden',
    height: pageHeight,
  },
});


class App extends React.Component {
  constructor(props) {
    super(props);
    this.classes = props.classes;
    this.state = {
      visible: false,

    };
    this.selectBusiness = this.selectBusiness.bind(this);
    this.updateDetail = this.updateDetail.bind(this);
    this.renderReview = this.renderReview.bind(this);
    this.renderSentiment = this.renderSentiment.bind(this);
    this.search = this.search.bind(this);
  }

  componentDidMount() {
    this.timer = setInterval(this.progress, 500);

    fetch('/accuracy')
      .then(response => response.json())
      .then((accuracy) => {
        this.setState({ accuracy });
      });
  }

  componentWillUnmount() {
    clearInterval(this.timer);
    this.timer = null;
  }

  search(event) {
    setTimeout(() => {

    });
    fetch(`/business?name=${event.target.value}`)
      .then(response => response.json())
      .then((foundBusinesses) => {
        this.setState({ foundBusinesses });
      });
  }

  selectBusiness(b) {
    this.setState({
      currentBusiness: b,
    });

    fetch(`/review?sentiment=Good&limit=100&business_id=${b.business_id}`)
      .then(response => response.json())
      .then((reviews) => {
        this.setState({ goodReviews: reviews });
      });

    fetch(`/review?sentiment=Bad&limit=100&business_id=${b.business_id}`)
      .then(response => response.json())
      .then((reviews) => {
        this.setState({ badReviews: reviews });
      });
  }

  genReviewComponents(reviews) {
    console.log(reviews);
    const limit = 10;
    return reviews.map(review => <Review key={review.review_id} review={review} id={review.review_id} updateDetail={this.updateDetail} />).slice(0, limit - 1);
  }

  updateDetail(reviewState) {
    this.setState({
      reviewState,
      visible: true,
    });
  }

  renderSentiment() {
    if (this.state.reviewState) {
      if (this.state.reviewState.review.sentiment === 'Good') {
        return <Mood />;
      } else if (this.state.reviewState.review.sentiment === 'Bad') {
        return <MoodBad />;
      }
    }
    return null;
  }

  renderReview() {
    return (
      <div className={this.classes.reviewPane}>
        <Collapse in={this.state.visible}>
          <List>
            <ListItem>
              <Avatar>
                <DateRange />
              </Avatar>
              <ListItemText primary="Date" secondary={this.state.reviewState && this.state.reviewState.review.date} />
              <Avatar>
                <DateRange />
              </Avatar>
              <ListItemText primary="Business" secondary={this.state.reviewState && this.state.reviewState.business && this.state.reviewState.business.name} />
              <Avatar>
                {this.renderSentiment()}
              </Avatar>
              <ListItemText primary="Sentiment" secondary={this.state.reviewState && `${this.state.reviewState.review.sentiment}`} />

            </ListItem>
            <ListItem>
              <Paper className={this.classes.detail} elevation={4}>
                <Typography className={this.classes.typography}>
                  {this.state.reviewState && this.state.reviewState.review.text}
                </Typography>
              </Paper>
            </ListItem>

          </List>

          {this.state.reviewState && this.state.reviewState.similar.map(review =>
            <Review key={review.review_id} review={review} id={review.review_id} updateDetail={this.updateDetail} />)}
        </Collapse>
      </div>);
  }

  render() {
    return (
      <Router>
        <div>
          <AppBar position="sticky" color="default">
            <Toolbar>
              <Typography variant="title" color="inherit" className={this.classes.flex}>
                {'Review Analyser'}
              </Typography>
              <Typography variant="body1" className={this.classes.accuracy}>System Accuracy: </Typography>
              <Avatar>
                <Typography variant="body2">{this.state.accuracy && `${this.state.accuracy}%`}</Typography>
              </Avatar>
            </Toolbar>
          </AppBar>
          <div className={this.classes.root}>
            <Grid container spacing={0} >
              <Grid item sm={2} className={this.classes.searchPane}>
                <TextField
                  styles={{ marginLeft: '5px' }}
                  id="search"
                  label="Search for a business"
                  type="search"
                  margin="normal"
                  onChange={this.search}
                />

                <List component="nav">
                  {this.state.foundBusinesses && this.state.foundBusinesses.map(b =>
                  (
                    <ListItem key={b.id} button onClick={() => { this.selectBusiness(b); }}>
                      <ListItemText primary={b.name} />
                    </ListItem>
                  ))}
                </List>
              </Grid>
              <Grid item sm={2} className={this.classes.good}>
                <Typography className={this.classes.category} variant="title" align="center">Good</Typography>
                {this.state.goodReviews && this.genReviewComponents(this.state.goodReviews)}
              </Grid>
              <Grid item sm={2} className={this.classes.bad}>
                <Typography className={this.classes.category} variant="title" align="center">Bad</Typography>
                {this.state.badReviews && this.genReviewComponents(this.state.badReviews)}
              </Grid>
              <Grid item sm={6} className={this.classes.bad}>
                {this.renderReview()}
              </Grid>
            </Grid>
          </div>
        </div>
      </Router>
    );
  }
}

export default withStyles(styles)(App);
