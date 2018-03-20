import React from 'react';
import { withStyles } from 'material-ui/styles';
import Card, { CardActions, CardContent } from 'material-ui/Card';
import Grid from 'material-ui/Grid';
import Button from 'material-ui/Button';

import Typography from 'material-ui/Typography';

const getRelatedReview = id => new Promise(((resolve, reject) => {
  fetch(`/review?review_id=${id}`)
    .then(response => response.json())
    .then((review) => {
      if (review.length === 0) {
        resolve(null);
      } else {
        review[0].date = new Date(review[0].date).toLocaleDateString('en-US');
        resolve(review[0]);
      }
    })
    .catch((e) => {
      reject(e);
    });
}));

const getBusinessData = id => new Promise(((resolve, reject) => {
  fetch(`/business?business_id=${id}`)
    .then(response => response.json())
    .then((business) => {
      resolve(business[0]);
    })
    .catch((e) => {
      reject(e);
    });
}));


const styles = theme => ({
  card: {
    margin: theme.spacing.unit * 2,
  },
  title: {
    marginBottom: 16,
    fontSize: 14,
    color: theme.palette.text.secondary,
  },
  pos: {
    marginBottom: 12,
    color: theme.palette.text.secondary,
  },
});

class Review extends React.Component {
  constructor(props) {
    super(props);
    this.classes = props.classes;

    this.state = {
      review: {},
    };

    this.getSimilar = this.getSimilar.bind(this);
  }


  componentWillMount() {
    fetch(`/review?review_id=${this.props.id}`)
      .then(response => response.json())
      .then((review) => {
        review[0].date = new Date(review[0].date).toLocaleDateString('en-US');
        this.setState({
          review: review[0],
        });
      })
      .then(() => getBusinessData(this.state.review.business_id))
      .then((business) => {
        this.setState({
          business,
        });
      })

      .catch((e) => {
        console.log(e);
      });
  }

  getSimilar() {
    fetch('/similar', {
      headers: {
        Accept: 'application/json',
        'Content-Type': 'application/json',
      },
      method: 'POST',
      body: JSON.stringify({ text: this.state.review.text }),
    })
      .then(res => res.json())
      .then(data => Promise.all(data.map(id => getRelatedReview(id))))
      .then(similarReviews => similarReviews.filter((review) => {
        if (review != null) {
          return true;
        }
        return false;
      }))
      .then((filtered) => {
        this.setState({
          similar: filtered,
        });
        this.props.updateDetail(this.state);
      })

      .catch(e => console.log(e));
  }
  render() {
    return (
      <div>
        <Grid container>
          <Grid item xs>
            <Card className={this.classes.card} styles={{ backgroundColor: 'grey' }}>
              <CardContent>
                <Typography className={this.classes.title}>{this.state.review.date}</Typography>
                <Typography variant="headline" component="h2" />
                <Typography className={this.classes.pos}>We think this is a {this.state.review.sentiment} review.</Typography>
                <Typography paragraph noWrap component="p">{this.state.review.text}</Typography>
              </CardContent>
              <CardActions>
                <Button
                  size="small"
                  onClick={() => {
                    this.getSimilar();
                  }}
                >Open
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>
      </div>
    );
  }
}

export default withStyles(styles)(Review);
