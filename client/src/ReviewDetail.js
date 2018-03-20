import React from 'react';
import PropTypes from 'prop-types';
import { withStyles } from 'material-ui/styles';

const styles = theme => ({
  root: {
    width: '100%',
    maxWidth: 360,
    backgroundColor: theme.palette.background.paper,
  },
});


class ReviewDetail extends React.Component {
  constructor(props) {
    super(props);
    this.classes = props.classes;
    this.state = {
      text: '',
    };
  }


  render() {
    return (
      <div className={this.classes.root}>
        {this.props.text}
      </div>
    );
  }
}

ReviewDetail.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(ReviewDetail);
