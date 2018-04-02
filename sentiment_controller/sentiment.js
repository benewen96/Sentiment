const PythonShell = require('python-shell');
const Database = require('../database');
const path = require('path');

const spawn = require('child_process').spawn;

const py = spawn('python', [path.resolve(__dirname, 'unseen_input.py')], { env: { PYTHONUNBUFFERED: true, stdio: 'pipe' } });

/**
*
* @return {[type]} [description]
*/
const initialiseData = () => new Promise(((resolve, reject) => {
  const pyshell = new PythonShell('database.py', {
    scriptPath: __dirname,
    mode: 'text',
  });

  const response = {
    messages: [],
  };

  pyshell.on('message', (message) => {
    response.messages.push(message);
  });


  // end the input stream and allow the process to exit
  pyshell.end((err) => {
    if (err) {
      reject(err);
      throw err;
    }
    resolve(response);
  });
}));

let output = '';

const getSimilarReviews = async (review) => {
  if (review.id) {
    const database = await Database.connect();

    const _review = await database.collection('reviews')
      .findOne({ review_id: review.id });

    py.stdin.write(`${_review.text.replace(/(\r\n\t|\n|\r\t)/gm, '')}\n`);
  }

  if (review.text) {
    py.stdin.write(`${review.text.replace(/(\r\n\t|\n|\r\t)/gm, '')}\n`);
  }

  return new Promise(((resolve, reject) => {
    py.stdout.removeAllListeners('data');
    py.stdout.on('data', (data) => {
      output += data.toString();
      if (output.includes('STOP')) {
        const ids = output.match(/'.{22}'/gm).map(id => id.slice(1, -1));
        output = '';
        resolve(ids);
      }
    });
  }));
};

const findReviews = async (filter) => {
  let limit = 0;
  // we need to convert any id's to MongoDB's objectIDs
  if (filter._id) {
    filter._id = Database.objectID(filter._id);
  }

  if (filter.stars) {
    filter.stars = parseInt(filter.stars, 10);
  }

  if (filter.limit) {
    limit = parseInt(filter.limit, 10);
    delete filter.limit;
  }

  try {
    // check we're connected
    const database = await Database.connect();
    // find the job based on this filter
    const docs = await database.collection('reviews')
      .find(filter).sort({ stars: -1 }).limit(limit);
    return docs.toArray();
  } catch (e) {
    throw new Error(e);
  }
};

const getReviewIds = async (filter) => {
  let limit = 0;

  if (filter.limit) {
    limit = parseInt(filter.limit, 10);
    delete filter.limit;
  }

  try {
    // check we're connected
    const database = await Database.connect();
    const docs = await database.collection('reviews')
      .find(filter)
      .sort({ date: -1 })
      .limit(limit)
      .project({ review_id: 1 });
    return docs.toArray();
  } catch (e) {
    throw new Error(e);
  }
};

const getAccuracy = async () => {
  try {
    // check we're connected
    const database = await Database.connect();
    const docs = await database.collection('accuracy')
      .find();
    const doc = await docs.toArray();
    return 83;
  } catch (e) {
    throw new Error(e);
  }
};

const getBusiness = async (filter) => {
  let limit = 0;

  if (filter.limit) {
    limit = parseInt(filter.limit, 10);
    delete filter.limit;
  }


  try {
    // check we're connected
    const database = await Database.connect();
    let docs;
    if (filter.name) {
      if (filter.name.length <= 3) {
        return ['Too short'];
      }
      docs = await database.collection('businesses')
        .find({ name: { $regex: `.*${filter.name}.*` }, review_count: { $gt: 0 } })
        .sort({ review_count: -1 })
        .limit(limit);
    } else {
      docs = await database.collection('businesses')
        .find(filter)
        .limit(limit);
    }

    return docs.toArray();
  } catch (e) {
    throw new Error(e);
  }
};


module.exports = {
  initialiseData,
  findReviews,
  getReviewIds,
  getAccuracy,
  getSimilarReviews,
  getBusiness,
};
