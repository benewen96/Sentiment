const mongo = require('mongodb');

const MongoClient = mongo.MongoClient;

// Connection URL
const url = process.env.MONGODB;

// variables that will store our connection to our db for reuse
let client;
let database;

/**
 * called by every database method, handles connection to database
 * @return {Promise} [returns on successful connection]
 */
const connect = async () => {
  // if we haven't ever connected...
  if (database === undefined) {
    try {
      // connect!
      client = await MongoClient.connect(url);
      database = client.db('sentiment');
      return database;
    } catch (e) {
      throw new Error(e);
    }
  } else {
    return database;
  }
};

module.exports = {
  connect,
  objectID: mongo.ObjectID,
};
