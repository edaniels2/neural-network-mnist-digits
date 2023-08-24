const sqlite3 = require('sqlite3');
const p = require('./meta-parameters');

const db = new sqlite3.Database(p.DB_FILE);
db.serialize(() => {
  db.run('CREATE TABLE weights (id INTEGER PRIMARY KEY, layer INTEGER, node INTEGER, input INTEGER, value FLOAT)');
  db.run('CREATE TABLE biases (id INTEGER PRIMARY KEY, layer INTEGER, node INTEGER, value FLOAT)');
  db.run('CREATE UNIQUE INDEX weight_position ON weights(layer, node, input)');
  db.run('CREATE UNIQUE INDEX bias_position ON biases(layer, node)');
});
db.close();
