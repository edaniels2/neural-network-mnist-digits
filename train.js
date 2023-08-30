const createTables = require('./db_init');
const mnist = require('./mnist');
const Network = require('./network');
const p = require('./meta-parameters');
const readline = require('readline');
const sqlite3 = require('sqlite3');
const test = require('./test');

const BATCHES = Math.floor(p.TOTAL_TRAINING_SAMPLES / p.TRAIN_BATCH_SIZE);
const n = new Network(p.INPUT_SIZE, p.HIDDEN_LAYERS, p.OUTPUTS);

n.loadParams().then(() => {
  mnist.open();
  train();
  mnist.close();
});

function train() {
  for (let round = 0; round < p.TRAIN_ROUNDS; round++) {
    const shuffledIndexes = shuffle(Array.from(Array(p.TOTAL_TRAINING_SAMPLES).keys()));
    for (let j = 0; j < BATCHES; j++) {
      for (let i = 0; i < p.TRAIN_BATCH_SIZE; i++) {
        const s = shuffledIndexes[i + p.TRAIN_BATCH_SIZE * j];
        const data = mnist.getTrainingImage(s);
        const label = mnist.getTrainingLabel(s);
        const correctOutput = Array(n.outputLayer.length).fill(0);
        correctOutput[label] = 1;
        n.gradient(correctOutput, data);
      }
      n.updateParams();
    }
  }

  test(n);

  const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
  });
  rl.question('Store these params? y/N: ', function (store) {
    // TODO: add an option to continue training
    if (store.toLowerCase() === 'y') {
      save();
    }
    rl.close();
  });


  function save() {
    const flags = sqlite3.OPEN_FULLMUTEX | sqlite3.OPEN_READWRITE;
    const database = new sqlite3.Database(p.DB_FILE, flags, err => {
      if (err) {
        createTables().then(storeParams);
      } else {
        storeParams(database);
      }
    });

    function storeParams(/** @type sqlite3.Database */db) {
      n.hiddenLayers.forEach((layer, layerIndex) => {
        layer.forEach((neuron, neuronIndex) => storeNeuron(layerIndex, neuronIndex, neuron));
      });
      n.outputLayer.forEach((neuron, neuronIndex) => storeNeuron(n.hiddenLayers.length, neuronIndex, neuron));
      db.close();

      // this is pretty ineffecient but I don't know if there's another way to do it using the ON CONFLICT clause
      // doesn't really matter though, training takes ages already
      function storeNeuron(layerIndex, neuronIndex, neuron) {
  
        db.parallelize(() => {
          db.run(`INSERT INTO biases (layer, node, value) VALUES (${layerIndex}, ${neuronIndex}, ${neuron.bias}) ON CONFLICT(layer, node) DO UPDATE SET value=${neuron.bias}`);
          neuron.inputs.forEach((input, inputIndex) => {
            db.run(`INSERT INTO weights (layer, node, input, value) VALUES (${layerIndex}, ${neuronIndex}, ${inputIndex}, ${input.weight.value}) ON CONFLICT(layer, node, input) DO UPDATE SET value=${input.weight.value}`);
          });
        });
      }
    }

  }

}

// https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
// I haven't tested this personally but it's probably good
function shuffle(/** @type Array<any> */array) {
  for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}
