const { Network } = require('./network');
const mnist = require('./mnist');
const readline = require('readline');
const sqlite3 = require('sqlite3');

const ROUNDS = 50;
const BATCH_SIZE = 1000;
const EPOCHS = Math.floor(60000 / BATCH_SIZE);
const db = new sqlite3.cached.Database('./training_params.sqlite');
const n = new Network(28*28, 2, 16, [0,1,2,3,4,5,6,7,8,9]);
loadParams().then(train);

function train() {
  let count = 0;
  for (let round = 0; round < ROUNDS; round++) {
    for (let j = 0; j < EPOCHS; j++) {
      for (let i = 0; i < BATCH_SIZE; i++) {
        const s = i + BATCH_SIZE * j;
        const data = mnist.getTrainingImage(s);
        const label = mnist.getTrainingLabel(s);
        const correctOutput = Array(10).fill(0);
        correctOutput[label] = 1;
        n.gradient(correctOutput, data);
        count++;
      }
      const cost = n.currentTotalCost / count;
      console.log(cost);
      n.updateNetwork(cost * 0.01);
      n.currentTotalCost = 0;
      count = 0;
    }
  }

  console.log('Random samples:');
  for (let i = 0; i < 10; i++) {
    const sample = Math.round(Math.random() * 60000);
    const output = n.evaluate(mnist.getTrainingImage(sample));
    console.log('result:', output.result.label);
    console.log('confidence:', output.result.activation);
    console.log('actual: ', mnist.getTrainingLabel(sample));
    // console.log(n.outputLayer.map(node => ({[node.label]: node.activation})));
    console.log();
  }

  const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
  });
  rl.question('Store these params? y/N: ', function (store) {
    if (store.toLowerCase() === 'y') {
      storeNetwork();
    }
    db.close();
    rl.close();
  });


  function storeNetwork() {
    n.hiddenLayers.forEach((layer, layerIndex) => {
      layer.forEach(storeLayer(layerIndex));
    });
    n.outputLayer.forEach(storeLayer(n.hiddenLayers.length));

    // this is pretty ineffecient but I don't know if there's another way to do it using the ON CONFLICT clause
    // doesn't really matter though, training takes ages already
    function storeLayer(layerIndex) {
      return (neuron, neuronIndex) => {
        db.parallelize(() => {
          db.run(`INSERT INTO biases (layer, node, value) VALUES (${layerIndex}, ${neuronIndex}, ${neuron.bias}) ON CONFLICT(layer, node) DO UPDATE SET value=${neuron.bias}`);
          neuron.inputs.forEach((input, inputIndex) => {
            db.run(`INSERT INTO weights (layer, node, input, value) VALUES (${layerIndex}, ${neuronIndex}, ${inputIndex}, ${input.weight}) ON CONFLICT(layer, node, input) DO UPDATE SET value=${input.weight}`);
          });
        });
      };
    }
  }

}
