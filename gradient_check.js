const mnist = require('./mnist');
const { Network } = require('./network');
const p = require('./meta-parameters');

const n = new Network(p.INPUT_SIZE, p.HIDDEN_LAYERS, p.OUTPUTS);
const NUM_SAMPLES = 5;


n.loadParams().then(() => {
  mnist.open();
  check();
  mnist.close();
});


function check() {
  const calc = calculus_gradient();
  n.reset();
  const num = numeric_gradient();
  const sum = calc.map((cVal, i) => cVal + num[i]);
  const diff = calc.map((cVal, i) => cVal - num[i]);
  // this is a piecewise check to find any individual values significantly different from its counterpart
  const partialErrors = [];
  for (let i = 0; i < calc.length; i++) {
    if (!((Math.abs(calc[i]) - Math.abs(num[i])) <= (Math.abs(calc[i] / 100)))) {
      partialErrors.push({'Index of partial gradient': i, 'Calculated value': calc[i], 'Approximated value': num[i]});
    }
  }
  if (partialErrors.length) {
    console.log('Partial gradients differing by > 1% (possibly indicates a problem not captured by the total difference):');
    console.table(partialErrors);
    console.log();
  }
  console.log('Total difference (should be very small ~1e-8 or less): %d', (norm(diff) / norm(sum)).toExponential(2));
}

function norm(x) {
  return Math.sqrt(x.reduce((total, value) => total + Math.pow(value, 2), 0));
}

function calculus_gradient() {
  for (let i = 0; i < NUM_SAMPLES; i++) {
    const data = mnist.getTrainingImage(i);
    const label = mnist.getTrainingLabel(i);
    const correctOutput = Array(n.outputLayer.length).fill(0);
    correctOutput[label] = 1;
    n.gradient(correctOutput, data);
  }
  const g = n.hiddenLayers.flatMap(layer => {
    return layer.flatMap(node => {
      return node.inputs.flatMap(input => {
        return input.weightGradient;
      }).concat([node.biasGradient]);
    });
  }).concat(n.outputLayer.flatMap(node => {
    return node.inputs.flatMap(input => {
      return input.weightGradient;
    }).concat([node.biasGradient]);
  }));
  return g;
}

function numeric_gradient() {
  const E = 1e-5;
  const doubleE = E * 2;
  const size = p.INPUT_SIZE * p.HIDDEN_LAYERS[0] // input weights
    + p.HIDDEN_LAYERS.reduce((numWeights, numNodes, i, layers) => {
      if (!i) {
        return numWeights;
      }
      return numWeights + numNodes * layers[i - 1];
    }, 0) // hidden layers weights
    + p.HIDDEN_LAYERS.reduce((numBiases, numNodes) => numBiases + numNodes, 0) // hidden layers biases
    + p.HIDDEN_LAYERS.at(-1) * p.OUTPUTS.length // output layer weights
    + p.OUTPUTS.length; // output layer biases
  const g = Array(size).fill(0);
  let correctOutput, data;
  for (let i = 0; i < NUM_SAMPLES; i++) {
    let gIndex = 0;
    data = mnist.getTrainingImage(i);
    const label = mnist.getTrainingLabel(i);
    correctOutput = Array(n.outputLayer.length).fill(0);
    correctOutput[label] = 1;

    // traverse the network and find linear approximations for each partial gradient
    n.hiddenLayers.forEach(layer => {
      layer.forEach(node => {
        node.inputs.forEach(input => {
          g[gIndex++] += approx_dC_dW(input);
        });
        g[gIndex++] += approx_dC_dB(node);
      });
    });
    n.outputLayer.forEach(node => {
      node.inputs.forEach(input => {
        g[gIndex++] += approx_dC_dW(input);
      });
      g[gIndex++] += approx_dC_dB(node);
    });
  }
  return g;

  function approx_dC_dW(input) {
    const initialW = input.weight.value;
    input.weight.value = initialW + E;
    const pCost = n.singleCost(correctOutput, data);
    input.weight.value = initialW - E;
    const mCost = n.singleCost(correctOutput, data);
    input.weight.value = initialW;
    return (pCost - mCost) / doubleE;
  }

  function approx_dC_dB(node) {
    const initialB = node.bias;
    node.bias = initialB + E;
    const pCost = n.singleCost(correctOutput, data);
    node.bias = initialB - E;
    const mCost = n.singleCost(correctOutput, data);
    node.bias = initialB;
    return (pCost - mCost) / doubleE;
  }
}
