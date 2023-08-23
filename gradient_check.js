const mnist = require('./mnist');
const { Network } = require('./network');
const p = require('./meta-parameters');

const n = new Network(p.INPUT_SIZE, p.NUM_LAYERS, p.NODES_PER_LAYER, p.OUTPUTS);
const NUM_SAMPLES = 5;

n.loadParams().then(check);
// check();

function check() {
  mnist.open();
  const calc = calculus_gradient();
  const num = numeric_gradient();
  const sum = calc.map((cVal, i) => cVal + num[i]);
  const diff = calc.map((cVal, i) => cVal - num[i]);
  // console.log(calc.slice(3000, 3010), num.slice(3000, 3010));
  for (let i = 0; i < calc.length; i++) {
    if ((Math.abs(calc[i]) - Math.abs(num[i])) > (Math.abs(calc[i] / 100))) {
      console.log(i, calc[i], num[i]);
    }
  }
  console.log((norm(diff) / norm(sum)).toExponential(10));
  mnist.close();
}

function norm(x) {
  return Math.sqrt(x.reduce((total, value) => total + Math.pow(value, 2), 0));
}

function calculus_gradient() {
  for (let i = 0; i < NUM_SAMPLES; i++) {
    const data = mnist.getTrainingImage(i);
    const label = mnist.getTrainingLabel(i);
    const correctOutput = Array(10).fill(0);
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
  // const ptb = 1e-5;
  // const ptb2 = ptb * 2;
  const ptb = 3e-7;
  const size = p.INPUT_SIZE * p.NODES_PER_LAYER // input weights
    + Math.pow(p.NODES_PER_LAYER, p.NUM_LAYERS) // hidden layers weights
    + p.NODES_PER_LAYER * p.NUM_LAYERS // hidden layers biases
    + p.NODES_PER_LAYER * p.OUTPUTS.length // output layer weights
    + p.OUTPUTS.length; // output layer biases
  const g = Array(size).fill(0);
  for (let i = 0; i < NUM_SAMPLES; i++) {
    let gIndex = 0;
    const data = mnist.getTrainingImage(i);
    const label = mnist.getTrainingLabel(i);
    const correctOutput = Array(10).fill(0);
    correctOutput[label] = 1;
    const cost = n.singleCost(correctOutput, data);

    // traverse the network, add the perturbation factor to each param and re-evaluate the cost,
    // then take the slope of the two costs as the numeric approximation of the derivitive
    n.hiddenLayers.forEach(layer => {
      layer.forEach(node => {
        node.inputs.forEach(input => {
          input.weight += ptb;
          const pCost = n.singleCost(correctOutput, data);
          // input.weight -= ptb2;
          // const mCost = n.singleCost(correctOutput, data);
          // g[gIndex++] += (pCost - mCost) / ptb2;
          // input.weight += ptb;

          g[gIndex++] += (pCost - cost) / ptb;
          input.weight -= ptb;
        });
        node.bias += ptb;
        const pCost = n.singleCost(correctOutput, data);
        // node.bias -= ptb2;
        // const mCost = n.singleCost(correctOutput, data);
        // g[gIndex++] += (pCost - mCost) / ptb2;
        // node.bias += ptb;

        g[gIndex++] += (pCost - cost) / ptb;
        node.bias -= ptb;
      });
    });
    n.outputLayer.forEach(node => {
      node.inputs.forEach(input => {
        input.weight += ptb;
        const pCost = n.singleCost(correctOutput, data);
        // input.weight -= ptb2;
        // const mCost = n.singleCost(correctOutput, data);
        // g[gIndex++] += (pCost - mCost) / ptb2;
        // input.weight += ptb;

        g[gIndex++] += (pCost - cost) / ptb;
        input.weight -= ptb;
      });
      node.bias += ptb;
      const pCost = n.singleCost(correctOutput, data);
      // node.bias -= ptb2;
      // const mCost = n.singleCost(correctOutput, data);
      // g[gIndex++] += (pCost - mCost) / ptb;
      // node.bias += ptb;

      g[gIndex++] += (pCost - cost) / ptb;
      node.bias -= ptb;
    });
  }
  return g;
}
