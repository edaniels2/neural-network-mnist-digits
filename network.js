const ACTIVATION_FN = sigmoid;
const sqlite3 = require('sqlite3');

function activationFn(value) {
  return ACTIVATION_FN(value);
}

function activationFnDerivitive(value) {
  return ACTIVATION_FN.d(value);
}

// haven't tested this one but it's supposed to be more effective for training
function ReLU(value) {
  return Math.max(0, value);
}
ReLU.d = function ReLU_d(value) {
  return value > 0 ? 1 : 0;
};

function sigmoid(value) {
  return 1 / (1 + Math.pow(Math.E, -value));
}
sigmoid.d = function sigmoid_d(value) {
  const f = sigmoid(value);
  return f * (1 - f);
}

class InputNeuron {
  activation = 0;
  isInput = true;
  /** @type Array<{neuron: Neuron, weight: number}> */fwdConnections = [];
}

class Neuron {
  activation = 0;
  bias = 0;
  z = 0;
  biasGradient = 0;
  dCost_dOut = 0;
  dOut_dZ = 0;
  /** @type Array<{neuron: Neuron, weight: number}> */fwdConnections = [];

  constructor(/** @type Array<Neuron|InputNeuron> */previousLayer) {
    /** @type Array<{neuron: Neuron, weight: number, weightGradient: number}> */
    this.inputs = previousLayer.map(neuron => {
      const weight = Math.random() * 2 - 1;
      neuron.fwdConnections.push({ neuron: this, weight });
      return { neuron, weight, weightGradient: 0 };
    });
  }

  toString() {
    return JSON.stringify((({fwdConnections, ...rest}) => rest)(this));
  }

  updateWeights(/** @type Array<number> */weights) {
    if (weights.length !== this.inputs.length) {
      throw new Error('Number of weights does not match previous layer size');
    }
    this.inputs.forEach((connection, i) => {
      connection.weight[i] = weights[i];
    });
  }

  updateActivation() {
    const value = this.inputs.reduce((sum, input) => {
      sum += input.neuron.activation * input.weight;
      return sum;
    }, 0) + this.bias;
    this.z = value;
    this.activation = activationFn(value);
  }

}

class OutputNeuron extends Neuron {

  constructor(
    /** @type Array<Neuron> */previousLayer,
    /** @type string|number */label
  ) {
    super(previousLayer);
    /** @type string */this.label = String(label);
  }
}

class Network {
  currentTotalCost = 0;

  constructor(
    /** @type number */numInputs,
    /** @type number */numHiddenLayers,
    /** @type number */neuronsPerLayer,
    /** @type Array<string | number> */outputLabels,
  ) {
    /** @type Array<InputNeuron> */
    this.inputLayer = Array(numInputs).fill(null).map(_k => new InputNeuron);
    /** @type Array<Array<Neuron>> */
    this.hiddenLayers = Array(numHiddenLayers).fill(null);
    for (let i = 0; i < numHiddenLayers; i++) {
      const previousLayer = i ? this.hiddenLayers[i - 1] : this.inputLayer;
      this.hiddenLayers[i] = Array(neuronsPerLayer).fill(null).map(_k => new Neuron(previousLayer));
    }
    /** @type Array<OutputNeuron> */
    this.outputLayer = Array(outputLabels.length).fill(null).map((_k, i) => new OutputNeuron(this.hiddenLayers.at(-1), outputLabels[i]));
  }

  loadParams() {
    /** @type sqlite3.Database */
    const db = new sqlite3.cached.Database('./training_params.sqlite');
    // sqlite3 hijacks the `this` binding in the callback
    const self = this;
    return Promise.all([
      new Promise(resolve => db.all('SELECT * FROM weights', function (err, results) {
        err && console.log(err)
        results.forEach(row => {
          const layer = row.layer >= self.hiddenLayers.length ? self.outputLayer : self.hiddenLayers[row.layer];
          layer[row.node].inputs[row.input].weight = row.value;
        });
        console.log('weights loaded');
        resolve();
      })),
      new Promise(resolve => db.all('SELECT * FROM biases', function (err, results) {
        err && console.log(err)
        results.forEach(row => {
          const layer = row.layer >= self.hiddenLayers.length ? self.outputLayer : self.hiddenLayers[row.layer];
          layer[row.node].bias = row.value;
        });
        console.log('biases loaded');
        resolve();
      }))
    ]).then(() => db.close());
  }

  evaluate(/** @type Array<number> */data, /** @type Array<number> */expected) {
    if (data.length !== this.inputLayer.length) {
      throw new Error('Data length does not match input layer size');
    }
    // set input activations - these will be in the range 0 - 255; normalized to 0 - 1 to match the sigmoid activation fn range
    this.inputLayer.forEach((neuron, i) => neuron.activation = data[i] / 255);
    // feed forward
    this.hiddenLayers.forEach(layer => {
      layer.forEach(neuron => neuron.updateActivation());
    });
    this.outputLayer.forEach(neuron => neuron.updateActivation());
    // find the most active output
    const result = this.outputLayer.reduce((maxActive, neuron) => {
      if (!maxActive || neuron.activation > maxActive.activation) {
        maxActive = neuron;
      }
      return maxActive;
    });
    this.currentTotalCost += expected ? this.cost(expected) : null;

    return { result: {label: result.label, activation: result.activation}, fullOutput: this.outputLayer.map(n => n.activation) };
  }

  cost(/** @type Array<number> */expected, /** @optional @type Array<number> */input) {
    if (input) {
      this.evaluate(input);
    }
    return this.outputLayer.reduce((/** @type number */totalCost, output, i) => {
      totalCost += Math.pow(output.activation - expected[i], 2);
      return totalCost;
    }, 0) / expected.length;
  }

  gradient(
    /** @type Array<number> */correct,
    /** @type Array<number> */input,
  ) {
    if (input) {
      this.evaluate(input, correct);
    }
    this.outputLayer.forEach((node, i) => {
      node.dCost_dOut = 2 * (node.activation - correct[i]);
      node.dOut_dZ = activationFnDerivitive(node.z);
      node.biasGradient += node.dCost_dOut * node.dOut_dZ;

      node.inputs.forEach(input => {
        input.weightGradient += input.neuron.activation * node.dCost_dOut * node.dOut_dZ;
      });
    });

    for (let layerI = this.hiddenLayers.length-1; layerI >= 0; layerI--) {
      const layer = this.hiddenLayers[layerI];
      layer.forEach(node => {
        node.dCost_dOut = 0;
        node.fwdConnections.forEach(fwdConnection => {
          node.dCost_dOut += fwdConnection.neuron.dCost_dOut * fwdConnection.neuron.dOut_dZ * fwdConnection.weight;
        });
        node.dOut_dZ = activationFnDerivitive(node.z);
        const partialBiasGradient = node.dCost_dOut * node.dOut_dZ;
        node.biasGradient += partialBiasGradient;
        node.inputs.forEach(input => {
          input.weightGradient += input.neuron.activation * partialBiasGradient;
        });
      });
    }
  }

  updateNetwork(cost = 1) {
    this.hiddenLayers.forEach(layer => {
      layer.forEach(updateNeuron);
    });
    this.outputLayer.forEach(updateNeuron);

    function updateNeuron(/** @type Neuron */neuron) {
      neuron.bias -= cost * neuron.biasGradient;
      neuron.biasGradient = 0;
      neuron.inputs.forEach(input => {
        input.weight -= cost * input.weightGradient;
        input.weightGradient = 0;
      });
    }
  }
}

module.exports = { Network };
