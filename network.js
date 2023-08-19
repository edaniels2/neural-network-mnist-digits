const ACTIVATION_FN = sigmoid;

function activationFn(value) {
  return ACTIVATION_FN(value);
}

function activationFnPrime(value) {
  return ACTIVATION_FN.d(value);
}

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
    /** @type Array<{neuron: Neuron, weight: number, weightGradient: number, dZ_dW: number}> */
    this.inputs = previousLayer.map(neuron => {
      const weight = 1//neuron.isInput ? 1 : (Math.random() - 0.5) * 20;
      neuron.fwdConnections.push({ neuron: this, weight });
      return { neuron, weight, weightGradient: 0, dZ_dW: 0 };
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
    let layerIndex = 0;
    for (let i = 0; i < numHiddenLayers; i++) {
      const previousLayer = layerIndex ? this.hiddenLayers[i - 1] : this.inputLayer;
      this.hiddenLayers[i] = Array(neuronsPerLayer).fill(null).map(_k => new Neuron(previousLayer));
    }
    /** @type Array<OutputNeuron> */
    this.outputLayer = Array(outputLabels.length).fill(null).map((_k, i) => new OutputNeuron(this.hiddenLayers.at(-1), outputLabels[i]));
  }

  evaluate(/** @type Array<number> */data) {
    if (data.length !== this.inputLayer.length) {
      throw new Error('Data length does not match input layer size');
    }
    // set input activations - these will be in the range 0 - 255; i guess that needs to be normalized too but as long as we're using ReLU it's a freebie
    this.inputLayer.forEach((neuron, i) => neuron.activation = data[i]);
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

    return result;
  }

  cost(/** @type Array<number> */expected, /** @optional @type Array<number> */input) {
    if (input) {
      this.evaluate(input);
    }
    return this.outputLayer.reduce((/** @type number */totalCost, output, i) => {
      totalCost += Math.pow(output.activation - expected[i], 2);
      return totalCost;
    }, 0);
  }

  gradient(
    /** @type Array<number> */expected,
    /** @optional @type Array<number> */input,
  ) {
    if (input) {
      this.evaluate(input);
    }
    this.outputLayer.forEach((node, i) => {
      node.dCost_dOut = 2 * (node.activation - expected[i]);
      node.dOut_dZ = activationFnPrime(node.z);
      node.biasGradient = node.dCost_dOut * node.dOut_dZ;

      node.inputs.forEach(input => {
        input.weightGradient = input.neuron.activation * node.biasGradient;
      });
    });

    for (let layerI = this.hiddenLayers.length-1; layerI >= 0; layerI--) {
      const layer = this.hiddenLayers[layerI];
      layer.forEach(node => {
        node.dOut_dZ = activationFnPrime(node.z);
        let totalBiasGradient = 0;
        node.fwdConnections.forEach(fwdConnection => {
          node.dCost_dOut = fwdConnection.weight * fwdConnection.neuron.dOut_dZ;
          const partialBiasGradient = node.dCost_dOut * node.dOut_dZ;
          totalBiasGradient += partialBiasGradient;
          node.inputs.forEach(input => {
            input.weightGradient = input.neuron.activation * partialBiasGradient * fwdConnection.neuron.dCost_dOut;
          });
        })
        node.biasGradient = totalBiasGradient;
      });
    }
  }

  updateNetwork() {
    this.hiddenLayers.forEach(layer => {
      layer.forEach(updateNeuron);
    });
    this.outputLayer.forEach(updateNeuron)

    function updateNeuron(/** @type Neuron */neuron) {
      neuron.bias -= neuron.biasGradient;
      neuron.inputs.forEach(input => input.weight -= input.weightGradient);
    }
  }
}

// const n = new Network(28*28, 2, 16, [0,1,2,3,4,5,6,7,8,9]);
const n = new Network(2, 2, 4, ['f', 'm']);
for (i = 0; i < 500; i++) {
  n.gradient([1, 0], [-2, -1]);
  n.updateNetwork();
  n.gradient([0, 1], [25, 6]);
  n.updateNetwork();
  n.gradient([0, 1], [17, 4]);
  n.updateNetwork();
  n.gradient([1, 0], [-15, -6]);
  n.updateNetwork();
}
const result = n.evaluate([-7, -3]);
console.log(result.label, result.activation)

// console.log(
//   JSON.stringify(n, (k, v) => k === 'fwdConnections' ? undefined : v, 2)
// );
