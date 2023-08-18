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
}

class Neuron {
  activation = 0;
  bias = 0;
  z = 0;

  constructor(/** @type Array<Neuron|InputNeuron> */previousLayer) {
    /** @type Array<{neuron: Neuron, weight: number}> */this.inputs = previousLayer.map(neuron => {
      const weight = 1//neuron.isInput ? 1 : (Math.random() - 0.5) * 20;
      return { neuron, weight };
    });
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

    return result.label;
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
    /** @optional @type Array<Array<{biasDerivitive: number, weightDerivitives: Array<number>}>> */runningTally,
    /** @optional @type Array<number> */input,
  ) {
    const resultsPerLayer = runningTally || Array(this.hiddenLayers.length + 1).fill(null).map(_k => []);
    // figure out how to work in the sum for running tally (pushing the new results on the end is wrong)
    resultsPerLayer.push(this.outputLayer.map((node, i) => {
      const dAdZ = activationFnPrime(node.z);
      const dCdA = 2 * (node.activation - expected[i]);
      console.log(dAdZ, dCdA);
      const biasDerivitive = dAdZ * dCdA;
      const weightDerivitives = node.inputs.map((previousNode, i) => {
        const dZdW = previousNode.neuron.activation;
        console.log(dZdW);
        console.log(activationFnPrime(previousNode.neuron.z) * previousNode.neuron.inputs[0].neuron.activation)
        return dZdW * biasDerivitive;
      });
      console.log({biasDerivitive, weightDerivitives})
      return { biasDerivitive, weightDerivitives };
    }));

    return resultsPerLayer;
  }
}

// const n = new Network(28*28, 2, 16, [0,1,2,3,4,5,6,7,8,9]);
const n = new Network(2, 1, 2, ['o']);
n.evaluate([-2, -1]);
console.log(n.gradient([1]));
