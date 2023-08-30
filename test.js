const mnist = require('./mnist');
const Network = require('./network');
const p = require('./meta-parameters');

const TEST_COUNT = 10000;

if (require.main === module) { // run from cli
  const n = new Network(p.INPUT_SIZE, p.HIDDEN_LAYERS, p.OUTPUTS);
  n.loadParams().then(() => {
    mnist.open();
    test(n);
    mnist.close();
  });
}

function test(/** @type Network */n) {
  let correctCount = 0;
  for (let i = 0; i < TEST_COUNT; i++) {
    const data = mnist.getTestImage(i);
    const label = mnist.getTestLabel(i);
    const output = n.evaluate(data);
    if (Number(output.result.label) === label) {
      correctCount++;
    }
  }
  console.log('Evaluated %d samples, %d correct', TEST_COUNT, correctCount);
  console.log((correctCount / TEST_COUNT * 100).toFixed(2), '%');
}

module.exports = test;
