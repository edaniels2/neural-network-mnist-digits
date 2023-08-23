const { Network } = require('./network');
const mnist = require('./mnist');
const p = require('./meta-parameters');

const TEST_COUNT = 10000;

const n = new Network(p.INPUT_SIZE, p.NUM_LAYERS, p.NODES_PER_LAYER, p.OUTPUTS);
mnist.open();
n.loadParams().then(() => {
  test();
  mnist.close();
});

function test() {
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
