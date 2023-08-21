const { Network } = require('./network');
const mnist = require('./mnist');

const TEST_COUNT = 10000;

const n = new Network(28*28, 2, 16, [0,1,2,3,4,5,6,7,8,9]);
n.loadParams().then(test);

function test() {
  let correctCount = 0;
  for (let i = 0; i < TEST_COUNT; i++) {
    const data = mnist.getTestImage(i);
    const label = mnist.getTestLabel(i);
    const expected = Array(n.outputLayer.length).fill(0);
    expected[label] = 1;
    const output = n.evaluate(data, expected);
    if (Number(output.result.label) === label) {
      correctCount++;
    }
  }
  console.log('Evaluated %d samples, %d correct', TEST_COUNT, correctCount);
  console.log((correctCount / TEST_COUNT * 100).toFixed(2), '%');
}
