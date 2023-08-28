const INPUT_SIZE = 28 * 28;
const HIDDEN_LAYERS = [16, 16];
const OUTPUTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const TOTAL_TRAINING_SAMPLES = 60000;
const TOTAL_TEST_SAMPLES = 10000;
const TRAIN_ROUNDS = 200;
const TRAIN_BATCH_SIZE = 1000;
const LEARN_RATE = 0.001;
const DB_FILE = `./training_params_${HIDDEN_LAYERS.join('_')}.sqlite`;

module.exports = {
  INPUT_SIZE,
  HIDDEN_LAYERS,
  OUTPUTS,
  TOTAL_TRAINING_SAMPLES,
  TOTAL_TEST_SAMPLES,
  TRAIN_ROUNDS,
  TRAIN_BATCH_SIZE,
  LEARN_RATE,
  DB_FILE,
};

// current success rate on test set for each configuration
// 16x16: 92.49 %
// 18x16: 86.98 % (needs retraining after bug fix)
// 24x16: 86.69 % (needs retraining after bug fix)
// 48x32: 88.28 % (needs retraining after bug fix)
