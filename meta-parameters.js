const INPUT_SIZE = 28 * 28;
const HIDDEN_LAYERS = [16, 16];
const OUTPUTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const TRAIN_ROUNDS = 200;
const TRAIN_BATCH_SIZE = 1000;
const LEARN_RATE = 0.001;
const DB_FILE = `./training_params_${HIDDEN_LAYERS.join('_')}.sqlite`;

module.exports = {
  INPUT_SIZE,
  HIDDEN_LAYERS,
  OUTPUTS,
  TRAIN_ROUNDS,
  TRAIN_BATCH_SIZE,
  LEARN_RATE,
  DB_FILE,
};

// current success rate on test set for each configuration
// 16x16: 87.15 %
// 18x16: 86.98 %
// 24x16: 86.69 %
// 48x32: 88.28 %
