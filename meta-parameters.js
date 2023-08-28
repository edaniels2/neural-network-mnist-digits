const INPUT_SIZE = 28 * 28;
const HIDDEN_LAYERS = [24, 16];
const OUTPUTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const TOTAL_TRAINING_SAMPLES = 60000;
const TOTAL_TEST_SAMPLES = 10000;
const TRAIN_ROUNDS = 10; // i think the folks who know what they're doing call these epochs
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

// current success rate on test set for some configurations I've messed with
// 16, 16: 92.49 %
// 18, 16: 91.96 %
// 24, 16: 92.29 %
// 48, 32: 93.18 %
// 24: 90.59 % -- this one surprised me, not much drop in performance when going down to a single layer
