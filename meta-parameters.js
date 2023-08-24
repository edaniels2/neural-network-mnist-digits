const INPUT_SIZE = 28 * 28;
const HIDDEN_LAYERS = [24, 16];
const OUTPUTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

const TRAIN_ROUNDS = 100;
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
