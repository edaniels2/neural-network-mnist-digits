const fs = require('fs');

const imageSize = 28 * 28;
const imageHeaderOffset = 16;
const labelHeaderOffset = 8;

function getImage(/** @type number */i) {
  const trainingImagesFile = fs.openSync('./mnist_dataset/train-images-idx3-ubyte');
  const readBuffer = new Uint8Array(imageSize);
  fs.readSync(trainingImagesFile, readBuffer, 0, imageSize, imageHeaderOffset + i * imageSize);
  fs.close(trainingImagesFile);
  return readBuffer;
}

function getLabel(/** @type number */i) {
  const trainingLabelsFile = fs.openSync('./mnist_dataset/train-labels-idx1-ubyte');
  const readBuffer = new Uint8Array(1);
  fs.readSync(trainingLabelsFile, readBuffer, 0, 1, labelHeaderOffset + Number(i));
  fs.close(trainingLabelsFile);
  return String(readBuffer.at(0));
}

module.exports = { getImage, getLabel };
