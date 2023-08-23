const fs = require('fs');

const imageSize = 28 * 28;
const imageHeaderOffset = 16;
const labelHeaderOffset = 8;

/** @type number */let trainingImagesFile;
/** @type number */let trainingLabelsFile;
/** @type number */let testImagesFile;
/** @type number */let testLabelsFile;
const readBuffer = new Uint8Array(imageSize);

function open() {
  trainingImagesFile = fs.openSync('./mnist_dataset/train-images-idx3-ubyte');
  trainingLabelsFile = fs.openSync('./mnist_dataset/train-labels-idx1-ubyte');
  testImagesFile = fs.openSync('./mnist_dataset/t10k-images-idx3-ubyte');
  testLabelsFile = fs.openSync('./mnist_dataset/t10k-labels-idx1-ubyte');
}

function close() {
  fs.close(trainingImagesFile);
  fs.close(trainingLabelsFile);
  fs.close(testImagesFile);
  fs.close(testLabelsFile);
}

function clone(/** @type Uint8Array */src)  {
  var dst = new ArrayBuffer(src.length);
  new Uint8Array(dst).set(new Uint8Array(src));
  return dst;
}

function getTrainingImage(/** @type number */i) {
  fs.readSync(trainingImagesFile, readBuffer, 0, imageSize, imageHeaderOffset + i * imageSize);
  return new Uint8Array(readBuffer);
}

function getTrainingLabel(/** @type number */i) {
  fs.readSync(trainingLabelsFile, readBuffer, 0, 1, labelHeaderOffset + Number(i));
  return readBuffer.at(0);
}

function getTestImage(/** @type number */i) {
  fs.readSync(testImagesFile, readBuffer, 0, imageSize, imageHeaderOffset + i * imageSize);
  return new Uint8Array(readBuffer);
}

function getTestLabel(/** @type number */i) {
  fs.readSync(testLabelsFile, readBuffer, 0, 1, labelHeaderOffset + Number(i));
  return readBuffer.at(0);
}

module.exports = { getTrainingImage, getTrainingLabel, getTestImage, getTestLabel, open, close };
