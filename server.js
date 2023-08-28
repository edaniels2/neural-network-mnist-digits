const fs = require('fs');
const http = require('http');
const mnist = require('./mnist');
const p = require('./meta-parameters');
const path = require('path');
const { Network } = require('./network');

const PORT = 1234;
const BASE_PATH = process.cwd();
const MIME_TYPES = {
  html: 'text/html; charset=UTF-8',
  js: 'application/javascript; charset=UTF-8',
};


const n = new Network(p.INPUT_SIZE, p.HIDDEN_LAYERS, p.OUTPUTS);
n.loadParams().then(startServer);

function startServer() {
  const server = http.createServer(/* options,  */function (req, res) {
    const url = req.url === '/' ? 'index.html' : req.url;

    let match = url.match(/\/img\/(\d{1,4})/);
    if (match) {
      mnist.open();
      const data = mnist.getTrainingImage(match[1]);
      mnist.close();
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      res.write(data);
      res.end();
      return;
    }

    match = url.match(/\/label\/(\d{1,4})/);
    if (match) {
      mnist.open();
      const data = mnist.getTrainingLabel(match[1]);
      mnist.close();
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.write(String(data));
      res.end();
      return;
    }

    match = url.match(/\/test\/(\d{1,4})/);
    if (match) {
      mnist.open();
      const data = mnist.getTestImage(match[1]);
      const label = mnist.getTestLabel(match[1]);
      mnist.close();
      const output = n.evaluate(data);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.write(JSON.stringify({ output, actual: label }));
      res.end();
      return;
    }

    match = url.match(/\/weights\/(\d{1,2})/);
    if (match) {
      const neuronIndex = parseInt(match[1], 10);
      if (!n.hiddenLayers[0][neuronIndex]) {
        res.writeHead(404);
        res.end();
        return;
      }
      const rawData = n.hiddenLayers[0][neuronIndex].inputs.map(input => input.weight.value);
      let maxMag = 0;
      rawData.forEach(value => {
        maxMag = Math.max(maxMag, Math.abs(value));
      });
      const scale = 255 / maxMag;
      const normalized = [];
      rawData.forEach(value => {
        // each value will push 2 values to the data, representing positive and negative activation scaled to 255
        const red = value > 0 ? Math.round(value * scale) : 0;
        const blue = value < 0 ? Math.round(Math.abs(value) * scale) : 0;
        normalized.push(red);
        normalized.push(blue);
      });
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      res.write(Buffer.from(normalized));
      res.end();
      return;
    }

    if (req.method === 'POST' && url === '/evaluate') {
      req.on('data', (/** @type Buffer */data) => {
        const output = n.evaluate(data);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.write(JSON.stringify(output));
        res.end();
      });
      return;
    }

    if (url === '/check') {
      res.statusCode = 200;
      res.end(JSON.stringify({success: true}));
      return;
    }
    let filepath = path.join(BASE_PATH, url);
    let status = 200;
    fs.promises.access(filepath).catch(
      function notFound() {
        filepath = path.join(BASE_PATH, '404.html');
        status = 404;
      }
    ).finally(function send() {
      const type = MIME_TYPES[path.extname(filepath).substring(1).toLowerCase()] || 'text/plain';
      res.writeHead(status, { 'Content-Type': type });
      const stream = fs.createReadStream(filepath);
      stream.pipe(res);
      console.log(req.httpVersion, req.method, req.url, status);
    });
  })
  server.listen(PORT);
}