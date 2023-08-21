const http = require('http');
const fs = require('fs');
const path = require('path');
const mnist = require('./mnist');
const { Network } = require('./network');

const PORT = 1234;
const BASE_PATH = process.cwd();
const MIME_TYPES = {
  html: 'text/html; charset=UTF-8',
  js: 'application/javascript; charset=UTF-8',
};

const n = new Network(28*28, 2, 16, [0,1,2,3,4,5,6,7,8,9]);
n.loadParams().then(startServer);

function startServer() {
  const server = http.createServer(/* options,  */function (req, res) {
    const url = req.url === '/' ? 'index.html' : req.url;

    let match = url.match(/\/img\/(\d{1,4})/);
    if (match) {
      const data = mnist.getTrainingImage(match[1]);
      res.writeHead(200, { 'Content-Type': 'application/octet-stream' });
      res.write(data);
      res.end();
      return;
    }

    match = url.match(/\/label\/(\d{1,4})/);
    if (match) {
      const data = mnist.getTrainingLabel(match[1]);
      res.writeHead(200, { 'Content-Type': 'text/plain' });
      res.write(String(data));
      res.end();
      return;
    }

    match = url.match(/\/test\/(\d{1,4})/);
    if (match) {
      const data = mnist.getTestImage(match[1]);
      const label = mnist.getTestLabel(match[1]);
      const output = n.evaluate(data);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.write(JSON.stringify({ output, actual: label }));
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