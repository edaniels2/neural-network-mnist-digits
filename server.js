const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 1234;
const BASE_PATH = process.cwd();
const MIME_TYPES = {
  html: 'text/html; charset=UTF-8',
  js: 'application/javascript; charset=UTF-8',
};

const server = http.createServer(/* options,  */function (req, res) {
  const url = req.url === '/' ? 'index.html' : req.url;
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
