//node server entry point for the Angular Universal application
import {
  AngularNodeAppEngine,
  createNodeRequestHandler,
  isMainModule,
  writeResponseToNodeResponse,
} from '@angular/ssr/node';
import express from 'express';
import { join } from 'node:path';
import { Readable } from 'node:stream';

const browserDistFolder = join(import.meta.dirname, '../browser');

const app = express();
const angularApp = new AngularNodeAppEngine();

/**
 * Proxy /api/* to the backend (strip the /api prefix), streaming request and
 * response bodies so SSE endpoints work. Target is configurable via
 * API_PROXY_TARGET (defaults to the Docker Compose backend service).
 */
const apiProxyTarget = process.env['API_PROXY_TARGET'] ?? 'http://backend:8000';

app.use('/api', (req, res) => {
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined || key === 'host' || key === 'connection') continue;
    headers.set(key, Array.isArray(value) ? value.join(', ') : value);
  }

  const hasBody = req.method !== 'GET' && req.method !== 'HEAD';
  fetch(`${apiProxyTarget}${req.url}`, {
    method: req.method,
    headers,
    body: hasBody ? (Readable.toWeb(req) as unknown as BodyInit) : undefined,
    // Node fetch requires half-duplex for streamed request bodies
    ...({ duplex: 'half' } as object),
  })
    .then((upstream) => {
      res.status(upstream.status);
      upstream.headers.forEach((value, key) => {
        // fetch already decoded the body — drop hop-by-hop/encoding headers
        if (['content-encoding', 'content-length', 'transfer-encoding', 'connection'].includes(key))
          return;
        res.setHeader(key, value);
      });
      if (upstream.body) {
        Readable.fromWeb(upstream.body as never).pipe(res);
      } else {
        res.end();
      }
    })
    .catch((error) => {
      console.error(`API proxy error for ${req.method} ${req.originalUrl}:`, error);
      if (!res.headersSent) {
        res.status(502).json({ detail: 'API proxy error' });
      } else {
        res.end();
      }
    });
});

/**
 * Serve static files from /browser
 */
app.use(
  express.static(browserDistFolder, {
    maxAge: '1y',
    index: false,
    redirect: false,
  }),
);

/**
 * Handle all other requests by rendering the Angular application.
 */
app.use((req, res, next) => {
  angularApp
    .handle(req)
    .then((response) => (response ? writeResponseToNodeResponse(response, res) : next()))
    .catch(next);
});

/**
 * Start the server if this module is the main entry point, or it is ran via PM2.
 * The server listens on the port defined by the `PORT` environment variable, or defaults to 4000.
 */
if (isMainModule(import.meta.url) || process.env['pm_id']) {
  const port = process.env['PORT'] || 4000;
  app.listen(port, (error) => {
    if (error) {
      throw error;
    }

    console.log(`Node Express server listening on http://localhost:${port}`);
  });
}

/**
 * Request handler used by the Angular CLI (for dev-server and during build) or Firebase Cloud Functions.
 */
export const reqHandler = createNodeRequestHandler(app);
