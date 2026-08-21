import { HttpInterceptorFn, HttpResponse } from '@angular/common/http';
import { isDevMode } from '@angular/core';
import { tap } from 'rxjs';

/**
 * Dev-only HTTP logging.
 * Logs method + URL + response status. Never logs bodies or headers.
 */
export const loggingInterceptor: HttpInterceptorFn = (req, next) => {
  if (!isDevMode()) {
    return next(req);
  }

  return next(req).pipe(
    tap({
      next: (event) => {
        if (event instanceof HttpResponse) {
          console.log('[HTTP]', req.method, req.url, event.status);
        }
      },
      error: (err: unknown) => {
        const status = (err as { status?: number })?.status ?? '?';
        console.log('[HTTP ERROR]', req.method, req.url, status);
      },
    }),
  );
};
