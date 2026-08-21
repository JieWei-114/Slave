import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    pathMatch: 'full',
    redirectTo: 'chat',
  },
  {
    path: 'chat',
    loadChildren: () => import('./features/chat/chat.routes').then((m) => m.routes),
  },
  {
    path: 'models',
    loadChildren: () => import('./features/models/models.routes').then((m) => m.routes),
  },
  {
    path: '**',
    redirectTo: 'chat',
  },
];
