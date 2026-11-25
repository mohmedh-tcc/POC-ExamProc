import { createRootRoute, createRoute, createRouter, Outlet } from '@tanstack/react-router';
import { DiagnosticPlayground } from './routes/index';

const rootRoute = createRootRoute({
  component: () => (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-sky-500/30 selection:text-sky-200">
      <Outlet />
    </div>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: DiagnosticPlayground,
});

const routeTree = rootRoute.addChildren([indexRoute]);

export const router = createRouter({ routeTree });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
