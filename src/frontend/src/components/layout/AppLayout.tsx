import { Outlet, useLocation } from 'react-router-dom';
import { RetrainingBanner } from '@/components/RetrainingBanner';
import { titleForPath } from '@/config/nav';
import { Sidebar } from './Sidebar';
import { MobileNav } from './MobileNav';

/**
 * The application shell: a fixed-height row of nav rail + main column.
 *
 * The shell itself never scrolls. Each page owns its own scrolling region,
 * which is what lets the dashboard keep its ticker, hero and filter bar pinned
 * while only the match list moves.
 */
export function AppLayout() {
  const location = useLocation();

  return (
    <div className="h-screen overflow-hidden bg-bg text-fg flex">
      <Sidebar />

      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        <RetrainingBanner />

        <header className="flex items-center gap-3 px-5 md:px-7 py-4 border-b border-line shrink-0">
          <h1 className="text-xl font-extrabold tracking-[-0.01em] text-fg">
            {titleForPath(location.pathname)}
          </h1>
        </header>

        {/* pb on mobile clears the fixed bottom nav. */}
        <div className="flex-1 flex flex-col overflow-hidden pb-20 md:pb-0">
          <Outlet />
        </div>
      </div>

      <MobileNav />
    </div>
  );
}
