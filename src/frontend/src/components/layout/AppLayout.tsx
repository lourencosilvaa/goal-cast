import { Outlet, useLocation } from 'react-router-dom';
import { AnimatePresence, motion } from 'framer-motion';
import { AuroraBackground } from '@/components/ui/AuroraBackground';
import { RetrainingBanner } from '@/components/RetrainingBanner';
import { Sidebar } from './Sidebar';
import { MobileNav } from './MobileNav';

export function AppLayout() {
  const location = useLocation();

  return (
    <div className="h-screen overflow-hidden bg-bg text-fg flex">
      <AuroraBackground />
      <Sidebar />
      <main className="flex-1 overflow-y-auto scrollbar-hide relative z-10">
        <RetrainingBanner />
        <div className="p-4 md:p-6 pb-24 md:pb-6 max-w-[1600px] mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
      <MobileNav />
    </div>
  );
}
