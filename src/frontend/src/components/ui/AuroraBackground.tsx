export function AuroraBackground() {
  return (
    <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
      <div className="grid-overlay absolute inset-0" />
      <div
        className="aurora-blob absolute w-[600px] h-[600px] bg-green-600/20"
        style={{ top: '-10%', left: '-5%' }}
      />
      <div
        className="aurora-blob absolute w-[500px] h-[500px] bg-emerald-500/15"
        style={{ top: '30%', right: '-10%', animationDelay: '4s' }}
      />
      <div
        className="aurora-blob absolute w-[400px] h-[400px] bg-green-400/10"
        style={{ bottom: '-5%', left: '30%', animationDelay: '8s' }}
      />
    </div>
  );
}
