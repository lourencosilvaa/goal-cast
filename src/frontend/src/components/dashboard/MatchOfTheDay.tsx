import { outcomeShorthand, type HeroMatch } from './derive';

/**
 * The banner above the fixture list.
 *
 * It shows the 1X2 shorthand rather than a scoreline, because the API's
 * headline prediction is an outcome — the most likely scoreline is a separate,
 * much weaker signal that lives in the match detail.
 */
export function MatchOfTheDay({ hero, onOpen }: { hero: HeroMatch; onOpen: () => void }) {
  const { match } = hero;

  return (
    <button
      type="button"
      onClick={onOpen}
      className="relative flex items-center gap-5 md:gap-7 w-full text-left px-5 md:px-7 py-5 border-b border-line shrink-0 overflow-hidden cursor-pointer bg-card hover:bg-card-2 transition-colors"
    >
      <span className="absolute top-0 right-0 h-full w-[180px] bg-accent/12 [clip-path:polygon(60%_0,100%_0,100%_100%,20%_100%)] pointer-events-none" />

      <span className="font-mono text-[11px] font-bold tracking-[0.1em] text-accent shrink-0 leading-tight">
        JOGO
        <br />
        DO
        <br />
        DIA
      </span>

      <span className="flex items-center gap-3 md:gap-4 flex-1 min-w-0 z-10">
        <span className="text-lg md:text-2xl font-extrabold text-fg truncate">
          {match.home_team}
        </span>
        <span className="font-mono text-2xl md:text-4xl font-extrabold text-accent shrink-0">
          {outcomeShorthand(match)}
        </span>
        <span className="text-lg md:text-2xl font-extrabold text-fg truncate">
          {match.away_team}
        </span>
      </span>

      <span className="flex flex-col items-end gap-0.5 shrink-0 z-10">
        <span className="font-mono text-xl md:text-[26px] font-bold text-fg">
          {(match.confidence * 100).toFixed(0)}%
        </span>
        <span className="label-mono text-fg-subtle">Confiança</span>
      </span>
    </button>
  );
}
