"""
Gerar relatório HTML de análise de jogos e apostas de valor.

Produz uma página HTML estilizada com:
- Cards de jogos com previsões e estatísticas
- Destaques de apostas de valor
- Análise Over/Under e Ambas Marcam
- Previsão de resultados
- Relatório textual NLP com análise detalhada
"""

from datetime import datetime, timezone
from pathlib import Path

from src.analysis.match_stats import MatchStats
from src.analysis.value_detector import ValueBet
from src.models.predictor import MatchPrediction
from src.scrapers.fixtures_fetcher import Fixture


def _pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def _confidence_color(confidence: str) -> str:
    return {"HIGH": "#22c55e", "MEDIUM": "#eab308", "LOW": "#ef4444"}.get(
        confidence, "#94a3b8"
    )


def _confidence_pt(confidence: str) -> str:
    return {"HIGH": "ALTA", "MEDIUM": "MÉDIA", "LOW": "BAIXA"}.get(
        confidence, confidence
    )


def _outcome_pt(outcome: str) -> str:
    return {
        "Home Win": "Vitória da Casa",
        "Draw": "Empate",
        "Away Win": "Vitória Fora",
    }.get(outcome, outcome)


def _form_label_pt(form_pts: float) -> str:
    if form_pts >= 2.0:
        return "Boa"
    elif form_pts >= 1.5:
        return "Média"
    return "Fraca"


def _prob_bar(label: str, prob: float, color: str) -> str:
    width = max(2, prob * 100)
    return f"""
    <div class="prob-row">
      <span class="prob-label">{label}</span>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill" style="width:{width:.0f}%;background:{color}"></div>
      </div>
      <span class="prob-value">{_pct(prob)}</span>
    </div>"""


def _scoreline_badges(scorelines: list[tuple[int, int, float]]) -> str:
    badges = ""
    for h, a, p in scorelines[:5]:
        badges += f'<span class="score-badge">{h}-{a} <small>{_pct(p)}</small></span>'
    return badges


def _form_dots(form_pts: float, team: str) -> str:
    if form_pts >= 2.0:
        color = "#22c55e"
    elif form_pts >= 1.5:
        color = "#eab308"
    else:
        color = "#ef4444"
    label = _form_label_pt(form_pts)
    return f'<span class="form-indicator" style="background:{color}">{form_pts:.1f} ppj</span> <small class="form-label">{label}</small>'


# ── NLP Textual Analysis ─────────────────────────────────────────────


def _generate_match_narrative(
    fixture: Fixture,
    prediction: MatchPrediction,
    stats: MatchStats | None,
    value_bets: list[ValueBet],
) -> str:
    """Generate a natural language analysis paragraph for a match in PT-PT."""
    parts: list[str] = []
    home = fixture.home_team
    away = fixture.away_team

    # Opening — prediction
    outcome = _outcome_pt(prediction.predicted_outcome)
    conf = prediction.confidence
    if conf >= 0.50:
        parts.append(
            f"O modelo prevê com boa confiança ({_pct(conf)}) uma <strong>{outcome}</strong> "
            f"neste jogo entre {home} e {away}."
        )
    elif conf >= 0.40:
        parts.append(
            f"A previsão para {home} vs {away} aponta para <strong>{outcome}</strong> "
            f"({_pct(conf)}), embora com alguma incerteza."
        )
    else:
        parts.append(
            f"O jogo {home} vs {away} apresenta grande equilíbrio — o modelo indica "
            f"ligeira tendência para <strong>{outcome}</strong> ({_pct(conf)})."
        )

    # Probabilities comparison with bookmaker
    bk_probs = fixture.implied_probabilities()
    if bk_probs["home"] > 0:
        home_diff = prediction.home_win_prob - bk_probs["home"]
        away_diff = prediction.away_win_prob - bk_probs["away"]
        draw_diff = prediction.draw_prob - bk_probs["draw"]

        biggest = max(
            ("casa", home_diff, prediction.home_win_prob, bk_probs["home"]),
            ("empate", draw_diff, prediction.draw_prob, bk_probs["draw"]),
            ("fora", away_diff, prediction.away_win_prob, bk_probs["away"]),
            key=lambda x: abs(x[1]),
        )
        if abs(biggest[1]) > 0.05:
            direction = "acima" if biggest[1] > 0 else "abaixo"
            parts.append(
                f"A maior divergência com as casas de apostas está na {biggest[0]} — "
                f"o modelo atribui {_pct(biggest[2])} contra {_pct(biggest[3])} das odds, "
                f"ficando {abs(biggest[1])*100:.0f} pontos percentuais {direction}."
            )

    # Goals / Over-Under / BTTS
    if stats:
        total_xg = stats.total_xg
        if total_xg >= 3.0:
            parts.append(
                f"Espera-se um jogo com golos (xG total: {total_xg:.1f}). "
                f"A probabilidade de Over 2.5 é de {_pct(stats.over25_prob)}, "
                f"sugerindo que apostar em mais de 2.5 golos pode ter interesse."
            )
        elif total_xg >= 2.0:
            parts.append(
                f"O jogo deverá ter um número moderado de golos (xG total: {total_xg:.1f}). "
                f"Over 2.5 está a {_pct(stats.over25_prob)}."
            )
        else:
            parts.append(
                f"Prevê-se um jogo fechado e de poucos golos (xG total: {total_xg:.1f}). "
                f"Under 2.5 tem uma probabilidade de {_pct(stats.under25_prob)}."
            )

        if stats.btts_yes_prob > 0.60:
            parts.append(
                f"Ambas as equipas deverão marcar — probabilidade de {_pct(stats.btts_yes_prob)}. "
                f"Historicamente, jogos de {home} têm BTTS em {_pct(stats.home_btts_pct)} dos casos "
                f"e {away} em {_pct(stats.away_btts_pct)}."
            )
        elif stats.btts_no_prob > 0.60:
            parts.append(
                f"É mais provável que pelo menos uma equipa não marque "
                f"(BTTS Não: {_pct(stats.btts_no_prob)})."
            )

        # Top scoreline
        if stats.top_scorelines:
            h, a, p = stats.top_scorelines[0]
            parts.append(
                f"O resultado mais provável é <strong>{h}-{a}</strong> ({_pct(p)})."
            )

        # Form
        home_form = _form_label_pt(stats.home_form).lower()
        away_form = _form_label_pt(stats.away_form).lower()
        if abs(stats.home_form - stats.away_form) > 0.5:
            better = home if stats.home_form > stats.away_form else away
            parts.append(
                f"Em termos de forma recente, {better} chega em melhor momento. "
                f"{home} com forma {home_form} ({stats.home_form:.1f} ppj) "
                f"e {away} com forma {away_form} ({stats.away_form:.1f} ppj)."
            )
        else:
            parts.append(
                f"Ambas as equipas apresentam forma semelhante: "
                f"{home} ({stats.home_form:.1f} ppj) e {away} ({stats.away_form:.1f} ppj)."
            )

    # Value bets
    if value_bets:
        best_vb = value_bets[0]
        outcome_text = _outcome_pt(best_vb.outcome).lower()
        conf_text = _confidence_pt(best_vb.confidence).lower()
        parts.append(
            f"<strong>💡 Aposta de valor detetada:</strong> {outcome_text} com edge de "
            f"+{_pct(best_vb.edge)} (confiança {conf_text}). "
            f"Procure odds mínimas de {best_vb.best_odds:.2f}. "
            f"Kelly sugere {_pct(best_vb.kelly_fraction)} da banca."
        )
        if len(value_bets) > 1:
            others = ", ".join(
                f"{_outcome_pt(vb.outcome).lower()} (+{_pct(vb.edge)})"
                for vb in value_bets[1:]
            )
            parts.append(f"Outras oportunidades: {others}.")
    else:
        parts.append(
            "Não foram detetadas apostas de valor significativas neste jogo "
            "face às odds atuais."
        )

    return " ".join(parts)


def _generate_summary_narrative(
    fixtures: list[Fixture],
    predictions: list[MatchPrediction],
    match_stats_list: list[MatchStats | None],
    value_bets: list[ValueBet],
) -> str:
    """Generate a global summary narrative in PT-PT."""
    parts: list[str] = []

    n = len(fixtures)
    n_vb = len(value_bets)
    high = sum(1 for vb in value_bets if vb.confidence == "HIGH")
    medium = sum(1 for vb in value_bets if vb.confidence == "MEDIUM")

    parts.append(
        f"Foram analisados <strong>{n} jogos</strong> nesta sessão, "
        f"tendo sido identificadas <strong>{n_vb} apostas de valor</strong>"
    )
    if high or medium:
        confs = []
        if high:
            confs.append(f"{high} de confiança alta")
        if medium:
            confs.append(f"{medium} de confiança média")
        parts[-1] += f" ({', '.join(confs)})."
    else:
        parts[-1] += "."

    if value_bets:
        best = value_bets[0]
        parts.append(
            f"A melhor oportunidade é em <strong>{best.home_team} vs {best.away_team}</strong> — "
            f"{_outcome_pt(best.outcome).lower()} com um edge de +{_pct(best.edge)} "
            f"sobre as casas de apostas."
        )

    # High-scoring games
    high_xg = [
        (f, s) for f, s in zip(fixtures, match_stats_list) if s and s.total_xg >= 3.0
    ]
    if high_xg:
        names = ", ".join(f"{f.home_team} vs {f.away_team}" for f, _ in high_xg)
        parts.append(f"Jogos com mais golos esperados: {names}.")

    parts.append(
        "<em>Nota: Este relatório é gerado por um modelo de Machine Learning. "
        "Resultados passados não garantem resultados futuros. Aposte com responsabilidade.</em>"
    )

    return " ".join(parts)


# ── HTML Card Builders ────────────────────────────────────────────────


def _value_bet_card(vb: ValueBet) -> str:
    color = _confidence_color(vb.confidence)
    conf_pt = _confidence_pt(vb.confidence)
    outcome_pt = _outcome_pt(vb.outcome)
    return f"""
    <div class="value-bet-item">
      <div class="vb-header">
        <span class="vb-match">{vb.home_team} vs {vb.away_team}</span>
        <span class="vb-confidence" style="background:{color}">{conf_pt}</span>
      </div>
      <div class="vb-details">
        <div class="vb-outcome">Aposta: <strong>{outcome_pt}</strong></div>
        <div class="vb-stats">
          <div class="vb-stat">
            <span class="vb-stat-label">Prob. ML</span>
            <span class="vb-stat-value">{_pct(vb.ml_probability)}</span>
          </div>
          <div class="vb-stat">
            <span class="vb-stat-label">Bookmaker</span>
            <span class="vb-stat-value">{_pct(vb.bookmaker_probability)}</span>
          </div>
          <div class="vb-stat">
            <span class="vb-stat-label">Edge</span>
            <span class="vb-stat-value" style="color:#22c55e">+{_pct(vb.edge)}</span>
          </div>
          <div class="vb-stat">
            <span class="vb-stat-label">Odds Mín.</span>
            <span class="vb-stat-value">{vb.best_odds:.2f}</span>
          </div>
          <div class="vb-stat">
            <span class="vb-stat-label">Kelly</span>
            <span class="vb-stat-value">{_pct(vb.kelly_fraction)}</span>
          </div>
        </div>
      </div>
    </div>"""


def _match_card(
    fixture: Fixture,
    prediction: MatchPrediction,
    match_stats: MatchStats | None,
    value_bets: list[ValueBet],
    min_odds: dict[str, float],
    narrative: str,
) -> str:
    # Value bet indicators
    vb_html = ""
    for vb in value_bets:
        color = _confidence_color(vb.confidence)
        outcome_pt = _outcome_pt(vb.outcome)
        vb_html += f'<span class="vb-badge" style="border-color:{color}">VALOR: {outcome_pt} (+{_pct(vb.edge)})</span>'

    # Stats section
    stats_html = ""
    if match_stats:
        stats_html = f"""
        <div class="stats-grid">
          <div class="stat-section">
            <h4>Golos Esperados</h4>
            <div class="xg-display">
              <span class="xg-home">{match_stats.home_xg:.1f}</span>
              <span class="xg-label">xG</span>
              <span class="xg-away">{match_stats.away_xg:.1f}</span>
            </div>
            <div class="xg-total">xG Total: {match_stats.total_xg:.1f}</div>
          </div>

          <div class="stat-section">
            <h4>Mercado de Golos</h4>
            <div class="market-rows">
              <div class="market-row">
                <span>Over 1.5</span>
                <div class="mini-bar-bg"><div class="mini-bar" style="width:{match_stats.over15_prob*100:.0f}%"></div></div>
                <span>{_pct(match_stats.over15_prob)}</span>
              </div>
              <div class="market-row highlight-row">
                <span>Over 2.5</span>
                <div class="mini-bar-bg"><div class="mini-bar" style="width:{match_stats.over25_prob*100:.0f}%"></div></div>
                <span>{_pct(match_stats.over25_prob)}</span>
              </div>
              <div class="market-row">
                <span>Over 3.5</span>
                <div class="mini-bar-bg"><div class="mini-bar" style="width:{match_stats.over35_prob*100:.0f}%"></div></div>
                <span>{_pct(match_stats.over35_prob)}</span>
              </div>
            </div>
          </div>

          <div class="stat-section">
            <h4>Ambas Marcam</h4>
            <div class="btts-display">
              <div class="btts-option {'btts-yes-active' if match_stats.btts_yes_prob > 0.5 else ''}">
                <span class="btts-label">Sim</span>
                <span class="btts-value">{_pct(match_stats.btts_yes_prob)}</span>
              </div>
              <div class="btts-option {'btts-no-active' if match_stats.btts_no_prob > 0.5 else ''}">
                <span class="btts-label">Não</span>
                <span class="btts-value">{_pct(match_stats.btts_no_prob)}</span>
              </div>
            </div>
            <div class="btts-historical">
              Histórico: {fixture.home_team} {_pct(match_stats.home_btts_pct)} | {fixture.away_team} {_pct(match_stats.away_btts_pct)}
            </div>
          </div>

          <div class="stat-section">
            <h4>Resultados Mais Prováveis</h4>
            <div class="scorelines">{_scoreline_badges(match_stats.top_scorelines)}</div>
          </div>

          <div class="stat-section">
            <h4>Forma Recente</h4>
            <div class="form-display">
              <div>{fixture.home_team}: {_form_dots(match_stats.home_form, fixture.home_team)}</div>
              <div>{fixture.away_team}: {_form_dots(match_stats.away_form, fixture.away_team)}</div>
            </div>
          </div>
        </div>"""

    outcome_pt = _outcome_pt(prediction.predicted_outcome)

    return f"""
    <div class="match-card {'has-value' if value_bets else ''}">
      <div class="match-header">
        <div class="match-meta">
          <span class="match-league">{fixture.league}</span>
          <span class="match-time">{fixture.time}</span>
        </div>
        <div class="match-teams">
          <span class="team home-team">{fixture.home_team}</span>
          <span class="vs">vs</span>
          <span class="team away-team">{fixture.away_team}</span>
        </div>
        {f'<div class="vb-badges">{vb_html}</div>' if vb_html else ''}
      </div>

      <div class="match-body">
        <div class="prediction-section">
          <h4>Previsão</h4>
          {_prob_bar("Casa", prediction.home_win_prob, "#22c55e")}
          {_prob_bar("Empate", prediction.draw_prob, "#7fb88a")}
          {_prob_bar("Fora", prediction.away_win_prob, "#ef4444")}

          <div class="prediction-verdict">
            <span class="verdict-label">Previsão:</span>
            <span class="verdict-value">{outcome_pt}</span>
            <span class="verdict-conf">({_pct(prediction.confidence)})</span>
          </div>

          <div class="odds-section">
            <h4>Odds (B365)</h4>
            <div class="odds-row">
              <span class="odds-chip">1 {fixture.b365_home:.2f}</span>
              <span class="odds-chip">X {fixture.b365_draw:.2f}</span>
              <span class="odds-chip">2 {fixture.b365_away:.2f}</span>
            </div>
          </div>

          <div class="look-for-section">
            <h4>Procure odds &ge;</h4>
            <div class="odds-row">
              <span class="odds-chip target">1 {min_odds['home']:.2f}</span>
              <span class="odds-chip target">X {min_odds['draw']:.2f}</span>
              <span class="odds-chip target">2 {min_odds['away']:.2f}</span>
            </div>
          </div>
        </div>

        {stats_html}
      </div>

      <div class="match-narrative">
        <h4>📝 Análise</h4>
        <p>{narrative}</p>
      </div>
    </div>"""


# ── CSS ──────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #0a1a0e;
  --card-bg: #0f2d16;
  --card-border: #1e5c2a;
  --text: #e2f0e6;
  --text-muted: #7fb88a;
  --accent: #22c55e;
  --green: #16a34a;
  --yellow: #eab308;
  --red: #ef4444;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.6;
  padding: 2rem;
}

.container { max-width: 1200px; margin: 0 auto; }

.header {
  text-align: center;
  margin-bottom: 2rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid var(--card-border);
}
.header h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.25rem; }
.header .subtitle { color: var(--text-muted); font-size: 0.9rem; }

.section-title {
  font-size: 1.3rem;
  font-weight: 600;
  margin: 2rem 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--accent);
  display: inline-block;
}

/* Summary narrative */
.summary-box {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-left: 4px solid var(--accent);
  border-radius: 8px;
  padding: 1.25rem 1.5rem;
  margin-bottom: 2rem;
  font-size: 0.95rem;
  line-height: 1.7;
}
.summary-box em { color: var(--text-muted); font-size: 0.8rem; }

/* Match Cards */
.matches-grid {
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.5rem;
}

.match-card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  overflow: hidden;
  transition: border-color 0.2s;
}
.match-card.has-value { border-color: var(--green); border-width: 2px; }
.match-card:hover { border-color: var(--accent); }

.match-header {
  padding: 1.25rem 1.5rem;
  background: rgba(255,255,255,0.02);
  border-bottom: 1px solid var(--card-border);
}
.match-meta { display: flex; justify-content: space-between; margin-bottom: 0.5rem; }
.match-league {
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--accent);
  font-weight: 600;
}
.match-time { font-size: 0.8rem; color: var(--text-muted); }
.match-teams { display: flex; align-items: center; gap: 0.75rem; font-size: 1.25rem; font-weight: 600; }
.vs { color: var(--text-muted); font-size: 0.85rem; font-weight: 400; }

.vb-badges { margin-top: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
.vb-badge {
  font-size: 0.7rem;
  padding: 0.2rem 0.6rem;
  border: 1.5px solid;
  border-radius: 999px;
  font-weight: 600;
  color: var(--green);
}

.match-body {
  padding: 1.5rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
}

.match-narrative {
  padding: 1rem 1.5rem 1.25rem;
  border-top: 1px solid var(--card-border);
  background: rgba(255,255,255,0.01);
}
.match-narrative h4 {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--accent);
  margin-bottom: 0.5rem;
}
.match-narrative p {
  font-size: 0.88rem;
  line-height: 1.75;
  color: var(--text);
}

@media (max-width: 768px) {
  .match-body { grid-template-columns: 1fr; }
  body { padding: 1rem; }
}

/* Prediction */
.prediction-section h4, .stats-grid h4 {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 0.75rem;
}

.prob-row { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
.prob-label { width: 55px; font-size: 0.8rem; color: var(--text-muted); }
.prob-bar-bg { flex: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }
.prob-bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.prob-value { width: 40px; text-align: right; font-size: 0.8rem; font-weight: 600; }

.prediction-verdict {
  margin-top: 0.75rem;
  padding: 0.5rem 0.75rem;
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
  font-size: 0.85rem;
}
.verdict-value { font-weight: 700; color: var(--accent); margin: 0 0.25rem; }
.verdict-conf { color: var(--text-muted); }

.odds-section, .look-for-section { margin-top: 1rem; }
.odds-row { display: flex; gap: 0.5rem; }
.odds-chip {
  padding: 0.35rem 0.75rem;
  background: rgba(255,255,255,0.05);
  border-radius: 6px;
  font-size: 0.85rem;
  font-weight: 600;
}
.odds-chip.target { background: rgba(34,197,94,0.1); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }

/* Stats Grid */
.stats-grid { display: flex; flex-direction: column; gap: 1.25rem; }

.stat-section {
  padding: 0.75rem;
  background: rgba(255,255,255,0.02);
  border-radius: 8px;
}

.xg-display { display: flex; align-items: center; justify-content: center; gap: 1rem; font-size: 1.5rem; font-weight: 700; }
.xg-label { font-size: 0.75rem; color: var(--text-muted); font-weight: 400; }
.xg-home { color: var(--accent); }
.xg-away { color: var(--red); }
.xg-total { text-align: center; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.25rem; }

.market-rows { display: flex; flex-direction: column; gap: 0.3rem; }
.market-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; }
.market-row span:first-child { width: 60px; color: var(--text-muted); }
.market-row span:last-child { width: 35px; text-align: right; font-weight: 600; }
.highlight-row { font-weight: 600; color: var(--text); }
.mini-bar-bg { flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px; overflow: hidden; }
.mini-bar { height: 100%; background: var(--accent); border-radius: 3px; }

.btts-display { display: flex; gap: 0.75rem; justify-content: center; }
.btts-option { padding: 0.5rem 1rem; background: rgba(255,255,255,0.03); border-radius: 8px; text-align: center; min-width: 80px; }
.btts-yes-active { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); }
.btts-no-active { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); }
.btts-label { display: block; font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; }
.btts-value { display: block; font-size: 1.1rem; font-weight: 700; }
.btts-historical { font-size: 0.7rem; color: var(--text-muted); text-align: center; margin-top: 0.5rem; }

.scorelines { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.score-badge {
  padding: 0.3rem 0.6rem;
  background: rgba(255,255,255,0.05);
  border-radius: 6px;
  font-size: 0.8rem;
  font-weight: 600;
}
.score-badge small { color: var(--text-muted); font-weight: 400; }

.form-display { display: flex; flex-direction: column; gap: 0.4rem; font-size: 0.85rem; }
.form-indicator {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 700;
  color: #000;
  margin-left: 0.25rem;
}
.form-label { color: var(--text-muted); }

/* Value Bets Summary */
.value-bets-section { margin-top: 2rem; }
.value-bet-item {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  margin-bottom: 0.75rem;
}
.vb-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
.vb-match { font-weight: 600; font-size: 1rem; }
.vb-confidence { padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.7rem; font-weight: 700; color: #000; }
.vb-outcome { font-size: 0.9rem; margin-bottom: 0.5rem; }
.vb-stats { display: flex; flex-wrap: wrap; gap: 1rem; }
.vb-stat { }
.vb-stat-label { display: block; font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; }
.vb-stat-value { font-size: 1rem; font-weight: 700; }

.no-value { color: var(--text-muted); font-style: italic; padding: 1rem 0; }

.footer {
  text-align: center;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--card-border);
  color: var(--text-muted);
  font-size: 0.75rem;
}
"""


# ── Main Generator ───────────────────────────────────────────────────


def generate_html_report(
    fixtures: list[Fixture],
    predictions: list[MatchPrediction],
    match_stats_list: list[MatchStats | None],
    value_bets: list[ValueBet],
    output_path: str | Path,
) -> Path:
    """Gerar relatório HTML completo em Português de Portugal."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%d/%m/%Y, %H:%M UTC")

    # Summary narrative
    summary = _generate_summary_narrative(
        fixtures, predictions, match_stats_list, value_bets
    )

    # Build match cards
    cards_html = ""
    for fixture, pred, stats in zip(fixtures, predictions, match_stats_list):
        match_vbs = [
            vb
            for vb in value_bets
            if vb.home_team == pred.home_team and vb.away_team == pred.away_team
        ]

        min_odds = {
            "home": round(1 / pred.home_win_prob, 2) if pred.home_win_prob > 0 else 99,
            "draw": round(1 / pred.draw_prob, 2) if pred.draw_prob > 0 else 99,
            "away": round(1 / pred.away_win_prob, 2) if pred.away_win_prob > 0 else 99,
        }

        narrative = _generate_match_narrative(fixture, pred, stats, match_vbs)
        cards_html += _match_card(fixture, pred, stats, match_vbs, min_odds, narrative)

    # Value bets summary
    vb_summary = ""
    if value_bets:
        for vb in value_bets:
            vb_summary += _value_bet_card(vb)
    else:
        vb_summary = '<p class="no-value">Não foram detetadas apostas de valor com os limiares atuais.</p>'

    html = f"""<!DOCTYPE html>
<html lang="pt-PT">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Previsões Futebol - {now}</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>⚽ Relatório de Previsões</h1>
      <div class="subtitle">{now} &middot; {len(fixtures)} jogos analisados</div>
    </div>

    <div class="section-title">Resumo</div>
    <div class="summary-box">{summary}</div>

    <div class="section-title">Análise por Jogo</div>
    <div class="matches-grid">
      {cards_html}
    </div>

    <div class="value-bets-section">
      <div class="section-title">Apostas de Valor</div>
      {vb_summary}
    </div>

    <div class="footer">
      Gerado pelo Football Prediction Agent &middot; Ensemble ML + Modelo Poisson &middot; Resultados passados não garantem resultados futuros &middot; Aposte com responsabilidade
    </div>
  </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
