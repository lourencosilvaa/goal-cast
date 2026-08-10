"""AI analysis endpoint — Gemini or NVIDIA, whichever the user asked for.

Thin by design: pick the provider, build the prompt, translate a typed failure
into a status code. Which back-ends exist, which key each reads and what model
each defaults to all live in :mod:`src.backend.services.ai_providers` and in
the ``ai:`` config block, so adding one never touches this file.

It used to construct a ``genai.Client`` directly and fetch the Gemini key
regardless of what was asked for, which is why the NVIDIA key users had been
storing since Settings grew a field for it did nothing at all.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from src.backend.api.keys import get_api_key_service
from src.backend.core.auth import get_approved_user
from src.backend.services.ai_providers import (
    AnalysisRequest,
    MissingProviderKeyError,
    ProviderUnavailable,
    UnknownProviderError,
    build_analysis_provider,
)
from src.backend.services.api_key_service import ApiKeyService

router = APIRouter(prefix="/api", tags=["ai"])


class AIAnalysisRequest(BaseModel):
    match_data: dict
    language: str = "pt-PT"
    #: Empty means the provider's configured default, not a hardcoded name —
    #: a model string only means something next to the provider it belongs to.
    model: str = ""
    #: Empty means ``ai.default_provider``.
    provider: str = ""


class AIAnalysisResponse(BaseModel):
    analysis: str
    #: Echoed back so the UI can label which back-end actually answered.
    provider: str


@router.post("/ai/analyze", response_model=AIAnalysisResponse)
async def analyze_match(
    body: AIAnalysisRequest,
    request: Request,
    user_id: Annotated[str, Depends(get_approved_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> AIAnalysisResponse:
    """Generate a match analysis with the user's stored key for that provider."""
    config = request.app.state.config.ai
    try:
        provider = build_analysis_provider(body.provider, config, service)
    except UnknownProviderError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        analysis = provider.analyse(
            AnalysisRequest(
                prompt=_build_prompt(body.match_data, body.language),
                model=body.model,
                user_id=user_id,
            )
        )
    except MissingProviderKeyError as exc:
        # 400, not 401: the request authenticated fine, the account simply has
        # no key for this back-end, and the fix is in Settings.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderUnavailable as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    return AIAnalysisResponse(
        analysis=analysis, provider=(body.provider or config.default_provider).lower()
    )


def _build_prompt(match_data: dict, language: str) -> str:
    """Build the analysis prompt from match data."""
    home = match_data.get("home_team", "?")
    away = match_data.get("away_team", "?")
    probs = match_data.get("probabilities", {})
    odds = match_data.get("odds", {})
    xg = match_data.get("expected_goals", {})
    ou = match_data.get("over_under", {})
    btts = match_data.get("btts", {})
    form = match_data.get("form", {})
    value_bets = match_data.get("value_bets", [])
    scorelines = match_data.get("top_scorelines", [])

    lang_instruction = (
        "Responde em Português de Portugal (PT-PT)."
        if language == "pt-PT"
        else f"Respond in {language}."
    )

    return f"""Ès um analista de futebol especializado. Analisa o seguinte jogo com base nos dados do modelo de ML e estatísticas.

{lang_instruction}

**Jogo:** {home} vs {away}

**Probabilidades do Modelo ML:**
- Vitória Casa: {probs.get('home_win', 0):.1%}
- Empate: {probs.get('draw', 0):.1%}
- Vitória Fora: {probs.get('away_win', 0):.1%}

**Odds Bet365:** Casa {odds.get('home', 0)} | Empate {odds.get('draw', 0)} | Fora {odds.get('away', 0)}

**Golos Esperados (xG):** {home} {xg.get('home', 0):.1f} - {xg.get('away', 0):.1f} {away} (Total: {xg.get('total', 0):.1f})

**Mercado de Golos:**
- Over 1.5: {ou.get('over_15', 0):.0%} | Over 2.5: {ou.get('over_25', 0):.0%} | Over 3.5: {ou.get('over_35', 0):.0%}

**Ambas Marcam:** Sim {btts.get('yes', 0):.0%} | Não {btts.get('no', 0):.0%}

**Forma Recente (ppj):** {home} {form.get('home', 0):.1f} | {away} {form.get('away', 0):.1f}

**Resultados Mais Prováveis:** {', '.join(f"{s.get('score', '?')} ({s.get('prob', 0):.0%})" for s in scorelines[:5])}

**Apostas de Valor Detetadas:** {', '.join(f"{vb.get('outcome', '?')} (edge {vb.get('edge_pct', '?')})" for vb in value_bets) if value_bets else 'Nenhuma'}

Fornece uma análise concisa (3-4 parágrafos) que inclua:
1. Avaliação geral do jogo e previsão principal
2. Análise dos mercados de golos (Over/Under, BTTS)
3. Apostas de valor recomendadas (se existirem) com justificação
4. Fatores de risco e nível de confiança

Usa emojis para destacar pontos-chave (⚽ para golos, 💡 para insights, ⚠️ para riscos).
Sê direto e objetivo. Não repitas os dados em bruto, interpreta-os."""  # noqa: E501
