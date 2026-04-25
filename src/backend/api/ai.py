"""AI analysis endpoint — uses Gemini to generate personalized match analysis."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from google.genai.errors import APIError, ClientError, ServerError
from pydantic import BaseModel

from src.backend.api.keys import get_api_key_service
from src.backend.core.auth import get_current_user
from src.backend.services.api_key_service import ApiKeyService

router = APIRouter(prefix="/api", tags=["ai"])


class AIAnalysisRequest(BaseModel):
    match_data: dict
    language: str = "pt-PT"
    model: str = "gemini-2.5-flash"


class AIAnalysisResponse(BaseModel):
    analysis: str


@router.post("/ai/analyze", response_model=AIAnalysisResponse)
async def analyze_match(
    body: AIAnalysisRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    service: Annotated[ApiKeyService, Depends(get_api_key_service)],
) -> AIAnalysisResponse:
    """Generate AI-powered match analysis using the user's stored Gemini key."""
    gemini_api_key = service.get_user_key(user_id=user_id)
    if not gemini_api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key not configured. Add it in Settings.",
        )

    try:
        from google import genai

        client = genai.Client(api_key=gemini_api_key)

        prompt = _build_prompt(body.match_data, body.language)

        response = client.models.generate_content(
            model=body.model,
            contents=prompt,
        )

        return AIAnalysisResponse(analysis=response.text or "")
    except ClientError as e:
        if e.code in (401, 403):
            raise HTTPException(status_code=401, detail="API key do Gemini inválida.")
        if e.code == 404:
            raise HTTPException(
                status_code=400,
                detail=f"Modelo '{body.model}' não encontrado. Verifica o nome nas Definições.",  # noqa: E501
            )
        if e.code == 429:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Limite de pedidos excedido para o modelo {body.model}."
                    " Aguarda alguns segundos e tenta novamente."
                ),
            )
        raise HTTPException(status_code=e.code, detail=e.message or str(e))
    except ServerError as e:
        if e.code == 503:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"O modelo {body.model} está com elevada procura."
                    " Tenta novamente em alguns segundos ou"
                    " escolhe outro modelo nas Definições."
                ),
            )
        raise HTTPException(status_code=e.code, detail=e.message or str(e))
    except APIError as e:
        raise HTTPException(status_code=e.code or 500, detail=e.message or str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")


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
