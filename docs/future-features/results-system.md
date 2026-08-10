# Sistema de Resultados de Futebol — serviço dedicado (histórico + tempo real)

> **Estado**: plano **v2** aprovado a 2026-08-09 e **implementado** a
> 2026-08-09. O documento fica como registo da decisão de arquitetura; o que
> está a correr está descrito no README (secção *Results service (history +
> live)*). Diferenças face ao plano, todas deliberadas:
>
> - **`src/contracts/results.py`** (novo, não previsto no plano): os modelos
>   Pydantic da resposta têm de ser importáveis pelos **dois** lados da
>   fronteira HTTP. Vivê-los dentro de `src/results_service/api/` obrigaria a
>   imagem da app principal a copiar o serviço inteiro — exatamente o que a
>   separação evita. O plano pedia "contrato partilhado"; isto é a forma
>   literal de o ser.
> - **`src/results_service/app.py`** separado de `main.py`: `main.py` constrói
>   a app no import (para o processo morrer no arranque quando falta a key,
>   como planeado), e essa é a razão pela qual a *factory* tem de viver noutro
>   módulo para ser testável. O `CMD` continua a ser
>   `uvicorn src.results_service.main:app`, como no plano.
> - **`local_corpus` tem bloco de config próprio** em vez de uma entrada em
>   `results.providers`: não é um provider HTTP, não tem key nem `base_url` de
>   API, e a lista `leagues` (quais os códigos com feed CSV) não existe para
>   nenhum outro.
> - **`GET /v4/matches?date=…` confirmado contra a API real** a 2026-08-09: o
>   free tier responde `LIVE` (não `IN_PLAY`) e **não envia `minute`** — ambos
>   estão tratados e testados com o payload gravado.
>

> **Decisões do utilizador**:
> - API-primeiro (football-data.org) com scraper Flashscore como fallback opcional.
> - Tempo real por polling on-demand com TTL + endpoints REST (sem WebSocket/SSE).
> - **v2**: o scraping vive num **microserviço dedicado** — mesmo repositório,
>   código próprio (`src/results_service/`), `Dockerfile.results`, workflow
>   `deploy-results.yml` e serviço Render próprios; a app principal consome-o
>   por HTTP através de um gateway com interface substituível.

---

# 📋 Architectural Plan v2 — Serviço dedicado de Resultados (histórico + tempo real)

## Objective

Criar um **microserviço FastAPI dedicado de scraping/recolha de resultados** — código, Dockerfile, workflow de deployment e serviço Render próprios, no mesmo repositório — que serve histórico e resultados ao vivo; a app principal consome-o por HTTP através de um cliente com interface substituível.

## O que muda face ao plano v1

| Aspeto | v1 | v2 (aprovado) |
|--------|----|----------------|
| Onde vive a lógica | Dentro do backend principal | Serviço FastAPI próprio (`src/results_service/`) |
| Endpoints live/history | No backend principal | No serviço dedicado; backend expõe proxy fino para o frontend |
| Deployment | Junto com o deploy existente | `Dockerfile.results` + `deploy-results.yml` + 2º serviço Render |
| Playwright/deps de scraping | Entrariam na imagem principal | Isolados na imagem do serviço — a imagem principal não engorda |
| Falha do scraping | Podia afetar o backend | Isolada: o backend devolve 502/503 explícito e o resto da app vive |

O **desenho interno de recolha não muda**: providers com ABCs separadas (histórico/live), cadeia primeiro-não-vazio, tracker com TTL, repositório JSON — tudo do v1 mantém-se, apenas passa a correr dentro do serviço dedicado (detalhe integral na secção "Desenho interno de recolha" abaixo).

## Visão geral

```
┌────────────────────────── Render ───────────────────────────┐
│                                                             │
│  ┌─────────────────────┐  HTTP (X-API-Key)  ┌─────────────┐ │
│  │ App principal        │ ─────────────────► │ Results     │ │
│  │ (backend + frontend) │ GET /live          │ Service     │ │
│  │                      │ GET /history       │ (dedicado)  │ │
│  │ /api/results/* =     │ ◄───────────────── │             │ │
│  │ proxy fino p/ front  │      JSON          └──────┬──────┘ │
│  └─────────────────────┘                            │        │
└─────────────────────────────────────────────────────┼────────┘
                                                      ▼
                                    Cadeias de providers:
                                    histórico: corpus local → football-data.org
                                    live:      football-data.org → Flashscore
```

Deploys independentes: `deploy.yml` (existente, intocado) e `deploy-results.yml` (novo) → cada um constrói a sua imagem, publica no ghcr.io e dispara o seu deploy hook.

## Affected Modules

| Module / File | Action | Description |
|---------------|--------|-------------|
| `src/scrapers/results/*` | Create | Biblioteca de providers — igual ao v1 (models, base, local_corpus, football_data, flashscore_live, chained, live_tracker) |
| `src/results_service/__init__.py` | Create | Pacote do microserviço |
| `src/results_service/main.py` | Create | App factory FastAPI do serviço; carrega config, monta cadeias por injeção |
| `src/results_service/api/results.py` | Create | `GET /live`, `GET /history` (contratos JSON do v1) |
| `src/results_service/api/health.py` | Create | `GET /health` para o health check do Render |
| `src/results_service/auth.py` | Create | Verificação do header `X-API-Key` contra env `RESULTS_SERVICE_API_KEY` |
| `src/results_service/service.py` | Create | Orquestração cadeias + tracker + repositório (era o `ResultsService` do v1) |
| `src/results_service/repository.py` | Create | `ResultsRepository` (interface + JSON em `datasets/cache/results/`) — vive no serviço |
| `src/backend/services/results_gateway.py` | Create | Interface `ResultsGateway` + `HttpResultsGateway` (cliente do serviço, timeout/erros explícitos) |
| `src/backend/api/results.py` | Create | Proxy fino `/api/results/live` e `/api/results/history` para o frontend (autenticação de utilizador existente; a service key nunca chega ao browser) |
| `src/backend/main.py` (ou registo de routers) | Modify | Registar o router novo |
| `Dockerfile.results` | Create | Imagem dedicada: python-slim + grupo `results` + Playwright/Chromium (só aqui) |
| `.github/workflows/deploy-results.yml` | Create | Build → push `ghcr.io/<repo>-results` → hook `RENDER_RESULTS_DEPLOY_HOOK_URL`; paths filtrados ao código do serviço |
| `pyproject.toml` | Modify | Novo dependency group `results` (fastapi, uvicorn, pydantic, pyyaml, requests, playwright) |
| `config/config.yaml` + `config/config_loader.py` | Modify | Secção `results:` do v1 + bloco `service:` (porta, nomes de env vars) e, no lado da app, `results_gateway:` (URL do serviço via env, timeout) |
| `tests/scrapers/results/`, `tests/results_service/`, `tests/backend/` | Create | Testes espelhando cada módulo |

## Design detalhado (deltas sobre o v1)

### 1. Serviço dedicado — `src/results_service/`

FastAPI mínimo (sem Supabase, sem ML): app factory que lê a config, constrói `HttpTransport`, providers, cadeias, tracker e repositório **por injeção no arranque** e regista dois routers. Endpoints:

```
GET /health                          → 200 {"status": "ok", "git_sha": ...}   (sem auth — health check do Render)
GET /live?leagues=E0,P1              → payload do v1 (snapshot + events + stale + source)
GET /history?league=P1&season=2526   → payload do v1
```

Autenticação serviço-a-serviço: header `X-API-Key` validado contra `RESULTS_SERVICE_API_KEY` (mesmo padrão do `RETRAIN_API_KEY` já usado no projeto). Sem key configurada no ambiente → o serviço **recusa arrancar** em vez de servir aberto (sem fallbacks silenciosos, §7.4).

### 2. App principal — cliente com interface substituível

```python
class ResultsGateway(ABC):
    @abstractmethod
    def live(self, leagues: list[str]) -> LiveResultsResponse: ...
    @abstractmethod
    def history(self, league: str, season: str) -> HistoryResponse: ...

class HttpResultsGateway(ResultsGateway):
    def __init__(self, config: ResultsGatewayConfig, transport: Transport, api_key: str) -> None: ...
```

- O backend **não importa nada** de `src/scrapers/results/` — só conhece a interface e os modelos de resposta Pydantic (contrato partilhado).
- Serviço em baixo/timeout → o proxy devolve **503 com mensagem explícita** (`"results service unreachable"`), nunca lista vazia silenciosa.
- Config: `results_gateway.base_url_env: RESULTS_SERVICE_URL`, `api_key_env: RESULTS_SERVICE_API_KEY`, `timeout` — injetada por construtor.
- A interface é o que torna a decisão reversível: um `InProcessResultsGateway` (v1) seria uma implementação alternativa sem tocar no router.

### 3. `Dockerfile.results`

- Stage único `python:3.12-slim` (sem frontend, sem ML): `uv sync --frozen --no-dev --only-group results`, `playwright install --with-deps chromium` (o peso do Chromium fica **só** nesta imagem — benefício direto da separação), copia `config/`, `src/scrapers/`, `src/results_service/`, `src/teams/` se o resolver de aliases for usado no serviço.
- `CMD uvicorn src.results_service.main:app --host 0.0.0.0 --port 8000` (mesmo padrão de arranque direto do venv do Dockerfile existente, pelas mesmas razões de cold start).

### 4. `.github/workflows/deploy-results.yml`

Cópia estrutural do `deploy.yml` existente:

- **Trigger**: push a `main` com paths `src/results_service/**`, `src/scrapers/**`, `config/**`, `pyproject.toml`, `Dockerfile.results` + `workflow_dispatch`.
- **Imagem**: `ghcr.io/<repo>-results:latest` e `:sha`, cache GHA com `scope: results` (não colide com o `scope: app` do workflow atual).
- **Deploy**: hook em secret novo `RENDER_RESULTS_DEPLOY_HOOK_URL`.
- O `deploy.yml` existente **não é alterado** — os seus paths já não apanham o código novo do serviço.

### 5. Render — segundo serviço (setup manual, documentado no workflow)

1. Web Service novo a partir de `ghcr.io/<repo>-results:latest`, port 8000, health check `/health`.
2. Env vars: `FOOTBALL_DATA_API_KEY`, `RESULTS_SERVICE_API_KEY`.
3. No serviço principal, adicionar: `RESULTS_SERVICE_URL` (URL interno do serviço novo) e `RESULTS_SERVICE_API_KEY` (o mesmo valor).
4. Copiar o deploy hook → secret GitHub `RENDER_RESULTS_DEPLOY_HOOK_URL`.

### 6. Corpus local no serviço

O corpus local (CSVs de `datasets/`) tem de existir **na imagem/disco do serviço** — como o serviço Render não tem os datasets do pipeline de treino, o `LocalCorpusHistoryProvider` fará download/cache dos CSVs football-data.co.uk na primeira consulta (mesma fonte `data.base_url` já configurada), guardando em `datasets/cache/results/`. Em dev local usa os ficheiros já existentes.

---

## Desenho interno de recolha (herdado do v1, sem alteração)

### Modelos de domínio — `src/scrapers/results/models.py`

Dataclasses imutáveis, nomes de equipas **na grafia do provider** (a resolução para nomes canónicos acontece no serviço, via o resolver de aliases existente — mesma regra dos fixtures europeus):

```python
class MatchStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"          # inclui intervalo
    PAUSED = "paused"
    FINISHED = "finished"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class MatchResult:
    league: str            # código interno (E0, P1, CL…)
    kickoff: datetime
    home_team: str
    away_team: str
    status: MatchStatus
    home_goals: int | None # None enquanto não começou
    away_goals: int | None
    minute: str            # "67'", "HT", "" quando não aplicável
    source: str            # que provider respondeu
    source_id: str = ""

@dataclass(frozen=True)
class HistoryQuery:
    league: str
    season: str            # formato "2526", igual a data.seasons

@dataclass(frozen=True)
class LiveSnapshot:
    fetched_at: datetime
    matches: tuple[MatchResult, ...]
    source: str
```

### Interfaces — `base.py`

**Duas ABCs separadas**, seguindo o precedente do projeto (em `european/providers.py` está documentado que histórico e fixtures são abstrações deliberadamente separadas — cadência e modos de falha diferentes; aqui aplica-se o mesmo a histórico vs live):

```python
class HistoryProvider(ABC):
    name: ClassVar[str]
    def __init__(self, config, transport, api_key: str = "") -> None: ...
    @property
    def enabled(self) -> bool: ...
    @abstractmethod
    def fetch_history(self, query: HistoryQuery) -> list[MatchResult]: ...

class LiveResultsProvider(ABC):
    name: ClassVar[str]
    @abstractmethod
    def fetch_live(self, leagues: list[str]) -> list[MatchResult]: ...
```

Regra herdada da cadeia existente: um provider **nunca levanta exceção por condição esperada** (liga fora de época, competição não coberta pelo plano, erro transiente) — devolve vazio e reporta; a cadeia decide avançar.

### `LocalCorpusHistoryProvider` — `local_corpus.py`

Primeiro da cadeia de histórico. Lê os CSVs football-data.co.uk (locais em dev; descarregados/cacheados no serviço — ver "Corpus local no serviço"). Converte cada linha (`Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`) num `MatchResult` com `status=FINISHED`. É o que garante que o histórico responde instantaneamente e sem gastar quota.

### `FootballDataHistoryProvider` + `FootballDataLiveProvider` — `football_data.py`

Reutilizam o `HttpTransport` de `src/scrapers/european/http.py` (timeout/retry num só sítio) e o padrão do provider europeu existente (`src/scrapers/european/football_data.py`): header `X-Auth-Token`, `QuotaReport` dos headers, memória de competições `uncovered` (403/404 = plano não cobre → não repetir).

- **Histórico**: `GET /v4/competitions/{code}/matches?season=YYYY&status=FINISHED`. Serve a época corrente e ligas sem CSV local (ex.: CL). Necessita de mapa `liga interna → código football-data` na config (`E0→PL`, `P1→PPL`, `SP1→PD`, …).
- **Live**: `GET /v4/matches?date=YYYY-MM-DD` — **um único pedido devolve os jogos do dia de todas as competições do plano**, com `score` e `status` (`IN_PLAY`/`PAUSED`/`FINISHED`). Crítico para o polling caber na quota de 10 pedidos/minuto do free tier. Mapeamento de estados: `IN_PLAY→LIVE`, `PAUSED→PAUSED`, `TIMED/SCHEDULED→SCHEDULED`, etc.
- Nomes de equipa: preferir `shortName` (mesma razão documentada no provider existente — está mais perto das grafias canónicas).

### `FlashscoreLiveProvider` — `flashscore_live.py`

**Adapter fino** sobre o cliente Flashscore existente (`src/scrapers/flashscore/scraper.py`, `http_client.py`, `playwright_client.py`) — não se escreve um scraper novo, adapta-se o que já existe para extrair resultado e minuto dos jogos do dia, usando os slugs já configurados em `scrapers.flashscore.leagues`. Só é consultado quando a API devolve vazio ou falha (é o 2º da cadeia). Fica atrás de um flag `enabled` próprio na config — dado o histórico de agosto, o sistema **funciona sem ele**.

### Cadeias — `chained.py`

`ChainedHistoryProvider` e `ChainedLiveProvider`, cópia do contrato de `src/scrapers/european/chained.py`: tenta por ordem, exceção de um provider é reportada e salta-se ao próximo, **primeiro resultado não-vazio ganha** (nunca merge — fundir exigiria deduplicar entre grafias não-canónicas, que é exatamente onde este projeto se recusa a adivinhar). `last_source` exposto para diagnóstico e devolvido na resposta da API.

### `LiveResultsTracker` — `live_tracker.py`

O "tempo real" é **polling on-demand com cache TTL**, não uma tarefa em background:

- `get_snapshot()` → se o snapshot em cache tem menos de `poll_interval_seconds`, devolve-o; senão consulta a cadeia live, calcula o diff e guarda o novo snapshot.
- O diff entre snapshots produz `MatchEvent`s (golo marcado, jogo começou/terminou) — devolvidos no payload para o frontend poder destacar mudanças.
- **Porquê on-demand em vez de loop em background**: só gasta quota quando alguém está a ver; não introduz asyncio/threads (que por §4.4 exigiriam autorização do utilizador e mais superfície de teste); com TTL de 60s o comportamento observável é idêntico a um poller contínuo. Um refresh forçado nunca fura o TTL (proteção da quota).

### Contratos dos endpoints

```
GET /live?leagues=E0,P1
→ 200 {
    "fetched_at": "2026-08-09T15:00:00Z",
    "source": "football-data.org",
    "stale": false,                      // true se o snapshot servido é de cache expirada
    "matches": [ { "league": "P1", "home_team": "Sporting CP",
                   "away_team": "FC Porto", "status": "live",
                   "minute": "67'", "home_goals": 2, "away_goals": 1, ... } ],
    "events": [ { "type": "goal", "match": ..., "detected_at": ... } ]
  }

GET /history?league=P1&season=2526
→ 200 { "league": "P1", "season": "2526", "source": "local-corpus",
        "matches": [ ...status=finished, ordenados por data... ] }
→ 422 liga/época não configurada (mensagem explícita, nunca lista vazia silenciosa)
```

### Configuração — `config.yaml` + Pydantic

```yaml
results:
  enabled: true
  history_provider_order: [local_corpus, football_data]
  live_provider_order: [football_data, flashscore]
  live:
    poll_interval_seconds: 60
    stale_after_seconds: 300      # a partir daqui a resposta marca stale: true
  cache_dir: "datasets/cache/results"
  service:                        # v2: o próprio microserviço
    api_key_env: "RESULTS_SERVICE_API_KEY"
  providers:
    football_data:
      enabled: true
      base_url: "https://api.football-data.org/v4"
      api_key_env: "FOOTBALL_DATA_API_KEY"   # env var já usada no projeto
      timeout: 15
      competitions:            # liga interna -> código football-data
        E0: "PL"
        SP1: "PD"
        D1: "BL1"
        I1: "SA"
        F1: "FL1"
        N1: "DED"
        P1: "PPL"
        CL: "CL"
    flashscore:
      enabled: true            # false desliga o fallback por completo
      # reutiliza scrapers.flashscore.* (slugs, base_url) já existentes

results_gateway:                 # v2: lado da app principal
  base_url_env: "RESULTS_SERVICE_URL"
  api_key_env: "RESULTS_SERVICE_API_KEY"
  timeout: 10
```

`config_loader.py` ganha `ResultsConfig` e `ResultsGatewayConfig` (Pydantic) validadas no arranque — chaves de API só via env var (o nome da variável na config, nunca o valor), como em `european.providers`.

---

## Design Decisions

- **Abstractions**: `HistoryProvider` / `LiveResultsProvider` / `ResultsRepository` (v1) + `ResultsGateway` (lado da app) — a fronteira HTTP fica atrás de uma interface, substituível por in-process.
- **Patterns**: Strategy (providers), Chain of Responsibility, Adapter (Flashscore), Repository, TTL cache + API Gateway/proxy fino no backend e service-to-service auth por API key.
- **Configuration**: YAML + Pydantic; URLs e keys **só via nomes de env vars** na config, valores nos ambientes Render/GitHub.
- **Dependencies**: nenhuma biblioteca nova além do grupo `results` recombinando deps já usadas (Playwright já é usado pelo cliente Flashscore existente).

## Architecture Fit

O serviço novo é estruturalmente um segundo `src/backend` em miniatura (app factory, api/, service, repositório), e o seu deployment é uma cópia do pipeline já provado (ghcr.io + deploy hook). O frontend continua a falar só com o backend principal — não há CORS novo nem exposição direta do serviço ao browser. A biblioteca `src/scrapers/results/` fica no mesmo repo, testável com a mesma suite, mas só a imagem do serviço a embarca.

## Trade-offs & Alternatives

| Approach | Pros | Cons | Chosen? |
|----------|------|------|---------|
| Microserviço dedicado no mesmo repo (monorepo) | Deploys/escala independentes; Playwright fora da imagem principal; falhas isoladas | 2º serviço para operar; latência de um hop HTTP; contrato entre serviços para manter | ✅ |
| Tudo no backend principal (v1) | Menos infra | Imagem principal engorda com Chromium; scraping pode afetar a app | ❌ (revisto pelo utilizador) |
| Repositório separado | Isolamento total | Duplicar config/aliases/CI; fricção de desenvolvimento | ❌ |
| Backend importa a biblioteca diretamente como fallback do gateway | Resiliência extra | Recoloca as deps de scraping na imagem principal — anula a separação | ❌ |
| Serviço exposto diretamente ao frontend | Sem hop no backend | CORS + auth de utilizador duplicada; service key no browser | ❌ |
| Só football-data.org (sem fallback) | Mínimo código | ~12 competições; sem plano B | ❌ |
| Só scraping Flashscore | Cobertura máxima | Frágil, contra ToS, já revertido em agosto | ❌ |
| Poller em background / WebSocket | Snapshot quente / UX melhor | Quota 24/7, mais infra; adiável sem mudar a API interna | ❌ |

## Test Strategy

- **Phase 1**: providers, cadeias, tracker e config do v1 (parsing de payloads reais gravados, mapeamento de estados, TTL com relógio injetado, diff de golos/estados, validação de `ResultsConfig`) + `HttpResultsGateway` contra transport fake (200, 401, timeout, JSON inválido); auth do serviço (sem key → 401; key errada → 401; recusa de arranque sem env); endpoints do serviço com cadeias fake; proxy do backend com gateway fake (503 quando o serviço falha). Config explícita em cada teste (§7.3).
- **Phase 3**: adiados/cancelados; prolongamento e penáltis; resposta vazia vs `uncovered` (403/404 não repetido); timeout com retry; snapshot stale com `stale: true` quando todos os providers falham; nomes não resolvidos pelo alias resolver; liga sem mapeamento; cache corrompida; quota esgotada + contrato serviço↔cliente (respostas do serviço validam contra os modelos que o gateway espera — teste de contrato partilhado, no espírito do `test_hf_space_contract.py` existente); `/health` sem auth; timeout do gateway inferior ao do provider; env vars ausentes em cada um dos lados.
- Workflows/Dockerfile: validação estática (paths do trigger cobrem todos os módulos que entram na imagem — o teste que evita a imagem stale).
- Cobertura ≥95%; `uv run pytest tests/` completo no fim.

## Checklist

- [x] Follows OOP principles (§4.2)
- [x] Replaceable architecture (§4.5) — gateway com interface; providers intercambiáveis
- [x] No hardcoded values (§6.1) — URLs/keys via env vars nomeadas na config
- [x] Configuration via YAML + Pydantic (§6.3)
- [x] Dependency injection (§6.4)
- [x] Correct project structure placement (§5) — `tests/results_service/` espelha `src/results_service/`
- [x] Test structure mirrors src (§5)

## Postura de scraping e limites

Esta secção regista, de forma explícita e deliberada, **o que este sistema faz e não faz** em relação a scraping. É uma decisão de arquitetura, não uma lacuna a preencher mais tarde.

### Princípio

Sites de resultados protegem-se com camadas que **elevam o custo do scraping, não eliminam o risco**: WAFs/anti-bot (Cloudflare, DataDome, HUMAN/PerimeterX), fingerprinting de browser (Canvas, WebGL, fontes, timing), análise comportamental (scroll, rato, regularidade dos pedidos), renderização client-side pesada, ofuscação de DOM, honeypots e CAPTCHAs adaptativos por risk-score. A resposta de engenharia correta **não é vencer estas camadas — é torná-las irrelevantes**, indo buscar os dados a uma API oficial feita para consumo por máquinas. O scraping fica como fallback honesto e desligável, nunca como via principal.

### O que ESTÁ implementado (robustez e "bom cidadão")

Medidas cujo objetivo é **reduzir carga e sobreviver a falhas transitórias**, não escapar a deteção:

| Medida | Onde | Propósito |
|--------|------|-----------|
| Rate limiting do nosso lado | `scrapers.rate_limit_seconds`, `request_timeout` (config) | Não martelar a fonte |
| Retries com backoff, só em erros de transporte | `src/scrapers/european/http.py` (`HttpTransport`) | Recuperar de ConnectionError/Timeout; nunca repetir um status HTTP (403/401/404 não muda ao repetir) |
| Memória de `uncovered` (403/404) | provider football-data | Não repetir competições que o plano não cobre — poupa quota |
| Cache com TTL (`poll_interval_seconds`) | `LiveResultsTracker` (v2) | Um pedido real por janela; volume de tráfego mínimo |
| Corpus local como 1º provider de histórico | `LocalCorpusHistoryProvider` | A maior parte do histórico nunca gera um pedido de rede |
| Fallback em dois níveis HTTP → Playwright | cliente Flashscore existente | *Lidar com* renderização client-side (ler o DOM já pintado), não contorná-la |
| Kill-switch do fallback | `results.providers.flashscore.enabled: false` | Desligar o scraping por completo sem tocar em código |
| Isolamento no microserviço | `src/results_service/` (v2) | Falha/bloqueio do scraping fica contida; a app principal devolve 503, não arrasta o resto |

### O que NÃO está implementado (e não será, sem reavaliação explícita)

Contramedidas de **evasão de deteção** — deliberadamente ausentes do código e do plano:

- Rotação de IPs / pools de proxies para contornar rate limiting.
- Spoofing de fingerprint (Canvas/WebGL/fontes), stealth plugins ou browsers "undetected".
- Simulação de comportamento humano (movimentos de rato, timing de scroll) para enganar análise comportamental.
- Resolução de CAPTCHAs (serviços de terceiros ou modelos próprios).
- Deteção/evasão de honeypots para não ser marcado.

Nenhuma destas foi construída. São exatamente a atividade que os sistemas anti-bot existem para bloquear, quebram a cada mudança do site, e — no caso do Flashscore — os ToS proíbem scraping explicitamente (ver [[european-competitions-approach]]: a rota Flashscore-only foi revertida em agosto de 2026 por esta razão). Introduzir qualquer uma delas é uma **mudança de postura**, não uma tarefa de implementação, e exige decisão consciente registada aqui.

### Caminho de escalada correto

Se um dia a cobertura da API oficial não chegar, a resposta **não é** escalar a evasão do fallback. É, por ordem:

1. Ativar o fallback Flashscore existente (sem evasão) e aceitar a sua fragilidade.
2. Adicionar outro provider **de API** à cadeia (ex.: plano pago do football-data.org, ou outra API licenciada) — a arquitetura de cadeia primeiro-não-vazio absorve-o sem tocar no serviço.
3. Negociar acesso a dados / feed licenciado.

A cadeia de providers é o que torna esta escalada barata: cada nova fonte é uma classe nova por trás da mesma interface, e a ordem em `*_provider_order` decide a prioridade.

## Riscos conhecidos

1. **Free tier do football-data.org**: 10 pedidos/min e histórico limitado em épocas antigas — mitigado pelo corpus local como 1º provider de histórico e pelo endpoint `/v4/matches` único para o live.
2. **Fallback Flashscore**: HTML pode mudar e os ToS do site proíbem scraping — fallback opcional com kill-switch na config; o sistema é plenamente funcional sem ele.
3. **Live com ~1 min de atraso** no free tier da API — aceitável para o polling de 60s; minuto-a-minuto real fica para o fallback Flashscore.
4. **Drift de contrato** entre serviço e gateway — mitigado pelos modelos Pydantic partilhados e teste de contrato.
5. **Cold start no Render free tier**: se o serviço adormecer, o primeiro pedido paga o arranque — o proxy devolve 503 explícito e o frontend pode re-tentar.
6. **Chromium na imagem** (~400 MB): build mais lento no workflow novo; só afeta o serviço dedicado.
