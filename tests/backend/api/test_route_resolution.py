"""Route-resolution guards for the fully assembled application.

Every other API test mounts a single router in isolation, which cannot see
collisions *between* routers. Starlette matches routes in registration order,
so a permissive pattern registered early — ``GET /api/predictions/{league_code}``
— silently swallows any sibling path registered later. That is exactly how
``GET /api/predictions/teams`` became unreachable: it answered the league
handler's payload instead, and the custom-prediction team pickers stayed empty.

These tests exercise ``src.backend.main.app`` itself, so a future reordering or
a new catch-all cannot re-introduce the same class of bug unnoticed.
"""

import pytest
from starlette.routing import Match, Route

from src.backend.api.stats import get_match_stats as match_stats_endpoint
from src.backend.api.stats import get_team_stats as team_stats_endpoint
from src.backend.api.teams import get_teams as teams_endpoint


@pytest.fixture(scope="module")
def app():
    """The real application object, routers included in production order."""
    from src.backend.main import app as real_app

    return real_app


def _scope(method: str, path: str) -> dict:
    return {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "headers": [],
        "query_string": b"",
    }


def _first_matching_route(app, method: str, path: str):
    """Resolve ``path`` the way Starlette does: first full match in order."""
    for route in app.routes:
        match, _ = route.matches(_scope(method, path))
        if match == Match.FULL:
            return route
    return None


def _static_routes(app) -> list[Route]:
    """Routes whose path has no parameters, so they resolve to themselves."""
    return [
        route
        for route in app.routes
        if isinstance(route, Route) and "{" not in route.path and route.methods
    ]


class TestTeamsRouteIsReachable:

    def test_teams_route_is_registered(self, app):
        paths = [getattr(route, "path", "") for route in app.routes]
        assert "/api/teams" in paths

    def test_get_api_teams_resolves_to_the_teams_endpoint(self, app):
        route = _first_matching_route(app, "GET", "/api/teams")
        assert route is not None
        assert route.endpoint is teams_endpoint

    def test_teams_route_lives_outside_the_predictions_namespace(self, app):
        """Under /api/predictions/* it would be shadowed by {league_code}."""
        route = _first_matching_route(app, "GET", "/api/teams")
        assert not route.path.startswith("/api/predictions")


class TestStatsRoutesAreReachable:
    """The statistics endpoints sit outside /api/predictions for the same
    reason /api/teams does — see the module docstring."""

    def test_stats_routes_are_registered(self, app):
        paths = [getattr(route, "path", "") for route in app.routes]
        assert "/api/stats/team" in paths
        assert "/api/stats/match" in paths

    def test_get_team_stats_resolves_to_its_endpoint(self, app):
        route = _first_matching_route(app, "GET", "/api/stats/team")
        assert route is not None
        assert route.endpoint is team_stats_endpoint

    def test_post_match_stats_resolves_to_its_endpoint(self, app):
        route = _first_matching_route(app, "POST", "/api/stats/match")
        assert route is not None
        assert route.endpoint is match_stats_endpoint

    def test_stats_routes_live_outside_the_predictions_namespace(self, app):
        for method, path in (("GET", "/api/stats/team"), ("POST", "/api/stats/match")):
            route = _first_matching_route(app, method, path)
            assert not route.path.startswith("/api/predictions")


class TestAdminAliasRoutesAreReachable:

    def test_alias_route_is_registered(self, app):
        paths = [getattr(route, "path", "") for route in app.routes]
        assert "/api/admin/team-aliases" in paths

    def test_alias_route_is_not_shadowed_by_the_user_routes(self, app):
        """``/api/admin/users`` is static, but a future pattern could collide."""
        route = _first_matching_route(app, "GET", "/api/admin/team-aliases")
        assert route is not None
        assert route.path == "/api/admin/team-aliases"

    def test_alias_route_supports_every_review_action(self, app):
        """FastAPI registers one Route per verb, so the methods are unioned."""
        methods: set[str] = set()
        for route in app.routes:
            if getattr(route, "path", "") == "/api/admin/team-aliases":
                methods |= set(route.methods or set())
        assert {"GET", "POST", "DELETE"} <= methods


class TestNoRouteIsShadowed:

    def test_every_static_route_resolves_to_itself(self, app):
        shadowed = []
        for route in _static_routes(app):
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                resolved = _first_matching_route(app, method, route.path)
                if resolved is not route:
                    shadowed.append(
                        f"{method} {route.path} → {getattr(resolved, 'path', None)}"
                    )
        assert not shadowed, f"routes shadowed by an earlier pattern: {shadowed}"

    def test_league_code_pattern_still_serves_a_real_league_code(self, app):
        route = _first_matching_route(app, "GET", "/api/predictions/E0")
        assert route is not None
        assert route.path == "/api/predictions/{league_code}"
