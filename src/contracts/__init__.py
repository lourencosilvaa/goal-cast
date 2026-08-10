"""Wire contracts shared between separately deployed services.

Nothing here imports anything else in the project. That is the point: these
modules are copied into *both* Docker images, so a dependency on a provider,
a database client or FastAPI would drag that dependency into an image whose
job does not need it.

A shared module rather than two hand-kept copies because the alternative is
drift — the sort where one side starts sending a field the other silently
drops, and the bug surfaces as missing data rather than as an error.
"""
