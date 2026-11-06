"""Fixtures shared by the test modules."""

from __future__ import annotations

import pytest

from svmorph import backend


@pytest.fixture(params=backend.BACKENDS)
def each_backend(request):
    """Run the test once under every backend that is installed here.

    The backend that was in use is put back afterwards, so a test that does not ask for this
    fixture runs under whatever ``SVMORPH_BACKEND`` chose, whichever tests ran before it.
    """
    name = request.param
    if name not in backend.available_backends():
        pytest.skip(f"{name} is not installed")
    previous = backend.get_backend()
    backend.set_backend(name)
    yield name
    backend.set_backend(previous)
