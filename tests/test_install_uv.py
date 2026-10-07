import importlib
from unittest import mock

import pytest

from cluv.cache import ProjectStateOnCluster

sync_module = importlib.import_module("cluv.cli.sync")


def _remote(*outputs: str) -> mock.Mock:
    return mock.Mock(
        hostname="cluster", run=mock.AsyncMock(), get_output=mock.AsyncMock(side_effect=outputs)
    )


@pytest.mark.asyncio
async def test_install_uv_never_downgrades(monkeypatch):
    monkeypatch.setattr(sync_module.subprocess, "getoutput", lambda _: "uv 0.8.11")
    remote = _remote("/bin/uv", "uv 0.12.23")
    await sync_module.install_uv(remote, ProjectStateOnCluster())
    remote.run.assert_awaited_once_with("bash -l -c 'uv self update'", hide=True)
