import importlib
from pathlib import PurePosixPath
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


@pytest.mark.asyncio
@pytest.mark.parametrize(("last_uv", "reruns"), [("uv 0.12.23", False), ("uv 0.8.11", True)])
async def test_run_uv_sync_reruns_when_cluster_uv_changed(monkeypatch, last_uv, reruns):
    monkeypatch.setattr(sync_module.subprocess, "getoutput", lambda _: "abc123")
    remote = _remote("uv 0.12.23")
    state = ProjectStateOnCluster(
        last_uv_sync_git_commit="abc123", last_uv_sync_uv_version=last_uv
    )
    await sync_module.run_uv_sync(remote, PurePosixPath("/project"), state)
    assert remote.run.await_count == reruns
