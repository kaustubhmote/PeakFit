import pytest

from peakfit.fit.fitting import _resolve_worker_count


@pytest.mark.parametrize(
    ("requested", "cluster_count", "cpu_count", "expected"),
    [
        (-1, 1, 8, 1),
        (-1, 3, 8, 3),
        (8, 3, 16, 3),
        (2, 5, 16, 2),
        (0, 0, 8, 1),
    ],
)
def test_worker_count_is_capped_by_clusters(
    monkeypatch: pytest.MonkeyPatch,
    requested: int,
    cluster_count: int,
    cpu_count: int,
    expected: int,
) -> None:
    monkeypatch.setattr("peakfit.fit.fitting.multiprocessing.cpu_count", lambda: cpu_count)

    assert _resolve_worker_count(requested, cluster_count) == expected
