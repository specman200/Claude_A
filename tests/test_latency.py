"""Latency accounting: the numbers the HUD and the CSV report."""

import csv

import pytest

from ppe.latency import STAGES, Cycle, Metrics, Profiler, _pct, now


def test_stamps_partition_the_cycle():
    cycle = Cycle(now())
    for stage in ("wait", "inference"):
        cycle.stamp(stage)
    total = cycle.finish()
    assert set(cycle.stages) == {"wait", "inference"}
    # Stamps measure disjoint spans, so they cannot exceed the whole.
    assert sum(cycle.stages.values()) <= total + 1e-6
    assert total > 0


def test_repeated_stamps_accumulate():
    cycle = Cycle(now())
    cycle.stamp("relay")
    first = cycle.stages["relay"]
    cycle.stamp("relay")
    assert cycle.stages["relay"] > first


def test_merge_adopts_external_timings_without_double_counting():
    cycle = Cycle(now())
    cycle.merge({"preprocess": 5.0, "inference": 20.0})
    cycle.stamp("logic")
    assert cycle.stages["preprocess"] == 5.0
    assert cycle.stages["logic"] < 5.0  # the merge reset the clock


def test_percentiles_interpolate():
    data = [float(i) for i in range(1, 101)]
    assert _pct(data, 0.50) == pytest.approx(50.5)
    assert _pct(data, 0.95) == pytest.approx(95.05)
    assert _pct([7.0], 0.95) == 7.0


def test_rolling_window_forgets_old_samples():
    m = Metrics(window=3)
    for value in (100.0, 1.0, 2.0, 3.0):
        m.add("inference", value)
    stats = m.stats("inference")
    assert stats["n"] == 3 and stats["max"] == 3.0


def test_stats_of_an_unseen_series_are_zero():
    assert Metrics().stats("nothing") == {"n": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}


def test_measure_times_a_block():
    m = Metrics()
    with m.measure("render"):
        sum(range(10000))
    assert m.stats("render")["n"] == 1


def test_record_writes_one_csv_row_per_cycle(tmp_path):
    path = tmp_path / "logs" / "latency.csv"
    m = Metrics(window=10, csv_path=path)
    cycle = Cycle(now() - 0.010)  # a frame grabbed 10 ms ago
    cycle.merge({"inference": 12.5})
    cycle.finish()
    m.record("Line A", cycle)
    m.close()

    rows = list(csv.reader(path.open()))
    assert rows[0] == ["wall_time", "camera", *STAGES, "end_to_end_ms"]
    assert rows[1][1] == "Line A"
    assert float(rows[1][2 + STAGES.index("inference")]) == pytest.approx(12.5)
    assert float(rows[1][-1]) == pytest.approx(10.0, abs=5.0)


def test_csv_appends_without_repeating_the_header(tmp_path):
    path = tmp_path / "latency.csv"
    for _ in range(2):
        m = Metrics(csv_path=path)
        cycle = Cycle(now())
        cycle.finish()
        m.record("cam", cycle)
        m.close()
    rows = list(csv.reader(path.open()))
    assert len(rows) == 3 and rows[0][0] == "wall_time"


def test_report_shows_every_stage_and_the_total():
    m = Metrics()
    cycle = Cycle(now())
    cycle.merge(dict.fromkeys(STAGES, 4.0))
    cycle.finish()
    m.record("cam", cycle)
    report = m.report()
    for stage in STAGES:
        assert stage in report
    assert "end-to-end" in report and "100.0%" in report


def test_report_is_safe_before_any_data():
    assert "stage" in Metrics().report()


def test_profiler_writes_a_dump_and_names_hotspots(tmp_path):
    out = tmp_path / "profile.prof"
    prof = Profiler(out, top=5)
    prof.start()
    _burn_cpu()
    report = prof.stop()

    assert out.exists() and out.stat().st_size > 0
    assert "hotspots" in report and "cumulative" in report
    assert "_burn_cpu" in report


def test_profiler_captures_worker_threads_not_just_the_caller(tmp_path):
    """The real work happens off the main thread, so it must be profiled too."""
    import threading

    prof = Profiler(tmp_path / "p.prof", top=10)

    def worker():
        prof.start()
        _burn_cpu()
        prof.stop_thread()

    thread = threading.Thread(target=worker, name="worker")
    thread.start()
    thread.join()

    report = prof.stop()  # main thread never started one; only the worker's
    assert "_burn_cpu" in report


def test_profiler_without_any_samples_is_silent(tmp_path):
    assert Profiler(tmp_path / "p.prof").stop() == ""


def test_profiler_context_manager_still_works(tmp_path, capsys):
    with Profiler(tmp_path / "ctx.prof", top=3):
        _burn_cpu()
    assert "_burn_cpu" in capsys.readouterr().out


def _burn_cpu():
    return sum(i * i for i in range(50000))
