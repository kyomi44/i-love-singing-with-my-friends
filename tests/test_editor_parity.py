"""The browser editor re-implements the layout in JS; make sure it matches the Python renderer."""

import json
import random
import re
import shutil
import subprocess
from importlib import resources

import pytest

from deepcuts.stages.render.layout import schedule

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")


def _js_layout() -> str:
    html = resources.files("deepcuts.web").joinpath("player.html").read_text()
    m = re.search(r"// <layout>.*?\n(.*?)// </layout>", html, re.S)
    assert m, "layout markers missing from player.html"
    return m.group(1)


def _random_spans(rng: random.Random, n: int) -> list[tuple[float, float]]:
    t, spans = rng.uniform(0, 8), []
    for _ in range(n):
        t += rng.choice([0.1, 0.3, 0.8, 2.0, 9.0]) * rng.random()
        dur = rng.uniform(0.6, 5.0)
        spans.append((round(t, 3), round(t + dur, 3)))
        t += dur
    return spans


def test_js_schedule_matches_python():
    rng = random.Random(7)
    cases = [_random_spans(rng, rng.randint(1, 40)) for _ in range(40)]
    lead_in, hold = 1.2, 0.4
    harness = f"""
let song = {{ render: {{ lead_in: {lead_in}, hold: {hold}, highlight_lead_ms: 0 }} }};
let ov = {{ global_offset_ms: 0, line_shift_ms: {{}} }};
let lines, sched;
{_js_layout()}
const out = [];
for (const spans of {json.dumps(cases)}) {{
  lines = spans.map(([s, e]) => ({{ words: [{{ start: s, end: e }}] }}));
  schedule();
  out.push({{ rows: sched.rows, appear: sched.appear, disappear: sched.disappear, fresh: sched.fresh }});
}}
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "-e", harness], capture_output=True, text=True, check=True)
    js = json.loads(res.stdout)
    for spans, got in zip(cases, js, strict=True):
        want = schedule(spans, lead_in, hold)
        assert got["rows"] == want.rows
        assert got["fresh"] == want.fresh
        assert got["appear"] == pytest.approx(want.appear)
        assert got["disappear"] == pytest.approx(want.disappear)
