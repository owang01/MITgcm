#!/usr/bin/env python3
"""Sanity-check plot: parse run/STDOUT.0000's per-step %MON monitor output
and plot global diagnostics (Theta/Salt/Eta/U/V extrema and means) vs.
model time, to visually confirm the time-stepping itself is correct (one
day per step, per deltaTClock=86400 in input/data) and that nothing is
blowing up step to step.

This is a diagnostic over the REAL V4r6-derived initial condition run
(see HANDOFF.md's "V4r6-faithfulness audit" section) -- distinct from and
does not overwrite T_step0_vs_step10.png/Eta_step0_vs_step10.png/
S_step0_vs_step10.png, which were built against this config's earlier,
flat analytic initial condition and are now stale for that reason (not
because anything is wrong with them -- they still correctly describe the
runs they were made from, just with different starting data).
"""
import os
import re

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = os.path.abspath(os.path.join(HERE, "..", "run"))
STDOUT_PATH = os.path.join(RUN_DIR, "STDOUT.0000")

MON_FIELDS = [
    "time_tsnumber", "time_secondsf",
    "dynstat_theta_max", "dynstat_theta_min", "dynstat_theta_mean",
    "dynstat_salt_max", "dynstat_salt_min", "dynstat_salt_mean",
    "dynstat_eta_max", "dynstat_eta_min",
    "dynstat_uvel_max", "dynstat_vvel_max",
]

MON_LINE_RE = re.compile(r"%MON\s+(\S+)\s*=\s*([-+0-9.EeDd]+)")


def parse_monitor(path):
    """Return dict field_name -> list of float values, one per step, in
    the order they appear in the file (MITgcm prints one full %MON block
    per step, fields always in the same order within a block)."""
    data = {name: [] for name in MON_FIELDS}
    with open(path) as fh:
        for line in fh:
            m = MON_LINE_RE.search(line)
            if not m:
                continue
            name, val_str = m.group(1), m.group(2)
            if name not in data:
                continue
            data[name].append(float(val_str.replace("D", "E").replace("d", "e")))
    n = len(data["time_tsnumber"])
    for name, vals in data.items():
        assert len(vals) == n, (
            f"{name}: found {len(vals)} values, expected {n} (one per step) -- "
            f"STDOUT.0000 may be from an incomplete/different run"
        )
    return data, n


def main():
    data, nsteps = parse_monitor(STDOUT_PATH)
    print(f"Parsed {nsteps} monitor steps from {STDOUT_PATH}")

    tstep = data["time_tsnumber"]
    tsec = data["time_secondsf"]
    tdays = [s / 86400.0 for s in tsec]

    # Time-stepping sanity: confirm exactly 1 day per step (deltaTClock).
    dt_days = [tdays[i + 1] - tdays[i] for i in range(nsteps - 1)]
    print(f"Per-step dt (days): {dt_days}")
    assert all(abs(d - 1.0) < 1e-9 for d in dt_days), (
        f"Time-stepping check FAILED: expected exactly 1.0 day per step "
        f"(deltaTClock=86400 in input/data), got {dt_days}"
    )
    assert tstep == list(range(nsteps)), (
        f"time_tsnumber is not a simple 0..{nsteps - 1} sequence: {tstep}"
    )
    print(f"Time-stepping check PASSED: {nsteps} steps, exactly 1.0 day apart, "
          f"tsnumber running 0..{nsteps - 1}.")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(tdays, data["dynstat_theta_max"], "-o", color="firebrick", label="Theta max")
    ax.plot(tdays, data["dynstat_theta_mean"], "-o", color="darkorange", label="Theta mean")
    ax.plot(tdays, data["dynstat_theta_min"], "-o", color="steelblue", label="Theta min")
    ax.set_ylabel("Theta (degC)")
    ax.legend(loc="center left", fontsize=8)
    ax.set_title("LLC30 smoke test (real V4r6-derived IC): global Theta/Salt/Eta vs. model time")

    ax = axes[1]
    ax.plot(tdays, data["dynstat_salt_max"], "-o", color="firebrick", label="Salt max")
    ax.plot(tdays, data["dynstat_salt_mean"], "-o", color="darkorange", label="Salt mean")
    ax.plot(tdays, data["dynstat_salt_min"], "-o", color="steelblue", label="Salt min")
    ax.set_ylabel("Salt (psu)")
    ax.legend(loc="center left", fontsize=8)

    ax = axes[2]
    ax.plot(tdays, data["dynstat_eta_max"], "-o", color="firebrick", label="Eta max (m)")
    ax.plot(tdays, data["dynstat_eta_min"], "-o", color="steelblue", label="Eta min (m)")
    ax2 = ax.twinx()
    ax2.plot(tdays, data["dynstat_uvel_max"], "--s", color="seagreen", label="Uvel max (m/s)")
    ax2.plot(tdays, data["dynstat_vvel_max"], "--s", color="darkviolet", label="Vvel max (m/s)")
    ax.set_xlabel("Model time (days since 1992-01-01)")
    ax.set_ylabel("Eta (m)")
    ax2.set_ylabel("U/V max (m/s)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_xticks(range(nsteps))

    fig.tight_layout()
    out_path = os.path.join(HERE, "monitor_timeseries.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
