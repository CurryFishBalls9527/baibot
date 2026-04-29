"""Helpers for installing the scheduler as a macOS launch agent."""

import os
import plistlib
import subprocess
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_LABEL = "com.tradingagents.scheduler"
WATCHDOG_LABEL = "com.tradingagents.watchdog"


def _launch_agent_path(label: str = DEFAULT_LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _run_launchctl(args: Iterable[str], check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"launchctl {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def install_launch_agent(
    repo_root: str,
    python_bin: str,
    mode: str = "swing",
    symbols: Optional[list[str]] = None,
    label: str = DEFAULT_LABEL,
    experiment: Optional[str] = None,
) -> Path:
    repo_path = Path(repo_root).resolve()
    # Preserve the venv entrypoint path. Resolving it can collapse the symlink
    # to the base interpreter and drop the virtualenv's site-packages.
    python_path = Path(python_bin)
    plist_path = _launch_agent_path(label)

    log_dir = repo_path / "results" / "service_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    program_args = [
        str(python_path),
        str(repo_path / "run_trading.py"),
        "schedule",
        "--mode",
        mode,
    ]
    if symbols:
        program_args.extend(["--symbols", ",".join(symbols)])
    if experiment:
        experiment_path = Path(experiment)
        if not experiment_path.is_absolute():
            experiment_path = (repo_path / experiment_path).resolve()
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment YAML not found: {experiment_path}")
        program_args.extend(["--experiment", str(experiment_path)])

    plist = {
        "Label": label,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(repo_path),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "automation_service.out.log"),
        "StandardErrorPath": str(log_dir / "automation_service.err.log"),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
        },
    }

    plist_path.parent.mkdir(parents=True, exist_ok=True)

    if plist_path.exists():
        _run_launchctl(["bootout", f"gui/{os.getuid()}", str(plist_path)], check=False)

    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)

    _run_launchctl(["bootstrap", f"gui/{os.getuid()}", str(plist_path)])
    return plist_path


def uninstall_launch_agent(label: str = DEFAULT_LABEL) -> Path:
    plist_path = _launch_agent_path(label)
    if plist_path.exists():
        _run_launchctl(["bootout", f"gui/{os.getuid()}", str(plist_path)], check=False)
        plist_path.unlink()
    return plist_path


def install_watchdog_agent(
    repo_root: str,
    python_bin: str,
    label: str = WATCHDOG_LABEL,
) -> Path:
    """Install the watchdog as a separate launchd job."""
    repo_path = Path(repo_root).resolve()
    python_path = Path(python_bin)
    plist_path = _launch_agent_path(label)

    log_dir = repo_path / "results" / "service_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    program_args = [
        str(python_path),
        str(repo_path / "run_watchdog.py"),
    ]

    plist = {
        "Label": label,
        "ProgramArguments": program_args,
        "WorkingDirectory": str(repo_path),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ProcessType": "Background",
        "StandardOutPath": str(log_dir / "watchdog.out.log"),
        "StandardErrorPath": str(log_dir / "watchdog.err.log"),
        "EnvironmentVariables": {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONUNBUFFERED": "1",
        },
    }

    plist_path.parent.mkdir(parents=True, exist_ok=True)

    if plist_path.exists():
        _run_launchctl(["bootout", f"gui/{os.getuid()}", str(plist_path)], check=False)

    with plist_path.open("wb") as handle:
        plistlib.dump(plist, handle)

    _run_launchctl(["bootstrap", f"gui/{os.getuid()}", str(plist_path)])
    return plist_path


def uninstall_watchdog_agent(label: str = WATCHDOG_LABEL) -> Path:
    return uninstall_launch_agent(label)
