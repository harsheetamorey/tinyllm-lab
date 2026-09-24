"""Per-run context: output directory, seeding, device choice and run metadata."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from tinyllm.config import Config
from tinyllm.utils.env import environment_info
from tinyllm.utils.repro import detect_device, resolve_amp_dtype, set_seed

DEFAULT_RESULTS_ROOT = Path("results/raw")


@dataclass(frozen=True)
class GitInfo:
    """Code version a run was produced from."""

    commit: str | None
    dirty: bool | None

    @classmethod
    def from_repo(cls, repo_dir: Path) -> "GitInfo":
        """Read HEAD and dirtiness from ``repo_dir``; fields are None outside git."""
        commit = cls._git(repo_dir, "rev-parse", "HEAD")
        if commit is None:
            return cls(commit=None, dirty=None)
        status = cls._git(repo_dir, "status", "--porcelain")
        return cls(commit=commit, dirty=bool(status))

    @staticmethod
    def _git(repo_dir: Path, *args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return out.stdout.strip()


@dataclass
class RunContext:
    """Everything a run needs to be set up and later reproduced.

    Build one with :meth:`create`, which seeds RNGs, resolves the device and
    AMP dtype, and creates ``<results_root>/<experiment_id>/``. Call
    :meth:`save_metadata` to write ``config.yaml``, ``environment.json`` and
    ``run.json`` into that directory.
    """

    config: Config
    run_dir: Path
    device: torch.device
    amp_dtype: torch.dtype | None
    git: GitInfo
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @classmethod
    def create(
        cls,
        config: Config,
        results_root: Path = DEFAULT_RESULTS_ROOT,
        exist_ok: bool = False,
        overwrite: bool = False,
        deterministic: bool = False,
    ) -> "RunContext":
        """Set up a run for ``config``.

        Raises ``FileExistsError`` if the run directory already exists, so
        earlier results are never overwritten by accident. Two explicit opt-ins:
        ``exist_ok`` reuses the directory as is (e.g. to resume), ``overwrite``
        deletes it first and starts from an empty directory.
        """
        if exist_ok and overwrite:
            raise ValueError("exist_ok and overwrite are mutually exclusive")
        run_dir = Path(results_root) / config.experiment_id
        if overwrite and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=exist_ok)

        set_seed(config.seed, deterministic=deterministic)
        device = detect_device(config.device)
        return cls(
            config=config,
            run_dir=run_dir,
            device=device,
            amp_dtype=resolve_amp_dtype(config.amp_dtype, device),
            git=GitInfo.from_repo(Path(__file__).resolve().parent),
        )

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoints (git-ignored); created on first access."""
        path = self.run_dir / "checkpoints"
        path.mkdir(exist_ok=True)
        return path

    def metadata(self) -> dict[str, Any]:
        """Return run-level facts as a JSON-serializable dict."""
        return {
            "experiment_id": self.config.experiment_id,
            "started_at": self.started_at,
            "device": str(self.device),
            "amp_dtype": str(self.amp_dtype).removeprefix("torch.") if self.amp_dtype else None,
            "git_commit": self.git.commit,
            "git_dirty": self.git.dirty,
        }

    def save_metadata(self) -> None:
        """Write the config, environment snapshot and run metadata to ``run_dir``."""
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(self.config.to_dict(), sort_keys=False)
        )
        self._write_json("environment.json", environment_info())
        self._write_json("run.json", self.metadata())

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        (self.run_dir / name).write_text(json.dumps(data, indent=2) + "\n")
