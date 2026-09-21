"""Fine-tune RT-DETRv2 and evaluate its best checkpoint on the test split."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    """Parse a single-GPU RT-DETRv2 experiment."""
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETRv2, then evaluate its best checkpoint on the test split."
    )
    parser.add_argument("config", help="Training configuration path relative to the repository root.")
    parser.add_argument(
        "--pretrained",
        required=True,
        help="Checkpoint passed to tools/train.py -t for pretrained fine-tuning.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the training outputs and final test metrics.",
    )
    parser.add_argument(
        "--test-config",
        help="Test configuration path. Defaults to the training config with _test appended to its stem.",
    )
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value for this single-GPU run.")
    parser.add_argument("--seed", type=int, help="Optional random seed forwarded to both training and test.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision during training.")
    return parser.parse_args()


def resolve_path(path):
    """Resolve a repository-relative path without requiring the caller's current directory."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def run(command, env, log_path=None):
    """Run a command, teeing combined output to a log file when requested."""
    print("+", " ".join(command), flush=True)
    if log_path is None:
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        return

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def remove_extra_checkpoints(output_dir):
    """Remove saved model checkpoints after successful testing, retaining best.pth only."""
    deleted = []
    for pattern in ("last.pth", "checkpoint*.pth"):
        for checkpoint in output_dir.glob(pattern):
            checkpoint.unlink()
            deleted.append(checkpoint.name)
    (output_dir / "deleted_checkpoints.log").write_text(
        "\n".join(deleted) + ("\n" if deleted else ""), encoding="utf-8"
    )


def main():
    """Fine-tune the model and evaluate its best checkpoint without splitting result directories."""
    args = parse_args()
    config = resolve_path(args.config)
    if not config.is_file():
        raise FileNotFoundError(f"Training config not found: {config}")

    test_config = resolve_path(args.test_config) if args.test_config else config.with_name(
        f"{config.stem}_test{config.suffix}"
    )
    if not test_config.is_file():
        raise FileNotFoundError(
            f"Test config not found: {test_config}. Pass --test-config explicitly or add the inferred config."
        )

    pretrained = resolve_path(args.pretrained)
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")

    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device

    train_command = [
        sys.executable,
        "tools/train.py",
        "-c",
        str(config),
        "-t",
        str(pretrained),
        "--output-dir",
        str(output_dir),
    ]
    if args.seed is not None:
        train_command.extend(["--seed", str(args.seed)])
    if args.amp:
        train_command.append("--use-amp")
    run(train_command, env)

    checkpoint = output_dir / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training completed but best checkpoint was not found: {checkpoint}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "test_checkpoint.txt").write_text(f"{checkpoint}\n", encoding="utf-8")
    if args.seed is not None:
        (output_dir / "seed.txt").write_text(f"{args.seed}\n", encoding="utf-8")

    test_command = [
        sys.executable,
        "tools/train.py",
        "-c",
        str(test_config),
        "-r",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--test-only",
    ]
    if args.seed is not None:
        test_command.extend(["--seed", str(args.seed)])
    run(test_command, env, output_dir / "test_metrics.log")
    remove_extra_checkpoints(output_dir)


if __name__ == "__main__":
    main()
