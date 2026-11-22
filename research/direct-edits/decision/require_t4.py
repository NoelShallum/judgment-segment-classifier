"""Exit before model loading unless nvidia-smi lists only NVIDIA T4 accelerators."""
import re
import subprocess


def require_t4():
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.STDOUT, timeout=15,
        ).splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("T4 verification failed before model loading") from exc
    names = [n.strip() for n in lines if n.strip()]
    if not names or not all(re.fullmatch(r"(?:(?:NVIDIA|Tesla)\s+)*(?:T4)", n, re.I) for n in names):
        raise RuntimeError("This experiment requires T4-only hardware; nvidia-smi reported " + repr(names))
    return names


if __name__ == "__main__":
    print("nvidia-smi verified", require_t4(), flush=True)
