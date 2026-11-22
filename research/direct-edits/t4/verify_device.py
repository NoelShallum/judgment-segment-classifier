import json
from pathlib import Path
import subprocess
import time

def verify_t4():
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()
    devices = [line.split(",")[0].strip() for line in listing]
    if len(devices) != 1 or devices[0] not in ("Tesla T4", "NVIDIA T4", "T4"):
        raise RuntimeError("A single NVIDIA T4 is required. Assigned GPUs: " + str(devices))
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != devices[0]:
        raise RuntimeError("PyTorch and nvidia-smi disagree about the T4.")
    result = {"observed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "nvidia_smi":listing, "torch_cuda":torch.version.cuda,"torch_version":str(torch.__version__)}
    return result

if __name__ == "__main__":
    receipt = verify_t4()
    dest = Path("/content/attoflow-t4")
    dest.mkdir(exist_ok=True,parents=True)
    (dest / "device.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))
