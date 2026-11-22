"""Build a private transport bundle for the user's own Colab. Includes authorized CSV."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from weight_edit import ROOT


def main():
    bundle = ROOT / "colab-private-bundle.zip"
    stage = bundle.with_suffix(".partial")
    files = [ROOT / name for name in ("colab_worker.py", "score_official_test.py", "cache_full.py", "program_trial.py",
             "encoder.py", "weight_edit.py", "role_fixtures.py", "import_official.py", "public-manifest.json",
             "references/official/test.csv")]
    kit = ROOT / "portable-edit"
    files.extend(path for path in kit.iterdir() if path.is_file())
    with ZipFile(stage, "w", ZIP_DEFLATED, compresslevel=2) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    with ZipFile(stage) as archive:
        assert archive.testzip() is None
        assert "references/official/test.csv" in archive.namelist()
    stage.replace(bundle)
    print("Prepared private remote-evaluation bundle", bundle.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
