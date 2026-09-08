"""Build an allowlisted deployment archive without local configuration or runtime data."""
from pathlib import Path
import hashlib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    paths = [root / name for name in ("plugin.py", "_manifest.json", "requirements.txt", "requirements-dev.txt", "README.md", "LICENSE", "icon.jpg", "config.example.toml")]
    for folder, patterns in {"rh_generic_lib": ["*.py"], "prompt": ["*.txt", "*.md"],
                             "docs": ["*.md", "*.png", "*.jpg"], "examples": ["*.toml"], "tests": ["*.py"], "scripts": ["build_test_package.py"]}.items():
        for pattern in patterns:
            paths.extend((root / folder).rglob(pattern))
    output = root / "dist" / "runninghub-workflow-adapter-1.2.0-test.zip"
    output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            if path.is_file():
                archive.write(path, "runninghub-workflow-adapter/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith("/config.toml") or "/.git/" in name for name in archive.namelist())
    print(output)
    print("SHA256:", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
