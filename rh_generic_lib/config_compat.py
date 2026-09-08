"""Repair versionless legacy config files before Runner checks their version."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import tomllib


def ensure_config_version(path: Path, version: str) -> bool:
    """Add only the missing version, preserving configuration values and a backup."""
    if not path.is_file():
        return False  # Runner creates new configuration from the model defaults.
    original = path.read_bytes()
    content = original.decode("utf-8-sig")
    raw = tomllib.loads(content)
    section = raw.get("plugin", {})
    if not isinstance(section, dict) or str(section.get("config_version") or "").strip():
        return False

    expected = {**raw, "plugin": {**section, "config_version": version}}
    newline = "\r\n" if "\r\n" in content else "\n"
    value = json.dumps(version)
    version_line = f"config_version = {value}{newline}"
    candidates = []
    if "config_version" in section:
        empty_version = re.compile(
            r'''(?m)^([ \t]*(?:config_version|"config_version"|'config_version')[ \t]*=[ \t]*)'''
            r'''(?:"[ \t]*"|'[ \t]*'|false|0)([ \t]*(?:\#[^\r\n]*)?)(\r?$)'''
        )
        for match in empty_version.finditer(content):
            replacement = match.group(1) + value + match.group(2) + match.group(3)
            candidates.append(content[:match.start()] + replacement + content[match.end():])
    else:
        header = re.compile(r'''(?m)^[ \t]*\[[ \t]*(?:plugin|"plugin"|'plugin')[ \t]*\][ \t]*(?:\#[^\r\n]*)?(?:\r?\n|$)''')
        for match in header.finditer(content):
            separator = "" if match.group().endswith("\n") else newline
            candidates.append(content[:match.end()] + separator + version_line + content[match.end():])
        # Covers an absent [plugin] table or one implied by [plugin.some_subtable].
        candidates.append(content + newline + "[plugin]" + newline + version_line)

    patched = None
    for candidate in candidates:
        try:
            if tomllib.loads(candidate) == expected:
                patched = candidate
                break
        except tomllib.TOMLDecodeError:
            continue
    if patched is None:
        raise ValueError("请在 config.toml 的 [plugin] 内填写 config_version，当前格式无法自动修复")

    # Parsing the candidate also prevents mistaking text inside multiline prompts for a table.
    backup = path.with_name(f"{path.name}.pre-version-{time.time_ns()}.bak")
    shutil.copy2(path, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(patched.encode("utf-8-sig" if original.startswith(b"\xef\xbb\xbf") else "utf-8"))
        shutil.copymode(path, temporary)
        if path.read_bytes() != original:
            raise OSError("修复期间配置已被其他操作修改，请重新加载插件")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True
