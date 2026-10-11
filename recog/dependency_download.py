"""Verified, resumable downloads of the two main PyTorch wheels."""

from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
import hashlib
import html
import json
import re
import subprocess
import time


def wheel_urls(python: str, build: str) -> list[tuple[str, str]]:
    result = subprocess.run(
        [python, "-c", "import json,sys,platform;print(json.dumps([sys.version_info[:2],sys.platform,platform.machine()]))"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    version, system, machine = json.loads(result.stdout)
    tag = f"cp{version[0]}{version[1]}"
    arch = "aarch64" if machine.lower() in {"aarch64", "arm64"} else "x86_64"
    if system == "win32" and arch == "x86_64":
        platform = "win_amd64"
    elif system == "linux":
        platform = f"manylinux_2_28_{arch}"
    else:
        raise ValueError("当前平台暂无匹配的官方安装包，请使用自定义环境")
    wheels = []
    for package, version in (("torch", "2.11.0"), ("torchvision", "0.26.0")):
        index = f"https://download.pytorch.org/whl/{build}/{package}/"
        with urlopen(index, timeout=30) as response:
            listing = response.read().decode("utf-8")
        expected = f"{package}-{version}+{build}-{tag}-{tag}-{platform}.whl"
        for link in re.findall(r'href="([^"]+)"', listing):
            link = html.unescape(urljoin(index, link))
            parts = urlsplit(link)
            if unquote(parts.path.rsplit("/", 1)[-1]) != expected:
                continue
            sha = parts.fragment.removeprefix("sha256=")
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("官方安装包缺少 SHA256，无法核验")
            # Use the official CDN host; some networks deny the index's R2 host.
            wheels.append(("https://download.pytorch.org" + parts.path, sha))
            break
        else:
            raise ValueError(f"未找到匹配的官方安装包：{expected}")
    return wheels


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url: str, sha: str, directory: Path, update) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if not name.endswith(".whl") or Path(name).name != name:
        raise ValueError("无效安装包文件名")
    final = directory / name
    partial = directory / (name + ".part")
    if final.is_symlink() or partial.is_symlink():
        raise ValueError("安装包缓存不能是符号链接")
    if final.exists():
        if file_sha(final) == sha:
            size = final.stat().st_size
            update(phase="download", file=name, downloaded=size, total=size, percent=100, cached=True)
            return final
        final.unlink()
    for attempt in range(2):
        offset = partial.stat().st_size if partial.exists() else 0
        update(phase="download", file=name, downloaded=offset, total=0, percent=None, cached=False)
        headers = {"User-Agent": "MaiBot-self-recognition", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            response = urlopen(Request(url, headers=headers), timeout=30)
        except Exception as exc:
            if getattr(exc, "code", None) == 416 and offset:
                if file_sha(partial) == sha:
                    partial.replace(final)
                    update(phase="download", file=name, downloaded=offset, total=offset, percent=100)
                    return final
                partial.unlink()
                continue
            raise
        with response:
            content_range = response.headers.get("Content-Range", "")
            if response.status == 206 and content_range.startswith(f"bytes {offset}-"):
                total = int(content_range.rsplit("/", 1)[1])
                mode = "ab"
            else:
                # A server that ignores Range requires a clean restart.
                offset = 0
                total = int(response.headers.get("Content-Length", "0"))
                mode = "wb"
            downloaded = offset
            last = 0.0
            with partial.open(mode) as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last >= 0.5:
                        stream.flush()
                        update(phase="download", file=name, downloaded=downloaded, total=total,
                               percent=round(downloaded / total * 100, 1) if total else None)
                        last = now
            update(phase="verify_download", file=name, downloaded=downloaded, total=total,
                   percent=100 if total and downloaded == total else None)
        if file_sha(partial) == sha:
            partial.replace(final)
            return final
        partial.unlink()
    raise ValueError(f"安装包 SHA256 校验失败：{name}")


def prepare_wheels(root: Path, python: str, build: str, update) -> list[Path]:
    update(phase="resolve", file="", downloaded=0, total=0, percent=None)
    directory = root / "dependencies" / "downloads"
    if directory.is_symlink() or not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("安装包缓存路径不属于本插件")
    wheels = wheel_urls(python, build)
    result = []
    for number, (url, sha) in enumerate(wheels, 1):
        update(package_number=number, package_count=len(wheels))
        result.append(download(url, sha, directory, update))
    return result
