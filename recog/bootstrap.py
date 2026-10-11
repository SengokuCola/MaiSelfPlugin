"""Start a worker with plugin dependencies before the base interpreter packages."""

from pathlib import Path
import runpy
import site
import sys


def main() -> None:
    overlay = str(Path(sys.argv[1]).resolve())
    # Process .pth files as a normal site directory, then give its packages
    # priority. This happens only in this child, never in the host/Runner.
    before = list(sys.path)
    sys.path.insert(0, overlay)
    site.addsitedir(overlay)
    added = [path for path in sys.path if path not in before and path != overlay]
    sys.path[:] = [overlay, *added, *[path for path in before if path != overlay]]
    mode, target, *args = sys.argv[2:]
    sys.argv = [target, *args]
    if mode == "-m":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    elif mode == "-c":
        exec(compile(target, "<overlay-probe>", "exec"), {"__name__": "__main__"})
    else:
        raise ValueError("Unsupported worker bootstrap mode")


if __name__ == "__main__":
    main()
