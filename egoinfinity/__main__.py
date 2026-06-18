"""CLI dispatch for `python -m egoinfinity ...`.

Subcommands (added incrementally as phases progress):
    process     run pipeline end-to-end / resume
    status      inspect artifact dir state
    run         single-stage explicit invocation
    import      legacy pkl → artifact dir conversion
    filter      visual filter on a video list
"""
from __future__ import annotations
import sys


def _lazy_process(argv):
    from .cli.process import main as f
    return f(argv)


def _lazy_status(argv):
    from .cli.status import main as f
    return f(argv)


def _lazy_run(argv):
    from .cli.run import main as f
    return f(argv)


def _lazy_filter(argv):
    from .cli.filter import main as f
    return f(argv)


def _lazy_import(argv):
    from .cli.import_pkl import main as f
    return f(argv)


_SUBCOMMANDS = {
    "process": _lazy_process,
    "status": _lazy_status,
    "run": _lazy_run,
    "filter": _lazy_filter,
    "import": _lazy_import,
}

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    cmd, *rest = argv
    if cmd not in _SUBCOMMANDS:
        print(f"unknown subcommand: {cmd}\n", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2
    return _SUBCOMMANDS[cmd](rest)


if __name__ == "__main__":
    sys.exit(main())
