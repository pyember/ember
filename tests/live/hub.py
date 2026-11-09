"""Reads the first examples of each named dataset from the Hugging Face Hub.

Run only by the ``live-smoke`` workflow. It needs the network and the ``data``
extra, and no credentials. The test suite never imports this file, and its
tests replace the Hub with a fake, so this is the one place a real read runs.

The process must also end: the workflow bounds this step, because a read left
unfinished at the limit has kept a process from exiting.

Usage:
    uv run python tests/live/hub.py
"""

import sys

from ember import data

EXAMPLES = 3


def main() -> int:
    for name in sorted(data.DATASETS):
        stream = data.stream(name, limit=EXAMPLES)
        examples = list(stream)
        source = stream.source
        if len(examples) != EXAMPLES:
            print(f"FAIL {name}: read {len(examples)} examples, expected {EXAMPLES}")
            return 1
        print(f"PASS {name}: {source.path} ({source.config}, {source.split}) at {source.revision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
