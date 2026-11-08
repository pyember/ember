# Contributing to Ember

## Setup

```bash
git clone https://github.com/pyember/ember.git
cd ember
uv sync --group dev --all-extras
```

## Before you open a pull request

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy                      # strict, over src/
uv run pytest tests/<area>       # CI runs the whole suite on Python 3.11 and 3.13
```

## What we look for

- [docs/design/v0.2.md](docs/design/v0.2.md) is the contract. A change to behaviour it describes
  updates it in the same pull request.
- Google Python style, `mypy --strict` clean, and functions short enough to read in one screen. No
  `Any` without a reason, no `**kwargs` bags, no duck typing with `hasattr` or `getattr`, and no use
  of another module's private names.
- Errors are raised where they are detected, with a type and message that say what happened. A
  fallback names the exception types it absorbs; nothing catches `Exception` to carry on.
- Tests assert the contract, not the current output. A bug fix comes with a test that fails without
  it. Tests never reach the network: use `ember.llm.fake.FakeProvider` with `ember.llm.using`, or
  `httpx.MockTransport` for provider code.
- Every Python example in `README.md` and `docs/*.md` runs in CI, so examples stay true.
- Commit messages follow Conventional Commits, for example `fix(xcs): ...`.
