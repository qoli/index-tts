# Repository Guidelines

## Project Structure & Module Organization
Core runtime logic lives in `indextts/`, which houses the CLI entry point, inference graphs, tokenizers, and utilities that wrap PyTorch, Transformers, and emotion-control tooling. Visual assets (icons, promo art) stay in `assets/`, while long-form documentation and localized guides are under `docs/`. Prepackaged demo audio, prompts, and JSON recipes live in `examples/`. Test harnesses reside in `tests/`, and helper scripts (e.g., i18n, dataset tooling) sit in `tools/`. Keep downloaded weights inside `checkpoints/` (or a sibling directory you pass via `--model_dir`) so both the CLI and `webui.py` can resolve `config.yaml`, `gpt.pth`, `s2mel.pth`, and tokenizer files.

## Build, Test, and Development Commands
- `uv sync --all-extras`: creates `.venv`, installs base runtime plus `webui` + `deepspeed` extras; rerun after editing `pyproject.toml`.
- `uv run indextts --help`: sanity-checks the console entry point defined in `pyproject.toml`.
- `uv run python webui.py --model_dir checkpoints --port 7860`: launches the Gradio demo; add `--deepspeed` or `--cuda_kernel` when GPU kernels are available.
- `uv run python tests/padding_test.py checkpoints` and `uv run python tests/regression_test.py`: smoke-test text padding logic and reference inference outputs (they expect `tests/sample_prompt.wav` and a populated checkpoint folder).

## Coding Style & Naming Conventions
Target Python ≥3.10 with 4-space indentation, type hints on public methods, and concise docstrings that describe tensor shapes or sampling strategies. Follow snake_case for functions/variables, PascalCase for classes, and SCREAMING_SNAKE_CASE for constants or environment toggles. Run `uv run ruff check` before sending patches and `uv run ruff format` to normalize layout—these match the dependencies declared in `uv.lock`.

## Testing Guidelines
Prefer deterministic seeds (`transformers.set_seed(42)`) when touching sampling code. Mirror the layout in `tests/` by storing fixtures alongside scripts, and name new tests after the scenario they guard (e.g., `duration_alignment_test.py`). Use `uv run pytest tests` for suites that integrate multiple files, and keep generated WAVs in `outputs/` (ignored by git) so reviewers can replay regressions. Mention any large checkpoints or special hardware needs in the PR description.

## Commit & Pull Request Guidelines
Recent history uses Conventional Commit prefixes (`feat:`, `fix:`, `perf:`); keep subject lines imperative and under 72 chars. Each PR should describe the user-facing change, list test commands (include `uv run ...` outputs when relevant), and link to tracking issues or benchmarks. Attach short audio comparisons or screenshots for WebUI tweaks, confirm that `uv sync --all-extras` still succeeds, and flag any new dependencies so maintainers can refresh `uv.lock`.

## Model Assets & Configuration Tips
Never commit proprietary weights. Instead, document download steps in `README.md` updates or scripts under `tools/`. Validate that `webui.py` refuses to start when mandatory files (`bpe.model`, `gpt.pth`, `config.yaml`, `s2mel.pth`, `wav2vec2bert_stats.pt`) are missing, and mention the minimum GPU/CPU requirements in contribution notes. When editing configuration YAMLs, keep defaults backwards compatible and explain breaking changes in the PR body.
