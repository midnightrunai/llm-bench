# Changelog

## [0.1.0] - 2026-03-27

### Added
- Initial release
- Support for OpenAI, Anthropic, Gemini, Mistral, and Groq providers
- Rich terminal table output with p50/p95 latency, token counts, cost per 1k requests
- Optional quality scoring via LLM-as-judge
- YAML config file support for batch benchmarking
- JSON output for CI/CD integration
- `llm-bench init` to generate starter config
- `llm-bench list-models` to see all supported models
