# Production Test Framework

`production-test-framework` is an installable **Python package** of reusable building
blocks for deploying, validating, and load-testing production platform components —
Kubernetes clusters, the LGTM observability stack, network switches, and vLLM
inference services. It is distributed as a Python wheel and shipped inside a Docker
image that also bundles the CLI tooling (`kubectl`, `helm`, `k3d`) needed to drive an
end-to-end test run.

- **Package name:** `production-test-framework` (import as `production_test_framework`)
- **Python:** `>= 3.14`
- **Build backend:** hatchling + hatch-vcs (version derived from git tags)
- **License:** FSL-1.1-ALv2 — © Delos Data, Inc.

## Table of contents

- [Design principle: generic, open interfaces only](#design-principle-generic-open-interfaces-only)
- [What's included](#whats-included)
- [Installation & usage](#installation--usage)
- [Build & deploy](#build--deploy)
- [Tagging a release on GitHub](#tagging-a-release-on-github)
- [Running the test harness (Docker image)](#running-the-test-harness-docker-image)
- [Development](#development)
- [License](#license)

## Design principle: generic, open interfaces only

**This package integrates only through generic, open, standardized interfaces. It does
not depend on proprietary or system-specific vendor libraries or SDKs.**

Every integration is built on a protocol or API that is openly documented and
vendor-neutral:

| Concern | Open interface used | Library |
| --- | --- | --- |
| Remote command execution | SSH | `paramiko` |
| Kubernetes | `kubectl` CLI over SSH / locally | (kubectl) |
| Network switches | **NVUE REST API** (open, documented) | `requests` |
| Inference | OpenAI-compatible HTTP (vLLM) | `requests` |
| GPU collectives benchmarking | `nccl-tests` CLI + MPI launcher (`mpirun`) | (nccl-tests / Open MPI) |
| Telemetry | OTLP over gRPC (OpenTelemetry) | `opentelemetry-*` |
| Metrics query | Prometheus/Mimir HTTP API | `requests` |
| Configuration mgmt | Ansible playbooks | `ansible-core` |

The NVIDIA Cumulus switch driver, for example, talks to the switch exclusively over
the **open NVUE REST API** — not a proprietary SDK. This keeps drivers swappable and
the package portable.

> **Contribution rule:** new drivers and integrations must be implemented against an
> open, standard interface. Do not add dependencies on proprietary or system-specific
> vendor libraries.

## What's included

### Third-party libraries

Runtime dependencies (see [`pyproject.toml`](pyproject.toml)):

| Library | Purpose |
| --- | --- |
| `paramiko` | SSH transport and command execution |
| `requests` | HTTP/REST clients (NVUE, vLLM, Mimir) |
| `ansible-core` | Runs the playbooks under [`ansible/`](ansible/) |
| `locust` | Load generation |
| `opentelemetry-api` / `-sdk` / `-exporter-otlp-proto-grpc` | OTLP metrics emission |
| `qase-pytest` | Qase TestOps test reporting |
| `docopt` | CLI argument parsing for `switch-status` |
| `pyyaml` | Environment variable declaration files (`env_vars.load_declarations`) |
| `pip` | Runtime package management inside the image |

### Modules the package provides

Importable API under `production_test_framework`:

| Module | Key public API |
| --- | --- |
| `config` | `LGTMConfig` — LGTM/host config dataclass (`from_env()`, `environment_variables()`) |
| `env_vars` | `EnvironmentVariable`, `used_by`, `load_declarations`, `is_set`, `missing_required`, `render_report`, `env_tag_variables` — declare the environment variables a component reads, in Python or a YAML file, and report which are set without printing values |
| `ssh` | `SSHExecutor`, `CommandResult`, `check_ssh_password_login`, `wait_for_ssh_password_login` |
| `helper` | `run_command`, `run_cancellable_command`, `ping` / `wait_for_ping`, `check_tcp_connectivity`, `query_mimir` |
| `k8s` | `KubernetesClient`, `KubectlPortForwarder`, `LocalKubectlPortForwarder`, `Node`, `Pod` |
| `vllm` | `VllmClient`, `VllmConfig`, `InferenceResult` |
| `switch` | `NetworkSwitch` (ABC), `create_switch`, `SwitchType`, `NvidiaCumulusSwitch` (NVUE driver), `SwitchAPIError` |
| `workload` | `Workload` (ABC), `CommandWorkload` (ABC), `PromptWorkload`, `InferencexWorkload`, `NcclWorkload`, `NcclTest`, `NcclTestResult`, `WorkloadResult` |
| `telemetry` | `Otelp`, `OtelpConfig`, `create_otelp` |
| `loadgen` | `LocustMetricsUser`, `LocustTestConfig`, `run_locust_test` |
| `utils` | `wait_for` (generic polling) |

### Console script

Installing the package provides a `switch-status` command-line tool
(entry point `production_test_framework.switch.switch_status:main`) for querying
network switch status over the NVUE REST API.

## Installation & usage

The package targets Python 3.14+. Using [`uv`](https://docs.astral.sh/uv/):

```bash
# From a checkout of this repo
uv sync
```

Add it as a dependency of your own uv project directly from GitHub. The version is
resolved from git tags, so pin a tag/branch/commit with `@<ref>` for a reproducible
install:

```bash
# A specific release tag (recommended)
uv add "git+https://github.com/thurttdd/production-test-framework.git@v0.3.0"

# Latest from the default branch
uv add "git+https://github.com/thurttdd/production-test-framework.git"
```

For private-repo access over HTTPS, provide a token
(`git+https://<token>@github.com/...`), or use SSH form
(`git+ssh://git@github.com/thurttdd/production-test-framework.git@v0.3.0`).

Use the library in your own tests or scripts:

```python
from production_test_framework.config import LGTMConfig
from production_test_framework.k8s import KubernetesClient
from production_test_framework.switch import create_switch, SwitchType

config = LGTMConfig.from_env()
k8s = KubernetesClient(config)
print(k8s.all_nodes_ready())
```

Run the bundled CLI:

```bash
switch-status --help
switch-status --hostname switch01 --username admin --switch-type nvidia-cumulus
```

## Build & deploy

### Build the wheel

The package uses **hatchling** with **hatch-vcs**, so the version is derived from git
tags — there is no hard-coded version in `pyproject.toml`.

```bash
uv build          # produces sdist + wheel in dist/
```

For a local build on a commit **without** a tag (or with a trimmed git history), set a
pretend version so hatch-vcs can resolve one:

```bash
export SETUPTOOLS_SCM_PRETEND_VERSION=0.3.0.dev
uv build
```

Versioning rules:

- **Tagged commit** `v0.3.0` → version `0.3.0`.
- **Untagged commit** after a tag → dev suffix, e.g. `0.3.0.dev8+gc88d493`.

### Build the Docker image

The image ([`Dockerfile`](Dockerfile)) bundles the package plus `kubectl`, `helm`,
`k3d`, `uv`, and Python 3.14, and runs as a non-root user.

```bash
docker build -t production-test-framework .

# Optionally stamp the git hash / version (CI does this automatically):
docker build \
  --build-arg GIT_HASH=$(git rev-parse --short HEAD) \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=0.3.0 \
  -t production-test-framework .
```

### Deploy (Docker Hub via CI)

Images are published automatically by
[`.github/workflows/docker.yml`](.github/workflows/docker.yml):

| Trigger | Image tags pushed |
| --- | --- |
| Push to `main` | `latest`, `sha-<short-sha>` |
| Push tag `v*` | `<tag>` (e.g. `v0.3.0`), `latest` |
| Pull request | built only, not pushed (`pr-<n>`) |

On a `v*` tag, CI also passes `SETUPTOOLS_SCM_PRETEND_VERSION=<version without v>` into
the build so the packaged version matches the tag.

Required repository secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` (and optionally
`DOCKERHUB_REPO` to override the target repository).

> **Note:** the package is **not currently published to PyPI**. Distribution is via the
> built wheel and the Docker image.

## Tagging a release on GitHub

Because the version is git-tag-driven, cutting a release is just a tag push. The tag
simultaneously sets the package version (hatch-vcs) **and** the Docker image tag (CI).

```bash
# Create an annotated tag with a PEP 440 version, v-prefixed
git tag -a v0.3.0 -m "Release 0.3.0"
git push origin v0.3.0
```

This triggers `docker.yml`, which builds and pushes the `v0.3.0` and `latest` images to
Docker Hub with the version baked in.

Optionally publish a GitHub Release from the tag so release notes (and, if you attach
them, the built wheel/sdist) are visible on the Releases page:

```bash
uv build                                   # build artifacts to attach
gh release create v0.3.0 \
  --title "v0.3.0" \
  --generate-notes \
  dist/production_test_framework-0.3.0*
```

Guidelines:

- Use **`v`-prefixed, PEP 440** versions (`v0.3.0`, `v1.0.0rc1`).
- Prefer **annotated** tags (`-a`) so the release carries a message.
- Commits after a tag are automatically versioned as dev builds
  (`0.3.0.devN+g<sha>`) — no action needed for interim builds.

## Running the test harness (Docker image)

The Docker image is a ready-to-run consumer of the package: it bundles `kubectl`,
`helm`, and `k3d`, and drives test runs through the [`Makefile`](Makefile). This is the
recommended way to execute a full deploy → test → teardown cycle.

### Environment variables

Provide these to the container (via a mounted `.env` or `-e` flags). See
[`env.example`](env.example).

| Variable | Purpose |
| --- | --- |
| `ANSIBLE_REMOTE_USER` | SSH username for the target host |
| `REMOTE_HOST` | Target cluster hostname/IP |
| `CLUSTER` | Cluster name |
| `ANSIBLE_INVENTORY_FILE` | Path to the Ansible inventory |
| `TESTS_DIR` | Test directory override (optional) |
| `QASE_TESTOPS_API_TOKEN` | Qase reporting token (optional) |

### Run interactively

```bash
docker run -it --rm \
  -v $(pwd)/.env:/app/framework/.env:ro \
  -v /path/to/your/tests:/app/tests:ro \
  -v /path/to/helm/charts/mosaic:/app/framework/charts/mosaic:ro \
  -e SSH_AUTH_SOCK=/tmp/ssh-agent/socket \
  -v $SSH_AUTH_SOCK:/tmp/ssh-agent/socket \
  production-test-framework
```

The entrypoint ([`scripts/docker-entrypoint.sh`](scripts/docker-entrypoint.sh)) copies
mounted tests/charts into place and drops you into a shell in `/app/framework`. From
there, run a Makefile target:

| Target | What it does |
| --- | --- |
| `make test` | `prereqs` → deploy charts → run tests → teardown |
| `make test-run-only` | Run the main suite only (no deploy/undeploy) |
| `make test-production` | Create k3d cluster → deploy → test → destroy cluster |
| `make help` | List all targets |

### Run a single target (CI / non-interactive)

Set `RUN_MAKE_TARGET` to run one target and exit — no shell, no banner:

```bash
docker run -it --rm \
  -e RUN_MAKE_TARGET=test-run-only \
  -v $(pwd)/.env:/app/framework/.env:ro \
  -v /path/to/your/tests:/app/tests:ro \
  production-test-framework
```

## Development

```bash
uv sync --all-packages --group dev   # install package + dev tools
uv run pytest                        # run unit tests
uv run ruff check src unit_tests     # lint (line length 120, py314)
uv run ruff format src unit_tests    # format
pre-commit install                   # enable git hooks
```

Unit tests live in the [`unit_tests/`](unit_tests/) uv-workspace member and are **not**
shipped in the wheel (which packages only `src/production_test_framework`). CI runs
`uv run pytest` on every push and pull request
([`.github/workflows/test.yml`](.github/workflows/test.yml)).

## License

Functional Source License, Version 1.1, Apache 2.0 Future License
(**FSL-1.1-ALv2**). © 2025 Delos Data, Inc. See [`LICENSE`](LICENSE).
