# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright (c) 2026 Delos Data, Inc.
"""
Default power, cost and token-price inputs for inference economics, and the metrics they derive.

Values marked ``InferenceX`` are the per-chip figures https://inferencex.semianalysis.com
publishes (``/api/v1/views/options`` and its chip pages), as of October 2026: TDP, all-in
provisioned watts (GPU plus its share of host CPU, NICs and cooling), and $/chip-hour for owning
at large hyperscaler volume and for renting. Values marked ``estimate`` fill in chips InferenceX
does not cover and should be overridden when a better figure is known.

:func:`fetch_inferencex_gpus` reads InferenceX's current figures and :func:`update_gpu_economics`
applies them to :data:`GPU_ECONOMICS` for the running process, marking each as fetched that day.
"""

import dataclasses
import datetime as dt
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import requests

__all__ = [
    "COST_BASES",
    "DEFAULT_HOST_WATTS_PER_GPU",
    "DEFAULT_PUE",
    "ECONOMICS_METRICS",
    "GPU_ECONOMICS",
    "MODEL_TOKEN_PRICES",
    "SYSTEM_QUERY",
    "X_AXIS_METRICS",
    "EconomicsInputs",
    "GpuEconomics",
    "TokenPrice",
    "economics_metrics",
    "estimate_all_in_watts",
    "fetch_inferencex_gpus",
    "lookup_gpu",
    "resolve_economics",
    "update_gpu_economics",
]

# Calibrated so H100 reproduces InferenceX's 1.37 kW: (700 W + 354 W) x 1.3.
DEFAULT_HOST_WATTS_PER_GPU = 350.0
# power_model's air-cooled facility PUE.
DEFAULT_PUE = 1.3

#: The origin of an input probed from the running system, such as server wall power, rather
#: than configured.
SYSTEM_QUERY = "system query"

#: ``owning`` is InferenceX's default basis ("H" metrics); ``rental`` is its "R" variant.
COST_BASES = ("owning", "rental")


def estimate_all_in_watts(
    tdp_w: float, host_watts_per_gpu: float = DEFAULT_HOST_WATTS_PER_GPU, pue: float = DEFAULT_PUE
) -> float:
    """All-in provisioned watts per GPU: its TDP plus its host share, times facility PUE."""
    return (tdp_w + host_watts_per_gpu) * pue


@dataclass(frozen=True)
class GpuEconomics:
    """Per-chip power and cost defaults; ``None`` where no figure is known."""

    label: str
    vendor: str
    tdp_w: float
    all_in_w: float
    owning_cost_per_hour: float | None
    rental_cost_per_hour: float | None
    power_source: str
    cost_source: str
    aliases: tuple[str, ...] = ()


ESTIMATED_POWER = "estimate: (TDP + host share) x PUE"
MARKET_RENTAL = "estimate: market rental, October 2026"


def _estimated(label: str, vendor: str, tdp_w: float, rental: float, *aliases: str) -> GpuEconomics:
    return GpuEconomics(
        label, vendor, tdp_w, estimate_all_in_watts(tdp_w), None, rental, ESTIMATED_POWER, MARKET_RENTAL, aliases
    )


#: Keyed by a normalised name; see :func:`lookup_gpu`.
GPU_ECONOMICS: dict[str, GpuEconomics] = {
    "h100": GpuEconomics(
        "H100 SXM", "NVIDIA", 700, 1370, 1.17, 2.00, "InferenceX", "InferenceX", ("h100-sxm", "h100-80gb-hbm3")
    ),
    "h200": GpuEconomics("H200 SXM", "NVIDIA", 700, 1370, 1.22, 2.90, "InferenceX", "InferenceX", ("h200-sxm",)),
    "b200": GpuEconomics("B200", "NVIDIA", 1000, 1710, 1.73, 3.70, "InferenceX", "InferenceX"),
    "b300": GpuEconomics("B300", "NVIDIA", 1200, 1900, 2.26, 4.25, "InferenceX", "InferenceX"),
    "gb200": GpuEconomics(
        "GB200 NVL72", "NVIDIA", 1200, 1870, 1.86, 4.00, "InferenceX", "InferenceX", ("gb200-nvl72",)
    ),
    "gb300": GpuEconomics(
        "GB300 NVL72", "NVIDIA", 1400, 2120, 2.31, 5.00, "InferenceX", "InferenceX", ("gb300-nvl72",)
    ),
    "vr200": GpuEconomics(
        "Vera Rubin NVL72",
        "NVIDIA",
        1800,
        round(1800 * 2120 / 1400),
        3.61,
        8.50,
        "estimate: GB300's all-in/TDP ratio",
        "InferenceX",
        ("vera-rubin-nvl72",),
    ),
    "mi300x": GpuEconomics("MI300X", "AMD", 750, 1390, 0.95, 1.30, "InferenceX", "InferenceX", ("instinct-mi300x",)),
    "mi325x": GpuEconomics("MI325X", "AMD", 1000, 1690, 1.10, 1.60, "InferenceX", "InferenceX", ("instinct-mi325x",)),
    "mi355x": GpuEconomics("MI355X", "AMD", 1400, 2090, 1.50, 2.90, "InferenceX", "InferenceX", ("instinct-mi355x",)),
    "rtx-pro-6000-blackwell": GpuEconomics(
        "RTX PRO 6000 Blackwell",
        "NVIDIA",
        600,
        estimate_all_in_watts(600),
        0.68,
        0.52,
        ESTIMATED_POWER,
        "InferenceX",
        ("rtx6000pro", "rtx-pro-6000", "rtx-pro-6000-blackwell-server-edition"),
    ),
    "jalapeno": GpuEconomics(
        "Jalapeño", "OpenAI", 700, estimate_all_in_watts(700), 1.27, 1.27, ESTIMATED_POWER, "InferenceX"
    ),
    "tpuv7": GpuEconomics(
        "TPU7x", "Google", 980, estimate_all_in_watts(980), 1.21, 2.00, ESTIMATED_POWER, "InferenceX", ("tpu7x",)
    ),
    "a100": _estimated("A100 80GB SXM", "NVIDIA", 400, 1.38, "a100-sxm4-80gb", "a100-80gb"),
    "l40s": _estimated("L40S", "NVIDIA", 350, 0.97),
    "l4": _estimated("L4", "NVIDIA", 72, 0.49),
    "rtx-pro-2000-blackwell": GpuEconomics(
        "RTX PRO 2000 Blackwell",
        "NVIDIA",
        70,
        estimate_all_in_watts(70),
        None,
        0.21,
        ESTIMATED_POWER,
        "getdeploying.com: lowest rental (24-month reserved), October 2026",
        ("rtx-pro-2000",),
    ),
    "mi350x": _estimated("MI350X", "AMD", 1000, 3.40, "instinct-mi350x"),
}


@dataclass(frozen=True)
class TokenPrice:
    """List price in dollars per million tokens."""

    input: float
    output: float
    source: str


#: By served model name. InferenceX's fleet view defaults output to 4x the input price.
MODEL_TOKEN_PRICES: dict[str, TokenPrice] = {
    "Qwen/Qwen3-8B": TokenPrice(0.05, 0.20, "estimate: Qwen API list price"),
}


def _normalise(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"^(nvidia|amd)-", "", text)


def lookup_gpu(name: str | None) -> tuple[str, GpuEconomics] | None:
    """``(key, defaults)`` for a GPU named as a profile sku or as nvidia-smi/amd-smi prints it."""
    if not name:
        return None
    wanted = _normalise(name)
    for key, gpu in GPU_ECONOMICS.items():
        if wanted == key or wanted in gpu.aliases:
            return key, gpu
    return None


INFERENCEX_URL = "https://inferencex.semianalysis.com"
INFERENCEX_TIMEOUT = 30.0

_CHIP_LINK = re.compile(r"/chips/([a-z0-9-]+)")
_ALL_IN_POWER = re.compile(r"All-in power per chip\s*([\d,.]+)\s*(k?W)")


def _get(url: str, timeout: float) -> requests.Response:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response


def _all_in_watts(chip_page: str) -> float | None:
    """The "All-in power per chip" a chip page states, in watts, or None if it states none."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", chip_page, flags=re.S)
    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text))
    if match := _ALL_IN_POWER.search(text):
        return float(match.group(1).replace(",", "")) * (1000 if match.group(2) == "kW" else 1)
    return None


def fetch_inferencex_gpus(
    base_url: str = INFERENCEX_URL, timeout: float = INFERENCEX_TIMEOUT
) -> dict[str, GpuEconomics]:
    """
    Every chip InferenceX publishes now, by its key: TDP and $/chip-hour for owning and renting
    from its options API, and all-in power from the chip's page under ``/chips`` -- estimated
    from TDP for a chip without one.

    Raises :class:`requests.RequestException`, or ``ValueError``/``KeyError`` for a response not
    in the shape expected.
    """
    source = f"InferenceX, fetched {dt.date.today().isoformat()}"
    hardware = _get(f"{base_url}/api/v1/views/options", timeout).json()["hardware"]
    # A chip's page is its key, or its key and system, e.g. gb200-nvl72; -vs- pages compare two.
    pages = sorted(slug for slug in _CHIP_LINK.findall(_get(f"{base_url}/chips", timeout).text) if "-vs-" not in slug)

    chips = {}
    for chip in hardware:
        key, tdp, cost = chip["key"], float(chip["tdpW"]), chip.get("costPerHour") or {}
        page = next((slug for slug in pages if slug == key or slug.startswith(f"{key}-")), None)
        all_in = _all_in_watts(_get(f"{base_url}/chips/{page}", timeout).text) if page else None
        chips[key] = GpuEconomics(
            label=chip["label"],
            vendor=chip["vendor"],
            tdp_w=tdp,
            all_in_w=estimate_all_in_watts(tdp) if all_in is None else all_in,
            owning_cost_per_hour=cost.get("h"),
            rental_cost_per_hour=cost.get("r"),
            power_source=ESTIMATED_POWER if all_in is None else source,
            cost_source=source,
            # So a chip added from here is found by its name too, e.g. a profile's sku.
            aliases=tuple({_normalise(chip["label"])} - {key}),
        )
    return chips


def update_gpu_economics(fetched: Mapping[str, GpuEconomics]) -> list[str]:
    """
    Apply *fetched* (from :func:`fetch_inferencex_gpus`) to :data:`GPU_ECONOMICS` in place.

    A chip already there, found by key, alias or label, takes the fetched TDP and costs, and the
    fetched all-in power when InferenceX publishes one; an all-in power that was only estimated
    is re-estimated from the new TDP, and any other is kept. A chip not there is added. Returns
    the keys updated or added.
    """
    updated = []
    for key, chip in fetched.items():
        found = lookup_gpu(key) or lookup_gpu(chip.label)
        if found is None:
            GPU_ECONOMICS[key] = chip
            updated.append(key)
            continue
        ours_key, ours = found
        if chip.power_source != ESTIMATED_POWER:
            power = {"all_in_w": chip.all_in_w, "power_source": chip.power_source}
        elif ours.power_source == ESTIMATED_POWER:
            power = {"all_in_w": estimate_all_in_watts(chip.tdp_w)}
        else:
            power = {}
        GPU_ECONOMICS[ours_key] = dataclasses.replace(
            ours,
            tdp_w=chip.tdp_w,
            owning_cost_per_hour=chip.owning_cost_per_hour,
            rental_cost_per_hour=chip.rental_cost_per_hour,
            cost_source=chip.cost_source,
            **power,
        )
        updated.append(ours_key)
    return updated


@dataclass(frozen=True)
class EconomicsInputs:
    """The inputs one run's metrics are computed from, each with where it came from."""

    gpu: str | None
    cost_basis: str
    all_in_watts_per_gpu: float | None
    cost_per_gpu_hour: float | None
    input_price_per_million: float | None
    output_price_per_million: float | None
    sources: Mapping[str, str]
    #: The values ``all_in_watts_per_gpu`` was computed from -- ``tdp_w``,
    #: ``host_watts_per_gpu``, ``server_watts``, ``gpus_per_server``, ``pue`` -- each with its
    #: origin in :attr:`sources`. Empty when the figure was used as given or published.
    power_inputs: Mapping[str, float] = field(default_factory=dict)


def _number(overrides: Mapping[str, Any], key: str) -> float | None:
    value = overrides.get(key)
    return None if value is None else float(value)


def resolve_economics(
    gpu_name: str | None,
    model: str | None,
    overrides: Mapping[str, Any] | None = None,
    origins: Mapping[str, str] | None = None,
) -> EconomicsInputs:
    """
    The defaults for *gpu_name* and *model*, with *overrides* applied.

    *overrides* takes ``gpu``, ``cost_basis``, ``cost_per_gpu_hour``, ``input_price_per_million``,
    ``output_price_per_million`` and, for power, the first of: ``all_in_watts_per_gpu``;
    ``server_watts`` measured at the wall under load, spread over ``gpus_per_server`` and times
    ``pue``; ``host_watts_per_gpu`` and ``pue`` to re-estimate from TDP. A GPU whose published
    figure is itself that estimate is re-estimated the same way, so its inputs are recorded.

    *origins* names where an override came from, e.g. ``"AgentX profile"`` or
    :data:`SYSTEM_QUERY`, and becomes its source; an override with none is ``"override"``.
    """
    overrides = overrides or {}
    sources: dict[str, str] = {}
    power_inputs: dict[str, float] = {}
    found = lookup_gpu(overrides.get("gpu") or gpu_name)
    key, gpu = found if found else (None, None)

    def origin(name: str) -> str:
        return (origins or {}).get(name, "override")

    def power_input(name: str, default: float | None = None) -> float:
        """Record an input to the power figure: its override, else *default*."""
        value = _number(overrides, name)
        power_inputs[name], sources[name] = (default, "default") if value is None else (value, origin(name))
        return power_inputs[name]

    if overrides.get("gpu"):
        sources["gpu"] = origin("gpu")

    basis = str(overrides.get("cost_basis") or "owning")
    if basis not in COST_BASES:
        raise ValueError(f"cost_basis must be one of {', '.join(COST_BASES)}, not {basis!r}")
    sources["cost_basis"] = origin("cost_basis") if overrides.get("cost_basis") else "default"

    if (watts := _number(overrides, "all_in_watts_per_gpu")) is not None:
        sources["all_in_watts_per_gpu"] = origin("all_in_watts_per_gpu")
    elif _number(overrides, "server_watts") is not None:
        gpus = _number(overrides, "gpus_per_server")
        if not gpus or gpus <= 0:
            raise ValueError("server_watts needs gpus_per_server, the GPUs that power is shared by")
        watts = power_input("server_watts") * power_input("pue", DEFAULT_PUE) / power_input("gpus_per_server")
        sources["all_in_watts_per_gpu"] = "server_watts x pue / gpus_per_server"
    elif gpu is not None and (
        "host_watts_per_gpu" in overrides or "pue" in overrides or gpu.power_source == ESTIMATED_POWER
    ):
        power_inputs["tdp_w"], sources["tdp_w"] = gpu.tdp_w, "GPU defaults"
        watts = estimate_all_in_watts(
            gpu.tdp_w, power_input("host_watts_per_gpu", DEFAULT_HOST_WATTS_PER_GPU), power_input("pue", DEFAULT_PUE)
        )
        sources["all_in_watts_per_gpu"] = "(tdp_w + host_watts_per_gpu) x pue"
    elif gpu is not None:
        watts = gpu.all_in_w
        sources["all_in_watts_per_gpu"] = gpu.power_source
    else:
        watts = None

    if (cost := _number(overrides, "cost_per_gpu_hour")) is not None:
        sources["cost_per_gpu_hour"] = origin("cost_per_gpu_hour")
    elif gpu is not None:
        owning, rental = gpu.owning_cost_per_hour, gpu.rental_cost_per_hour
        cost = owning if basis == "owning" else rental
        if cost is None and (cost := rental if basis == "owning" else owning) is not None:
            sources["cost_basis"] = f"{sources['cost_basis']} {basis}, which {gpu.label} has no rate for"
            basis = "rental" if basis == "owning" else "owning"
        if cost is not None:
            sources["cost_per_gpu_hour"] = f"{gpu.cost_source} ({basis})"

    price = MODEL_TOKEN_PRICES.get(model or "")
    prices = {}
    for side in ("input", "output"):
        field = f"{side}_price_per_million"
        if (value := _number(overrides, field)) is not None:
            sources[field] = origin(field)
        elif price is not None:
            value = getattr(price, side)
            sources[field] = price.source
        prices[side] = value

    return EconomicsInputs(
        gpu=key,
        cost_basis=basis,
        all_in_watts_per_gpu=watts,
        cost_per_gpu_hour=cost,
        input_price_per_million=prices["input"],
        output_price_per_million=prices["output"],
        sources=sources,
        power_inputs=power_inputs,
    )


#: (key, label, unit) for every y-axis metric, in InferenceX's naming.
ECONOMICS_METRICS = (
    ("tpPerGpu", "Token Throughput per Chip", "tok/s/chip"),
    ("inputTputPerGpu", "Input Token Throughput per Chip", "tok/s/chip"),
    ("outputTputPerGpu", "Output Token Throughput per Chip", "tok/s/chip"),
    ("tpPerMw", "Token Throughput per All in Utility MW", "tok/s/MW"),
    ("inputTputPerMw", "Input Token Throughput per All in Utility MW", "tok/s/MW"),
    ("outputTputPerMw", "Output Token Throughput per All in Utility MW", "tok/s/MW"),
    ("tokenRevenuePerGpuHour", "Token Revenue per GPU Hour", "$/GPU/hr"),
    ("tokensPerDollar", "Total Tokens per $1 TCO", "tok/$"),
    ("outputTokensPerDollar", "Output Tokens per $1 TCO", "tok/$"),
    ("inputTokensPerDollar", "Input Tokens per $1 TCO", "tok/$"),
    ("cost", "Cost per Million Total Tokens", "$"),
    ("costOutput", "Cost per Million Output Tokens", "$"),
    ("costInput", "Cost per Million Input Tokens", "$"),
    ("jTotal", "All-in Provisioned Joules per Total Token", "J/tok"),
    ("jOutput", "All-in Provisioned Joules per Output Token", "J/tok"),
    ("jInput", "All-in Provisioned Joules per Input Token", "J/tok"),
)

#: (key, label, unit, path into an AgentX scored aggregate's request_metrics.latency) per x-axis.
X_AXIS_METRICS = (
    ("e2e-normalized-interactivity", "E2E Normalized Interactivity", "tok/s/user", "e2e_norm_intvty"),
    ("interactivity", "Interactivity", "tok/s/user", "intvty"),
    ("e2e", "E2E Latency", "s", "e2el"),
    ("ttft", "TTFT", "s", "ttft"),
)


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0 or not math.isfinite(denominator):
        return None
    return numerator / denominator


def economics_metrics(
    total_tps_per_gpu: float | None,
    input_tps_per_gpu: float | None,
    output_tps_per_gpu: float | None,
    inputs: EconomicsInputs,
) -> dict[str, float | None]:
    """Every :data:`ECONOMICS_METRICS` value, ``None`` where an input is missing."""
    throughput = {"total": total_tps_per_gpu, "input": input_tps_per_gpu, "output": output_tps_per_gpu}
    megawatts = None if inputs.all_in_watts_per_gpu is None else inputs.all_in_watts_per_gpu / 1e6
    cost = inputs.cost_per_gpu_hour
    per_hour = {side: None if tps is None else tps * 3600 for side, tps in throughput.items()}

    revenue = None
    if None not in (
        input_tps_per_gpu,
        output_tps_per_gpu,
        inputs.input_price_per_million,
        inputs.output_price_per_million,
    ):
        revenue = (
            (input_tps_per_gpu * inputs.input_price_per_million + output_tps_per_gpu * inputs.output_price_per_million)
            * 3600
            / 1e6
        )

    def cost_per_million(side: str) -> float | None:
        tokens = _ratio(per_hour[side], 1e6)
        return _ratio(cost, tokens)

    return {
        "tpPerGpu": total_tps_per_gpu,
        "inputTputPerGpu": input_tps_per_gpu,
        "outputTputPerGpu": output_tps_per_gpu,
        "tpPerMw": _ratio(total_tps_per_gpu, megawatts),
        "inputTputPerMw": _ratio(input_tps_per_gpu, megawatts),
        "outputTputPerMw": _ratio(output_tps_per_gpu, megawatts),
        "tokenRevenuePerGpuHour": revenue,
        "tokensPerDollar": _ratio(per_hour["total"], cost),
        "outputTokensPerDollar": _ratio(per_hour["output"], cost),
        "inputTokensPerDollar": _ratio(per_hour["input"], cost),
        "cost": cost_per_million("total"),
        "costOutput": cost_per_million("output"),
        "costInput": cost_per_million("input"),
        "jTotal": _ratio(inputs.all_in_watts_per_gpu, total_tps_per_gpu),
        "jOutput": _ratio(inputs.all_in_watts_per_gpu, output_tps_per_gpu),
        "jInput": _ratio(inputs.all_in_watts_per_gpu, input_tps_per_gpu),
    }
