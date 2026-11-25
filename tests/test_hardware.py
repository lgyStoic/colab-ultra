import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console

from colab_ultra import cli
from colab_ultra.remote import hardware


PRECISIONS = ("fp16", "bf16", "int8", "fp8", "nvfp4")
EXPECTED_METRICS = {
    "T4X2": (130.0, None, 260.0, None, None),
    "T4": (65.0, None, 130.0, None, None),
    "L4": (121.0, 121.0, 242.5, 242.5, None),
    "A100": (312.0, 312.0, 624.0, None, None),
    "G4": (504.0, 504.0, 1008.0, 1008.0, 2015.0),
    "H100": (989.0, 989.0, 1979.0, 1979.0, None),
    "CPU-STD": (None, None, None, None, None),
    "CPU-HIGH": (None, None, None, None, None),
    "TPU-V5E1": (None, None, None, None, None),
    "TPU-V6E1": (None, None, None, None, None),
}


def test_catalog_keys():
    assert hardware.SPECS.keys() == EXPECTED_METRICS.keys()


@pytest.mark.parametrize("key, metrics", EXPECTED_METRICS.items())
def test_precision_metrics_and_flags(key, metrics):
    spec = hardware.get_spec(key)
    assert spec.key == key
    flags = {}
    for precision, value in zip(PRECISIONS, metrics):
        assert spec.metric_for(precision) == value
        assert spec.metric_for(precision.upper()) == value
        flag = f"supports_{precision}"
        assert getattr(spec, flag) is (value is not None)
        flags[flag] = value is not None
    assert hardware.as_dict(spec) == {**asdict(spec), **flags}


def test_spec_is_frozen():
    with pytest.raises(FrozenInstanceError):
        hardware.get_spec("T4").key = "L4"


@pytest.mark.parametrize(
    "name, key",
    [
        ("t4x2", "T4X2"),
        ("2xT4", "T4X2"),
        ("dualt4", "T4X2"),
        ("KaggleT4", "T4X2"),
        ("cpu", "CPU-STD"),
        ("CPU_STD", "CPU-STD"),
        ("CPU-highram", "CPU-HIGH"),
        ("HIGH_RAM", "CPU-HIGH"),
        ("v5e1", "TPU-V5E1"),
        ("v6e1", "TPU-V6E1"),
        ("rtx pro 6000", "G4"),
        (" a100 ", "A100"),
    ],
)
def test_get_spec_aliases(name, key):
    assert hardware.get_spec(name) is hardware.SPECS[key]


def test_get_spec_unknown():
    with pytest.raises(KeyError):
        hardware.get_spec("unknown")


@pytest.mark.parametrize(
    "filters, keys",
    [
        ({"precision": "int8"}, {"T4X2", "T4", "L4", "A100", "G4", "H100"}),
        ({"precision": "nvfp4"}, {"G4"}),
        ({"precision": "BF16"}, {"L4", "A100", "G4", "H100"}),
        ({"min_vram_gb": 80}, {"A100", "G4", "H100"}),
        ({"kind": "gpu"}, {"T4X2", "T4", "L4", "A100", "G4", "H100"}),
        ({"kind": "cpu"}, {"CPU-STD", "CPU-HIGH"}),
        ({"kind": "tpu"}, {"TPU-V5E1", "TPU-V6E1"}),
        ({"precision": "fp8", "min_vram_gb": 80, "kind": "gpu"}, {"G4", "H100"}),
        ({"precision": "int8", "kind": "cpu"}, set()),
    ],
)
def test_list_specs_filters(filters, keys):
    assert {spec.key for spec in hardware.list_specs(**filters)} == keys


def test_invalid_precision():
    spec = hardware.get_spec("L4")
    with pytest.raises(ValueError):
        spec.metric_for("fp32")
    with pytest.raises(ValueError):
        spec.tflops_per_cu("fp32")
    with pytest.raises(ValueError):
        hardware.list_specs(precision="fp32", min_vram_gb=1000)


def test_tflops_per_cu():
    assert hardware.get_spec("L4").tflops_per_cu("fp8") == 141.81
    assert hardware.get_spec("T4").tflops_per_cu("int8") == 109.24
    assert hardware.get_spec("T4").tflops_per_cu("bf16") is None
    assert hardware.get_spec("T4X2").tflops_per_cu("fp16") is None


@pytest.mark.parametrize(
    "count, name, key",
    [
        (2, "Tesla T4", "T4X2"),
        (1, "Tesla T4", "T4"),
        (1, "NVIDIA RTX PRO 6000 Blackwell Server Edition", "G4"),
        (1, "NVIDIA H100 80GB HBM3", "H100"),
        (1, "NVIDIA A100-SXM4-80GB", "A100"),
        (1, "NVIDIA L4", "L4"),
        (1, "NVIDIA L40S", None),
        (0, "", None),
    ],
)
def test_current_cuda(monkeypatch, count, name, key):
    cuda = Mock()
    cuda.is_available.return_value = True
    cuda.device_count.return_value = count
    cuda.get_device_name.return_value = name
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    nvidia_smi = Mock()
    monkeypatch.setattr(hardware.subprocess, "check_output", nvidia_smi)

    assert hardware.current() is hardware.SPECS.get(key)
    cuda.is_available.assert_called_once_with()
    cuda.device_count.assert_called_once_with()
    if count:
        cuda.get_device_name.assert_called_once_with(0)
    else:
        cuda.get_device_name.assert_not_called()
    nvidia_smi.assert_not_called()


@pytest.mark.parametrize(
    "torch, output, key",
    [
        (None, "Tesla T4\n\nTesla T4\n", "T4X2"),
        (
            SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
            "NVIDIA RTX PRO 6000 Blackwell Server Edition\n",
            "G4",
        ),
        (None, "\n", None),
    ],
)
def test_current_nvidia_smi(monkeypatch, torch, output, key):
    monkeypatch.setitem(sys.modules, "torch", torch)
    nvidia_smi = Mock(return_value=output)
    monkeypatch.setattr(hardware.subprocess, "check_output", nvidia_smi)

    assert hardware.current() is hardware.SPECS.get(key)
    nvidia_smi.assert_called_once_with(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True,
    )


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.CalledProcessError(1, "nvidia-smi")])
def test_current_no_gpu(monkeypatch, error):
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(hardware.subprocess, "check_output", Mock(side_effect=error))

    assert hardware.current() is None


@pytest.mark.parametrize("width", [80, 120])
def test_cli_specs_table(monkeypatch, capsys, width):
    monkeypatch.setattr(cli, "console", Console(width=width))
    cli.main(["specs"])
    output = capsys.readouterr().out
    assert all(key in output for key in EXPECTED_METRICS)
    assert "INT8" in output and "(TOPS)" in output
    assert "Dense Tensor Core peak" in output
    assert "Sparse 2:4" in output

    cli.main(["specs", "--precision", "int8"])
    output = capsys.readouterr().out
    assert "INT8/CU" in output
    assert "FREE" in output
    assert "109.24" in output


def test_cli_specs_precision_json(capsys):
    cli.main(["specs", "--precision", "int8", "--json"])
    rows = {row["key"]: row for row in json.loads(capsys.readouterr().out)}
    assert rows.keys() == {"T4X2", "T4", "L4", "A100", "G4", "H100"}
    for key, row in rows.items():
        spec = hardware.get_spec(key)
        assert row == {
            **hardware.as_dict(spec),
            "selected_precision": "int8",
            "efficiency_per_cu": spec.tflops_per_cu("int8"),
        }
    assert rows["T4X2"]["efficiency_per_cu"] is None
    assert rows["T4"]["efficiency_per_cu"] == 109.24


def test_cli_specs_vram_and_kind_json(capsys):
    cli.main(["specs", "--min-vram", "80", "--kind", "gpu", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert rows == [hardware.as_dict(hardware.get_spec(key)) for key in ("A100", "G4", "H100")]
