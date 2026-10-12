# PyInstaller spec for the PPE station.
#
#   pyinstaller --noconfirm --clean packaging/ppe_station.spec     (from the repo root)
#
# Builds dist/PPEStation/, a folder ("onedir"), not a single .exe:
#   - a one-file build unpacks ~1 GB of torch into %TEMP% on every start,
#     which is slow, and App Control for Business commonly blocks code
#     running from user-writable temp folders;
#   - a folder installed under Program Files starts at normal speed and can
#     be allow-listed once by the IT department.
#
# Two executables share that folder:
#   PPEStation.exe          the station itself, no console window
#   PPEStation-console.exe  the same program with a console, for --selftest,
#                           --set-relay-ip and diagnosis on site

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "config.yaml"), "."),           # the default; each machine's copy lives in ProgramData
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "models"), "models"),
]
# Ultralytics reads its own YAML (model and default configs) at import time.
datas += collect_data_files("ultralytics")
# OpenVINO finds its CPU plugin and IR reader at run time through plugins.xml,
# so no import analysis can see them: take the package's libraries and data
# files whole. Without this the model fails to load only in the frozen build.
datas += collect_data_files("openvino")
binaries = collect_dynamic_libs("openvino")
# torchvision loads its compiled operators (_C_stable, image_stable) by file
# path through torch.ops.load_library, not by import, and the stock hook still
# looks for the old `torchvision._C` module name. Without them ultralytics
# fails at its first prediction ("operator torchvision::nms does not exist").
binaries += collect_dynamic_libs("torchvision", search_patterns=["*.so", "*.pyd", "*.dll"])
# Ultralytics and OpenVINO check installed versions through package metadata;
# without the dist-info folders a frozen build reports them as missing.
for dist in ("ultralytics", "torch", "torchvision", "openvino", "numpy", "opencv-python",
             "pyyaml", "pillow", "pymodbus", "PySide6", "pygame"):
    try:
        datas += copy_metadata(dist)
    except Exception:  # noqa: BLE001 — a dist that is not installed is simply not bundled
        pass

# Unpickling a .pt model imports ultralytics.nn.* by name, which static
# analysis cannot see.
hiddenimports = (
    collect_submodules("ultralytics.nn")
    + collect_submodules("openvino", filter=lambda name: not name.startswith("openvino.tools"))
    + ["pymodbus.client"]
)

excludes = [
    # Development and notebook tooling that ultralytics or torch can pull in.
    "IPython", "jupyter", "notebook", "pytest", "tensorboard", "torch.utils.tensorboard",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

common = dict(
    exclude_binaries=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-packed DLLs trip antivirus and App Control alike
    icon=None,
)
station = EXE(pyz, a.scripts, [], name="PPEStation", console=False, **common)
console = EXE(pyz, a.scripts, [], name="PPEStation-console", console=True, **common)

coll = COLLECT(
    station, console,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PPEStation",
)
