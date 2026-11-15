"""Assemble the Colab notebook from the percent-format analysis scripts and execute it locally to prove it runs.

Usage:  python src/build_notebook.py [--no-exec]
Output: deliverables/VoltRelay_Analysis.ipynb (executed copy with outputs)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "1_Colab_Notebook" / "VoltRelay_Analysis.ipynb"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1qsjGWrgmEvOaf2Is2fh7RgVEkKd1V3df"

MD_LINE = re.compile(r"^# ?")
FILE_ROOT_PATTERNS = [
    (re.compile(r"Path\(__file__\)\.resolve\(\)\.parents\[\d\]"), "PROJECT_ROOT"),
    (re.compile(r"Path\(__file__\)\.resolve\(\)\.parent\.parent"), "PROJECT_ROOT"),
    (re.compile(r"Path\(__file__\)\.parents\[\d\]"), "PROJECT_ROOT"),
    (re.compile(r"Path\(__file__\)\.resolve\(\)\.parent"), "(PROJECT_ROOT / 'analysis')"),
    (re.compile(r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)"), "str(PROJECT_ROOT)"),
    (re.compile(r"os\.path\.dirname\(os\.path\.abspath\(__file__\)\)"), "str(PROJECT_ROOT / 'analysis')"),
]


def percent_to_cells(path: Path, heading: str | None = None) -> list:
    """Split a jupytext percent-format script into notebook cells."""
    text = path.read_text(encoding="utf-8")
    for pat, rep in FILE_ROOT_PATTERNS:
        text = pat.sub(rep, text)
    text = text.replace('"__file__" in globals()', '"PROJECT_ROOT" in globals()').replace("'__file__' in globals()", "'PROJECT_ROOT' in globals()")
    if "__file__" in text:
        text = text.replace("__file__", "str(PROJECT_ROOT / 'analysis' / '_nb.py')")
    cells, cur_type, cur = [], "code", []

    def flush():
        body = "\n".join(cur).strip("\n")
        if not body.strip():
            return
        if cur_type == "markdown":
            lines = [MD_LINE.sub("", ln, count=1) if ln.startswith("#") else ln for ln in body.splitlines()]
            cells.append(nbf.v4.new_markdown_cell("\n".join(lines)))
        else:
            # drop a leading module docstring-only cell and `if __name__ == "__main__":` guards stay valid in notebooks
            cells.append(nbf.v4.new_code_cell(body))

    for line in text.splitlines():
        m = re.match(r"^# %%(\s*\[markdown\])?.*$", line)
        if m:
            flush()
            cur_type = "markdown" if m.group(1) else "code"
            cur = []
            continue
        cur.append(line)
    flush()
    if heading:
        cells.insert(0, nbf.v4.new_markdown_cell(heading))
    return cells


def setup_cells() -> list:
    common_src = (ROOT / "src" / "common.py").read_text(encoding="utf-8")
    # src/common.py already displays a light copy of each saved figure when running inside IPython
    cells = []
    cells.append(nbf.v4.new_code_cell(
        "# Environment setup - works on Google Colab and locally\n"
        "import sys, os, subprocess, importlib\n"
        "from pathlib import Path\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "if IN_COLAB:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'polars>=1.30', 'lifelines', 'gdown', 'shap', 'lightgbm'], check=True)\n"
        "    PROJECT_ROOT = Path('/content/voltrelay')\n"
        "else:\n"
        "    _here = Path(os.environ.get('VOLTRELAY_ROOT', Path.cwd()))\n"
        "    # run from anywhere inside the project (e.g. deliverables/1_Colab_Notebook): walk up to the project root\n"
        "    PROJECT_ROOT = next((c for c in [_here, *_here.parents] if (c / 'src' / 'common.py').exists() or (c / 'Hackathon Data Set _ Gradient').exists()), _here)\n"
        "for sub in ['src', 'analysis', 'data/parquet', 'data/clean', 'outputs/figures', 'outputs/tables', 'outputs/findings', 'Hackathon Data Set _ Gradient']:\n"
        "    (PROJECT_ROOT / sub).mkdir(parents=True, exist_ok=True)\n"
        "os.chdir(PROJECT_ROOT)\n"
        "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n"
        "print('Project root:', PROJECT_ROOT, '| Colab:', IN_COLAB)"))
    cells.append(nbf.v4.new_markdown_cell(
        "### Shared module: paths, business constants, loaders and chart style\n"
        "Every analysis below imports these definitions so that cleaning rules, the cost model and colours are identical across questions."))
    cells.append(nbf.v4.new_code_cell("%%writefile src/common.py\n" + common_src))
    cells.append(nbf.v4.new_code_cell(
        "import importlib, common\nimportlib.reload(common)\nfrom common import *\nset_style()\nprint('common loaded; clean dir =', CLEAN)"))
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Data acquisition\n"
        "The organisers' Google Drive folder is public. On Colab the eight CSVs (~0.85 GB) are downloaded with `gdown`; "
        "locally the existing copy is used. The raw CSVs are then converted once to typed, compressed Parquet "
        "(the 728 MB swap table becomes ~93 MB) so every later step reads only the columns it needs."))
    cells.append(nbf.v4.new_code_cell(
        "RAW_FILES = ['batteries.csv', 'city_daily_context.csv', 'fleet_partners.csv', 'riders.csv', 'station_hourly_status.csv',\n"
        "             'stations.csv', 'support_tickets.csv', 'swap_events.csv']\n"
        f"DRIVE_FOLDER = '{DRIVE_FOLDER}'\n"
        "missing = [f for f in RAW_FILES if not (RAW / f).exists() and not (RAW / (f + '.gz')).exists()]\n"
        "if missing:\n"
        "    print('Downloading', missing)\n"
        "    try:\n"
        "        import gdown\n"
        "        gdown.download_folder(DRIVE_FOLDER, output=str(RAW), quiet=False, remaining_ok=True)\n"
        "    except Exception as e:\n"
        "        print('Automatic download failed (Google Drive sometimes rate-limits large files):', e)\n"
        "        print('Fallback: open the Drive folder link above -> Organize -> Add shortcut to My Drive, then run:')\n"
        "        print(\"    from google.colab import drive; drive.mount('/content/drive')\")\n"
        "        print('and copy the 8 CSV files into', RAW, '(or upload them there with the Files panel). Then re-run this cell.')\n"
        "        raise\n"
        "# accept either .csv or .csv.gz deliveries\n"
        "import gzip, shutil\n"
        "for f in RAW_FILES:\n"
        "    gz = RAW / (f + '.gz')\n"
        "    if not (RAW / f).exists() and gz.exists():\n"
        "        with gzip.open(gz, 'rb') as src, open(RAW / f, 'wb') as dst:\n"
        "            shutil.copyfileobj(src, dst)\n"
        "for f in RAW_FILES:\n"
        "    p = RAW / f\n"
        "    print(f'{f:28s} {p.stat().st_size/1e6:8.1f} MB')"))
    return cells


def build(sections: list, execute: bool = True, timeout: int = 3600):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    cells = []
    for sec in sections:
        kind = sec[0]
        if kind == "md":
            cells.append(nbf.v4.new_markdown_cell(sec[1]))
        elif kind == "code":
            cells.append(nbf.v4.new_code_cell(sec[1]))
        elif kind == "setup":
            cells.extend(setup_cells())
        elif kind == "script":
            cells.extend(percent_to_cells(ROOT / sec[1], sec[2] if len(sec) > 2 else None))
        elif kind == "cleanup":
            cells.append(nbf.v4.new_code_cell(
                "# free memory between independent sections (each analysis re-reads what it needs from data/clean)\n"
                "import gc, matplotlib.pyplot as plt\n"
                "_big = [k for k, v in list(globals().items()) if not k.startswith('_') and type(v).__module__.split('.')[0] in "
                "('polars', 'pandas', 'numpy', 'lightgbm', 'shap', 'sklearn', 'statsmodels', 'lifelines', 'scipy')]\n"
                "for _k in _big:\n    del globals()[_k]\n"
                "gc.collect(); plt.close('all')\n"
                "from common import *  # restore shared helpers in case a section shadowed a name\n"
                "print(f'freed {len(_big)} objects')"))
    nb.cells = cells
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} with {len(cells)} cells")
    if execute:
        import json, os
        from nbconvert.preprocessors import ExecutePreprocessor
        # Private kernelspec pointing at this interpreter (the machine's default 'python3' spec may be stale);
        # the user's global Jupyter config is not modified.
        kroot = Path(os.environ.get("VOLTRELAY_KERNEL_DIR", ROOT / ".nbkernel"))
        kdir = kroot / "kernels" / "voltrelay-py"
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "VoltRelay (local)", "language": "python"}), encoding="utf-8")
        os.environ["JUPYTER_PATH"] = str(kroot) + os.pathsep + os.environ.get("JUPYTER_PATH", "")
        os.environ["PYTHONIOENCODING"] = "utf-8"
        t = time.time()
        ep = ExecutePreprocessor(timeout=timeout, kernel_name="voltrelay-py")
        try:
            ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
        finally:
            nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
            nbf.write(nb, OUT)
        print(f"executed OK in {time.time()-t:.0f}s")
    return OUT


if __name__ == "__main__":
    print("import and call build(sections) from a driver script")
