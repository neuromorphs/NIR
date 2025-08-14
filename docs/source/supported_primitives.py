import re
import requests
from nir.ir import __all_ir as primitives


# Delete Input, Output, and NIRGraph from the list of primitives
primitives = [p for p in primitives if p not in ["Input", "Output", "NIRGraph"]]

# Fetch raw converter file from GitHub
GITHUB_RAW_URLS = [
    # (LibraryName, Direction, URL, function)
    ("jaxsnn", "from_nir", "https://raw.githubusercontent.com/electronicvisions/jaxsnn/refs/heads/main/src/pyjaxsnn/jaxsnn/event/from_nir.py", None),
    ("Lava", "from_nir", "https://raw.githubusercontent.com/neuromorphs/NIR/refs/heads/main/paper/nir_to_lava.py", "_nir_to_lava"),
    ("Nengo", "from_nir", "https://raw.githubusercontent.com/neuromorphs/NIR/refs/heads/main/docs/source/examples/nengo/nir-lorentz.py", "nir_to_nengo"),
    ("Nengo", "to_nir", "https://raw.githubusercontent.com/neuromorphs/NIR/refs/heads/main/docs/source/examples/nengo/nir-lorentz.py", "nengo_to_nir"),
    ("Norse", "from_nir", "https://raw.githubusercontent.com/norse/norse/main/norse/torch/utils/import_nir.py", None),
    ("Norse", "to_nir", "https://raw.githubusercontent.com/norse/norse/main/norse/torch/utils/export_nir.py", None),
    ("rockpool", "from_nir", "https://raw.githubusercontent.com/synsense/rockpool/refs/heads/develop/rockpool/nn/modules/torch/nir.py", "_convert_nir_to_rockpool"),
    ("rockpool", "to_nir", "https://raw.githubusercontent.com/synsense/rockpool/refs/heads/develop/rockpool/nn/modules/torch/nir.py", "_extract_rockpool_module"),
    ("sinabs", "from_nir", "https://raw.githubusercontent.com/synsense/sinabs/refs/heads/develop/sinabs/nir.py", "_import_sinabs_module"),
    ("sinabs", "to_nir", "https://raw.githubusercontent.com/synsense/sinabs/refs/heads/develop/sinabs/nir.py", "_extract_sinabs_module"),
    ("snntorch", "from_nir", "https://raw.githubusercontent.com/jeshraghian/snntorch/refs/heads/master/snntorch/import_nir.py", None),
    ("snntorch", "to_nir", "https://raw.githubusercontent.com/jeshraghian/snntorch/refs/heads/master/snntorch/export_nir.py", None),
    ("SpiNNaker2", "from_nir", "https://gitlab.com/spinnaker2/py-spinnaker2/-/raw/main/src/spinnaker2/s2_nir.py?ref_type=heads", None),
    ("Spyx", "from_nir", "https://raw.githubusercontent.com/kmheckel/spyx/refs/heads/main/spyx/nir.py", "_nir_node_to_spyx_node"),
    ("Spyx", "to_nir", "https://raw.githubusercontent.com/kmheckel/spyx/refs/heads/main/spyx/nir.py", "to_nir"),
]

converter_contents = {}
for lib_name, direction, url, function in GITHUB_RAW_URLS:
    if function is None:
        key = f"{lib_name}_{direction}"
        try:
            converter_contents[key] = requests.get(url).text
        except Exception as e:
            converter_contents[key] = ""
            print(f"Failed to fetch {url}: {e}")
    else:
        key = f"{lib_name}_{direction}"
        try:
            response = requests.get(url)
            pattern = rf"def {function}\s*\(.*?\):(.*?)(?=\ndef |\Z)"
            match = re.search(pattern, response.text, re.DOTALL)
            converter_contents[key] = match.group(0)
        except Exception as e:
            converter_contents[key] = ""
            print(f"Failed to fetch {url}: {e}")

# Check which primitives are supported in each library/direction
supported = {}
for name in primitives:
    supported[name] = {}
    for lib_name, _, _, _ in GITHUB_RAW_URLS:
        from_key = f"{lib_name}_from_nir"
        to_key = f"{lib_name}_to_nir"
        from_supported = "ir.{})".format(name) in converter_contents.get(from_key, "") or "ir.{}(".format(name) in converter_contents.get(from_key, "") or "ir.{}:".format(name) in converter_contents.get(from_key, "")
        to_supported = "ir.{})".format(name) in converter_contents.get(to_key, "") or "ir.{}(".format(name) in converter_contents.get(to_key, "") or "ir.{}:".format(name) in converter_contents.get(to_key, "")
        if from_supported and to_supported:
            supported[name][lib_name] = "⟷"
        elif from_supported:
            supported[name][lib_name] = "→"
        elif to_supported:
            supported[name][lib_name] = "←"
        else:
            supported[name][lib_name] = ""

# Generate supported_nodes.md content
libs = []
for lib, _, _, _ in GITHUB_RAW_URLS:
    if lib not in libs:
        libs.append(lib)
header = "| Primitive | " + " | ".join(libs) + " |"
separator = "|-----------|" + "|".join([":-----:" for _ in libs]) + "|"
rows = []
for name in primitives:
    row = f"| {name} | " + " | ".join([supported[name][lib] for lib in libs]) + " |"
    rows.append(row)
dynamic_md = "\n".join([header, separator] + rows)

static_md = """
# Supported Primitives in NIR
This document lists which primitives are supported by the software frameworks for conversion to and from NIR:
- `→`: Supported for conversion from NIR
- `←`: Supported for conversion to NIR
- `⟷`: Supported for both conversion directions (to and from NIR)
<br />
"""

full_md = static_md + "\n\n" + dynamic_md

with open("supported_nodes.md", "w", encoding="utf-8") as f:
    f.write(full_md)

# TODO Generate dedicated table of supported nodes for each framework
