#!/usr/bin/env python3

import re
import sys
from pathlib import Path


source = Path(sys.argv[1]).read_text()
nullable_ref = re.compile(
    r"(?P<indent>^[ \t]*)anyOf:\n"
    r"(?P=indent)  - \$ref: (?P<ref>[^\n]+)\n"
    r"(?P=indent)  - type: ['\"]?null['\"]?(?=\n)",
    re.MULTILINE,
)
generated, substitutions = nullable_ref.subn(
    lambda match: f"{match.group('indent')}$ref: {match.group('ref')}",
    source,
)
if substitutions == 0:
    raise RuntimeError("expected at least one nullable reference schema")
Path(sys.argv[2]).write_text(generated)
