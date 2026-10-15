#!/usr/bin/env python3
"""Package only the reusable skill, leaving examples and development files out."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def package(destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = [ROOT / 'SKILL.md', ROOT / 'LICENSE']
    for folder in ('agents', 'references', 'assets'):
        files.extend(p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.append(ROOT / 'scripts/build_report.py')
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(files):
            archive.write(source, source.relative_to(ROOT).as_posix())
    return destination


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o','--output',type=Path,default=ROOT/'dist/code-to-report-skill.zip')
    args=parser.parse_args()
    print(package(args.output))
