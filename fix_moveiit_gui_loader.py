#!/usr/bin/env python3
"""Fix empty-recording false alarms in MoveEnetOFK Optical Flow Lab.

Compatible with the GUI previously patched by apply_moveiit_gui.py.
Uses ev::offlineLoader's getStartTime()/getLength(), removes the second
full-file metadata load, and makes packet cursor increments precision-safe.

Examples (from /workspace):
    python3 fix_moveiit_gui_loader.py --check
    python3 fix_moveiit_gui_loader.py

Examples (from /workspace/moveEnetFlow):
    python3 fix_moveiit_gui_loader.py --check
    python3 fix_moveiit_gui_loader.py

--check prints the patch without changing files. Applying creates a .bak
backup next to engine.cpp and does not touch the experimental runner.
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path


def locate_engine(root: Path) -> Path:
    roots = [root / 'moveEnetFlow/utils/of_gui/engine.cpp',
             root / 'utils/of_gui/engine.cpp',
             root / 'engine.cpp']
    found = [p for p in roots if p.is_file()]
    if len(found) != 1:
        raise RuntimeError('Could not uniquely locate GUI engine.cpp under ' + str(root) +
                           '\nExpected moveEnetFlow/utils/of_gui/engine.cpp or utils/of_gui/engine.cpp')
    return found[0]


def replace_exact(text: str, before: str, after: str, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise RuntimeError('{}: expected exactly one original block (found {}).'.format(label, count))
    return text.replace(before, after, 1)


def transform(source: str) -> str:
    if 'c.moveiit=rf.check("moveiit")' not in source:
        raise RuntimeError('GUI engine does not yet have --moveiit. First apply apply_moveiit_gui.py.')
    if 'const double eventStartTs=replay.getStartTime();' in source:
        raise RuntimeError('This fix appears to be already applied. Nothing changed.')
    # Require previous GUI patch so we do not silently mismatch its time domain.
    if 'const double ts=packetAbsTs-eventStartTs;' not in source:
        raise RuntimeError('Unexpected engine version: cannot find relative-time packet handling.')

    text = source
    # Previous script parsed raw packet headers, which need not have exactly the
    # timestamp representation that offlineLoader stores. Remove the helper.
    match = re.search(r'// YARP data\.log begins with an ASCII packet header: ID timestamp AE \.\.\. \.\n.*?(?=void printProgress\(double elapsed\) \{)', text, re.S)
    if not match:
        raise RuntimeError('Missing original recordingStartTimestamp() helper. No files changed.')
    if text[match.start():match.end()].count('recordingStartTimestamp(') != 1:
        raise RuntimeError('Unexpected timestamp helper layout. No files changed.')
    text = text[:match.start()] + text[match.end():]

    text = replace_exact(text, '#include <fstream>\n', '', 'unused ifstream header')
    text = replace_exact(text, '#include <iomanip>\n', '#include <iomanip>\n#include <limits>\n',
                         'limits header')

    start = '        const double eventStartTs=recordingStartTimestamp(c.input,c.moveiit);'
    end = '        double duration=c.durationHint>=0?c.durationHint:lastTs-firstTs;'
    a, b = text.find(start), text.find(end)
    if a < 0 or b < 0 or a >= b or text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError('Cannot find the expected metadata-scanning block. No files changed.')
    block = text[a:b + len(end)]
    if 'ev::offlineLoader<ev::AE> probe;' not in block or 'if(!found)throw std::runtime_error("Empty event recording")' not in block:
        raise RuntimeError('Metadata block differs from known previous patch. No files changed.')
    replacement = '''        // The event-driven library already knows the exact timestamp domain
        // and duration after load(). Never re-load/scan a potentially huge ATIS
        // recording just to discover its length. Its timestamps can be Unix time.
        const double eventStartTs=replay.getStartTime();
        const double availableDuration=replay.getLength();
        if (!std::isfinite(eventStartTs) || !std::isfinite(availableDuration) ||
            availableDuration <= 0.0) {
            throw std::runtime_error("Invalid or zero-duration event recording: " + c.input +
                                     " [" + replay.getinfo() + "]");
        }
        const double duration=c.durationHint>=0?c.durationHint:availableDuration;
        std::cerr << "[OF GUI] " << replay.getinfo()
                  << "; absolute start=" << std::setprecision(16) << eventStartTs
                  << "; duration=" << availableDuration << " s" << std::endl;'''
    text = text[:a] + replacement + text[b + len(end):]

    # `nextafter` always advances a timestamp even near ~1.7e9 Unix seconds.
    text = replace_exact(
        text,
        'double detTs=-1e30,netNext=0,flowNext=0,packetNext=eventStartTs+1e-6,firstPacketTs=0;',
        'double detTs=-1e30,netNext=0,flowNext=0,packetNext=std::nextafter(eventStartTs,std::numeric_limits<double>::infinity()),firstPacketTs=0;',
        'first event packet request')
    text = replace_exact(
        text,
        'if(replay.begin()==replay.end()){packetNext+=1e-6;continue;}',
        'if(replay.begin()==replay.end()){packetNext=std::nextafter(packetNext,std::numeric_limits<double>::infinity());continue;}',
        'empty packet window handling')
    text = replace_exact(
        text,
        'packetNext=packetAbsTs+1e-6;packets++;',
        'packetNext=std::nextafter(packetAbsTs,std::numeric_limits<double>::infinity());packets++;',
        'next packet request')
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='Repository root OR moveEnetFlow directory')
    parser.add_argument('--check', action='store_true', help='Show diff without making any change')
    ns = parser.parse_args()
    try:
        path = locate_engine(Path(ns.root).expanduser().resolve())
        before = path.read_text(encoding='utf-8')
        after = transform(before)
    except (RuntimeError, OSError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1

    if ns.check:
        sys.stdout.writelines(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                              fromfile=str(path) + ' (current)', tofile=str(path) + ' (patched)'))
        print('CHECK OK: no files changed.', file=sys.stderr)
    else:
        backup = path.with_suffix(path.suffix + '.bak')
        if backup.exists():
            raise RuntimeError('Backup already exists: {}. Move it aside to preserve the original before patching.'.format(backup))
        shutil.copy2(path, backup)
        path.write_text(after, encoding='utf-8')
        print('Updated: ' + str(path))
        print('Backup: ' + str(backup))
        print('Next: cmake -S utils/of_gui -B utils/of_gui/build && cmake --build utils/of_gui/build -j')
    return 0


if __name__ == '__main__':
    sys.exit(main())
