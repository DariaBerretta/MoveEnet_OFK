#!/usr/bin/env python3
"""Add move-iit-hpe support to the MoveEnet_OFK Optical Flow Lab GUI.

Run from the MoveEnet_OFK repository root:
    python3 /path/to/apply_moveiit_gui.py --check
    python3 /path/to/apply_moveiit_gui.py

The script validates all expected source snippets before changing any file.
It targets the main-branch GUI code inspected on 2026-10-30.
"""

from argparse import ArgumentParser
from pathlib import Path
import difflib
import shutil
import subprocess
import sys


def replace_one(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise ValueError(f'{label}: expected one matching source block, found {count}. No files were changed.')
    return text.replace(old, new, 1)


def transform_engine(text):
    text = replace_one(text, '#include <filesystem>\n#include <iomanip>',
                       '#include <filesystem>\n#include <fstream>\n#include <iomanip>', 'engine: include')
    text = replace_one(text, '    bool dhp19=false;\n    int width=640, height=480;',
                       '    bool dhp19=false, moveiit=false;\n    int width=640, height=480;', 'engine: config')
    text = replace_one(text,
        '    c.dhp19=rf.check("dhp19");\n'
        '    c.width=c.dhp19 ? 346:rf.check("w",Value(640)).asInt32();\n'
        '    c.height=c.dhp19 ? 260:rf.check("h",Value(480)).asInt32();',
        '    c.dhp19=rf.check("dhp19");\n'
        '    c.moveiit=rf.check("moveiit");\n'
        '    if(c.dhp19 && c.moveiit) throw std::runtime_error("--dhp19 and --moveiit are mutually exclusive");\n'
        '    c.width=c.dhp19 ? 346:(c.moveiit ? 1280:rf.check("w",Value(640)).asInt32());\n'
        '    c.height=c.dhp19 ? 260:(c.moveiit ? 720:rf.check("h",Value(480)).asInt32());', 'engine: dimensions')
    text = replace_one(text, 'void printProgress(double elapsed) {',
        '// YARP data.log begins with an ASCII packet header: ID timestamp AE ... .\n'
        '// Read the origin independently of offlineLoader: an unread iterator\n'
        '// may be empty. ATIS recordings can have absolute Unix timestamps.\n'
        'double recordingStartTimestamp(const std::string& path,bool requireHeader) {\n'
        '    std::ifstream file(path,std::ios::binary);\n'
        '    std::string packetId,packetType;\n'
        '    double timestamp=0.0;\n'
        '    if(file && (file>>packetId>>timestamp>>packetType) &&\n'
        '       packetType=="AE" && std::isfinite(timestamp)) return timestamp;\n'
        '    if(requireHeader)\n'
        '        throw std::runtime_error("Cannot read the first AE packet timestamp from move-iit-hpe data.log: "+path);\n'
        '    return 0.0; // Keep existing zero-origin fallback for other recordings.\n'
        '}\n'
        'void printProgress(double elapsed) {', 'engine: timestamp parser')
    text = replace_one(text,
        '        if(!replay.load(c.input))throw std::runtime_error("Failed to load data.log");\n'
        '        double firstTs=0,lastTs=0;',
        '        if(!replay.load(c.input))throw std::runtime_error("Failed to load data.log");\n'
        '        const double eventStartTs=recordingStartTimestamp(c.input,c.moveiit);\n'
        '        double firstTs=eventStartTs,lastTs=eventStartTs;', 'engine: start time')
    text = replace_one(text, '            double nxt=0;',
                       '            double nxt=eventStartTs+1e-6;', 'engine: metadata scan')
    text = replace_one(text,
        '        double detTs=-1e30,netNext=0,flowNext=0,packetNext=0,firstPacketTs=0;',
        '        double detTs=-1e30,netNext=0,flowNext=0,packetNext=eventStartTs+1e-6,firstPacketTs=0;',
        'engine: replay origin')
    text = replace_one(text,
        '            double ts=replay.begin().timestamp();\n'
        '            if(!hasFirstPacket){',
        '            const double packetAbsTs=replay.begin().timestamp();\n'
        '            const double ts=packetAbsTs-eventStartTs; // Relative dataset seconds for SAE, KF, playback\n'
        '            if(!std::isfinite(ts) || ts < -EPS)\n'
        '                throw std::runtime_error("Invalid packet timestamp or recording time origin");\n'
        '            if(!hasFirstPacket){', 'engine: timestamp normalization')
    text = replace_one(text, '            packetNext=ts+1e-6;packets++;',
                       '            packetNext=packetAbsTs+1e-6;packets++;', 'engine: cursor')
    return text


def transform_server(text):
    text = replace_one(text, "    'data_file': '', 'dhp19': False, 'w': 640, 'h': 480,",
                       "    'data_file': '', 'dhp19': False, 'moveiit': False, 'w': 640, 'h': 480,",
                       'server: defaults')
    text = replace_one(text, "    for key in ('dhp19', 'compare', 'reset_on_detection', 'use_lc'):",
                       "    for key in ('dhp19', 'moveiit', 'compare', 'reset_on_detection', 'use_lc'):",
                       'server: booleans')
    text = replace_one(text,
        "    if c['dhp19']:\n        c['w'], c['h'] = 346, 260",
        "    if c['dhp19'] and c['moveiit']:\n"
        "        raise ValueError('DHP19 and move-iit-hpe modes are mutually exclusive')\n"
        "    if c['dhp19']:\n        c['w'], c['h'] = 346, 260\n"
        "    elif c['moveiit']:\n        c['w'], c['h'] = 1280, 720",
        'server: validation')
    text = replace_one(text,
        "        if config['dhp19']:\n            args.append('--dhp19')\n"
        "        else:\n            args += ['--w', str(config['w']), '--h', str(config['h'])]",
        "        if config['dhp19']:\n            args.append('--dhp19')\n"
        "        elif config['moveiit']:\n            args.append('--moveiit')\n"
        "        else:\n            args += ['--w', str(config['w']), '--h', str(config['h'])]",
        'server: engine args')
    compile(text, 'server.py', 'exec')
    return text


def transform_js(text):
    text = replace_one(text,
        "  const isDhp = d==='dhp19'||(d==='auto'&&path.toLowerCase().includes('dhp19'));\n"
        "  return {\n    data_file:path, dhp19:isDhp, w:+$('sensorW').value, h:+$('sensorH').value,",
        "  const pathLower=path.toLowerCase();\n"
        "  const isDhp=d==='dhp19'||(d==='auto'&&pathLower.includes('dhp19'));\n"
        "  const isMoveiit=!isDhp&&(d==='moveiit'||(d==='auto'&&/move[-_]?iit|atis-s/.test(pathLower)));\n"
        "  return {\n    data_file:path, dhp19:isDhp, moveiit:isMoveiit, w:+$('sensorW').value, h:+$('sensorH').value,",
        'JS: dataset handling')
    text = replace_one(text,
        "  const vals={samplePath:s.data_file, dataset:s.dhp19?'dhp19':'eh36m',sensorW:s.w,sensorH:s.h,",
        "  const vals={samplePath:s.data_file, dataset:s.dhp19?'dhp19':s.moveiit?'moveiit':'eh36m',sensorW:s.w,sensorH:s.h,",
        'JS: fill form')
    text = replace_one(text,
        "  const dhp=d==='dhp19'||(d==='auto'&&path.includes('dhp19'));\n"
        "  $('sensorW').value=dhp?346:640;$('sensorH').value=dhp?260:480;\n"
        "  $('pu').value=dhp?0.36:0.77;$('muD').value=dhp?0.01:0.06;$('muV').value=dhp?0.01:0.97;",
        "  const dhp=d==='dhp19'||(d==='auto'&&path.includes('dhp19'));\n"
        "  const moveiit=!dhp&&(d==='moveiit'||(d==='auto'&&/move[-_]?iit|atis-s/.test(path)));\n"
        "  $('sensorW').value=dhp?346:moveiit?1280:640;$('sensorH').value=dhp?260:moveiit?720:480;\n"
        "  $('pu').value=dhp?0.36:0.77;$('muD').value=dhp?0.01:0.06;$('muV').value=dhp?0.01:0.97;\n"
        "  $('roiSize').value=moveiit?35:20;\n"
        "  const defaultGain=moveiit?1.0:1.4;\n"
        "  $('gainA').value=defaultGain;$('gainANum').value=defaultGain;\n"
        "  $('gainATag').textContent=defaultGain.toFixed(2);",
        'JS: presets')
    if shutil.which('node'):
        result = subprocess.run(['node', '--check'], input=text, text=True, capture_output=True)
        if result.returncode:
            raise ValueError('JavaScript syntax error: ' + result.stderr)
    return text


def transform_html(text):
    return replace_one(text,
        '<option value="dhp19">DHP19 · 346×260</option><option value="eh36m">eH36M / custom</option>',
        '<option value="dhp19">DHP19 · 346×260</option><option value="moveiit">move-iit-hpe · 1280×720</option><option value="eh36m">eH36M / custom</option>',
        'HTML: dataset selector')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.', help='MoveEnet_OFK repository root (default: current directory)')
    parser.add_argument('--check', action='store_true', help='Validate and print the diff without editing files')
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    base = 'moveEnetFlow/utils/of_gui/'
    transforms = {
        base + 'engine.cpp': transform_engine,
        base + 'server.py': transform_server,
        base + 'static/app.js': transform_js,
        base + 'static/index.html': transform_html,
    }
    updates = {}
    try:
        for relative, transform in transforms.items():
            p = root / relative
            old = p.read_text(encoding='utf-8')
            new = transform(old)
            updates[p] = (old, new, relative)
    except (OSError, ValueError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1

    if args.check:
        for old, new, relative in updates.values():
            print(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                     fromfile='a/' + relative, tofile='b/' + relative)), end='')
        print('\nCHECK OK: no files changed.', file=sys.stderr)
    else:
        for path, (_, new, relative) in updates.items():
            path.write_text(new, encoding='utf-8')
            print('Updated: ' + relative)
        print('Completed. Rebuild the GUI C++ engine and restart the Python GUI server.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
