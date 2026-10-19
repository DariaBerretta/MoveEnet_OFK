#!/usr/bin/env python3
"""Single-user local web bridge to the standalone C++ optical-flow engine.

Python dependencies: standard library only. The browser handles visualization.
Bind to 127.0.0.1 by default: use VS Code Port Forwarding or an SSH tunnel.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs

HERE = Path(__file__).resolve().parent
STATIC = HERE / 'static'
JOINTS = ['Head', 'Right shoulder', 'Left shoulder', 'Right elbow', 'Left elbow',
          'Left hip', 'Right hip', 'Right wrist', 'Left wrist', 'Right knee',
          'Left knee', 'Right ankle', 'Left ankle']
DEFAULT = {
    'data_file': '', 'dhp19': False, 'w': 640, 'h': 480,
    'net_period': 0.2, 'flow_period': 0.005, 'fps': 30,
    'gain_a': 1.4, 'gain_b': 1.0, 'compare': True,
    'roi': 20, 'of_joints': list(range(13)),
    'reset_on_detection': True, 'pu': 0.77, 'muD': 0.06,
    'muV': 0.97, 'use_lc': True, 'event_window_ms': 30,
    'device': 'cuda:0', 'movenet_script': '/usr/local/src/hpe-core/example/movenet/movenet_online.py',
    'checkpoint_path': '', 'python': 'python3'
}


def bounded_float(value, lo, hi, label):
    value = float(value)
    if not lo <= value <= hi:
        raise ValueError('{} must be in [{}, {}]'.format(label, lo, hi))
    return value


def validated(config):
    c = dict(DEFAULT)
    c.update(config)
    allowed = set(DEFAULT)
    if set(config) - allowed:
        raise ValueError('Unexpected setting: {}'.format(set(config) - allowed))
    path = Path(str(c['data_file'])).expanduser()
    if not path.is_absolute() or path.name != 'data.log' or not path.is_file():
        raise ValueError('Select an existing, absolute path ending in data.log on the server')
    c['data_file'] = str(path.resolve())
    for key, lo, hi in [('net_period', .001, 10), ('flow_period', .001, 1),
                        ('gain_a', 0, 4), ('gain_b', 0, 4), ('pu', 1e-7, 100),
                        ('muD', 1e-7, 100), ('muV', 1e-7, 100)]:
        c[key] = bounded_float(c[key], lo, hi, key)
    c['fps'] = int(c['fps'])
    if c['fps'] not in (30, 60):
        raise ValueError('Rendering FPS must be 30 or 60')
    c['roi'] = int(c['roi'])
    if not 3 <= c['roi'] <= 120:
        raise ValueError('ROI radius must be between 3 and 120')
    for dim in ('w', 'h'):
        c[dim] = int(c[dim])
        if not 32 <= c[dim] <= 2048:
            raise ValueError('Invalid resolution: {}'.format(dim))
    c['event_window_ms'] = int(c['event_window_ms'])
    if not 1 <= c['event_window_ms'] <= 500:
        raise ValueError('event_window_ms must be 1-500')
    if type(c['of_joints']) is not list or any(type(x) is not int or x < 0 or x >= 13 for x in c['of_joints']):
        raise ValueError('of_joints must be a list of indices 0..12')
    c['of_joints'] = sorted(set(c['of_joints']))
    for key in ('dhp19', 'compare', 'reset_on_detection', 'use_lc'):
        if not isinstance(c[key], bool):
            raise ValueError('{} must be boolean'.format(key))
    if c['dhp19']:
        c['w'], c['h'] = 346, 260
    for key in ('device', 'checkpoint_path', 'movenet_script', 'python'):
        c[key] = str(c[key])
    return c


class ReplayManager:
    def __init__(self, executable):
        self.executable = str(Path(executable).expanduser().resolve())
        self.lock = threading.RLock()
        self.restart_lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.proc = None
        self.generation = 0
        self.seq = 0
        self.frame = None
        self.duration = 0.0
        self.cursor = 0.0
        self.progress = 0.0
        self.ready = False
        self.busy = False
        self.finished = False
        self.error = ''
        self.settings = None
        self.paused = False
        self.speed = 1.0
        self.current_file = ''
        self.log = None

    def state(self):
        with self.lock:
            return {'seq': self.seq, 'generation': self.generation, 'duration': self.duration,
                    'time': self.cursor, 'progress': self.progress, 'ready': self.ready,
                    'busy': self.busy, 'finished': self.finished, 'error': self.error,
                    'paused': self.paused, 'speed': self.speed, 'file': self.current_file,
                    'settings': self.settings, 'joints': JOINTS}

    def stop_child(self):
        with self.lock:
            p = self.proc
            self.proc = None
        if p is None:
            return
        if p.poll() is None:
            try:
                p.stdin.write('QUIT\n')
                p.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                p.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                # start_new_session ensures these signals target ONLY our subprocess tree.
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait(timeout=1)
        for stream in (p.stdin, p.stdout):
            try:
                stream.close()
            except OSError:
                pass

    def _args(self, config, target, duration):
        args = [self.executable, '--data_file', config['data_file']]
        if config['dhp19']:
            args.append('--dhp19')
        else:
            args += ['--w', str(config['w']), '--h', str(config['h'])]
        mapping = {'net_period': 'net_period', 'flow_period': 'flow_period',
                   'fps': 'fps', 'gain_a': 'gain_a', 'gain_b': 'gain_b',
                   'roi': 'roi', 'pu': 'pu', 'muD': 'muD', 'muV': 'muV',
                   'use_lc': 'use_lc', 'compare': 'compare',
                   'reset_on_detection': 'reset_on_detection',
                   'event_window_ms': 'event_window_ms', 'device': 'device',
                   'movenet_script': 'movenet_script', 'python': 'python'}
        for key, arg in mapping.items():
            value = config[key]
            args += ['--' + arg, ('1' if value else '0') if isinstance(value, bool) else str(value)]
        args += ['--of_joints', ','.join(map(str, config['of_joints'])) or 'none',
                 '--start_at', str(max(0, target))]
        if duration > 0:
            args += ['--duration_hint', str(duration)]
        if config['checkpoint_path']:
            args += ['--checkpoint_path', config['checkpoint_path']]
        return args

    def restart(self, config=None, target=None, paused=None):
        with self.restart_lock:
            return self._restart(config, target, paused)

    def _restart(self, config=None, target=None, paused=None):
        # Validate BEFORE stopping the running engine; a bad setting must not
        # destroy the previous valid session.
        with self.lock:
            if config is None:
                if self.settings is None:
                    raise ValueError('Open a data.log first')
                c = self.settings
            else:
                c = validated(config)
        self.stop_child()
        with self.lock:
            file_changed = self.current_file != c['data_file']
            if file_changed:
                self.duration = 0.0
            if target is None:
                target = 0.0 if file_changed else self.cursor
            target = max(0.0, min(float(target), self.duration if self.duration else float(target)))
            if self.duration > 0 and target >= self.duration:
                target = max(0.0, self.duration - 1.0 / c['fps'])
            if paused is not None:
                self.paused = bool(paused)
            self.settings = c
            self.current_file = c['data_file']
            self.cursor = target
            self.progress = 0.0
            self.busy = True
            self.ready = False
            self.finished = False
            self.error = ''
            self.generation += 1
            g = self.generation
            args = self._args(c, target, self.duration)
            self.seq += 1
            self.changed.notify_all()
        # YARP name server may take a short moment to unregister the owned ports.
        time.sleep(0.15)
        if not os.path.isfile(self.executable):
            with self.lock:
                self.busy = False
                self.error = 'C++ engine not compiled: {}'.format(self.executable)
            return
        logname = '/tmp/of_gui_engine_{}.log'.format(os.getpid())
        log = open(logname, 'a', buffering=1)
        try:
            p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                 stderr=log, encoding='utf-8', errors='replace', bufsize=1,
                                 start_new_session=True)
        except OSError as exc:
            log.close()
            with self.lock:
                self.busy = False
                self.error = str(exc)
            return
        with self.lock:
            self.proc = p
            self.log = log
            self.frame = None
            self._send('SPEED {}'.format(self.speed))
        threading.Thread(target=self._reader, args=(p, g, log), daemon=True).start()

    def _send(self, command):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.write(command + '\n')
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                pass

    def command(self, action, value=None):
        if action == 'pause':
            with self.lock:
                self.paused = bool(value)
                self._send('PAUSE ' + ('1' if self.paused else '0'))
        elif action == 'speed':
            s = bounded_float(value, .25, 4, 'speed')
            with self.lock:
                self.speed = s
                self._send('SPEED {}'.format(s))
        else:
            raise ValueError('Unknown control action')
        return self.state()

    def _reader(self, p, generation, log):
        try:
            for line in p.stdout:
                tag, sep, data = line.partition(' ')
                if not sep or tag not in ('META', 'FRAME', 'PROGRESS', 'DONE', 'ERROR'):
                    continue
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                with self.lock:
                    if generation != self.generation:
                        break
                    if tag == 'META':
                        self.duration = float(obj['duration'])
                        self.busy = True
                    elif tag == 'PROGRESS':
                        self.progress = float(obj['time'])
                    elif tag == 'FRAME':
                        self.frame = obj
                        self.cursor = float(obj['time'])
                        self.seq += 1
                        self.busy = False
                        self.ready = True
                        if self.paused and self.proc:
                            # Let engine show requested frame, then pause before next.
                            self._send('PAUSE 1')
                        self.changed.notify_all()
                    elif tag == 'ERROR':
                        self.error = obj.get('message', 'Unknown C++ engine error')
                        self.busy = False
                        self.ready = False
                        self.changed.notify_all()
                    elif tag == 'DONE':
                        self.finished = True
                        self.busy = False
                        self.changed.notify_all()
            code = p.wait()
            with self.lock:
                if generation == self.generation and code != 0 and not self.error:
                    self.error = 'Engine stopped (code {}). Check /tmp/of_gui_engine_{}.log'.format(code, os.getpid())
                    self.busy = False
                    self.changed.notify_all()
        finally:
            log.close()

    def get_frame(self, after, timeout=1.2):
        with self.changed:
            if self.seq <= after and not self.finished and not self.error:
                self.changed.wait(timeout)
            if self.frame is None:
                return {'seq': self.seq, 'frame': None, 'state': self.state()}
            return {'seq': self.seq, 'frame': self.frame if self.seq > after else None,
                    'state': self.state()}

    def close(self):
        self.stop_child()


class Handler(BaseHTTPRequestHandler):
    manager = None
    browse_root = '/data'
    protocol_version = 'HTTP/1.1'

    def log_message(self, fmt, *args):
        if '/api/frame' not in args[0]:
            super().log_message(fmt, *args)

    def write_bytes(self, data, kind='application/json', code=200, extra=None):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if extra:
            for key, value in extra.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def json_response(self, obj, code=200):
        self.write_bytes(json.dumps(obj, ensure_ascii=False).encode('utf-8'), code=code)

    def read_json(self):
        size = int(self.headers.get('Content-Length', '0'))
        if not 0 < size < 65536:
            raise ValueError('Invalid JSON request size')
        return json.loads(self.rfile.read(size))

    def do_GET(self):
        try:
            url = urlsplit(self.path)
            if url.path == '/':
                return self.serve_file(STATIC / 'index.html', 'text/html; charset=utf-8')
            if url.path == '/static/app.js':
                return self.serve_file(STATIC / 'app.js', 'text/javascript; charset=utf-8')
            if url.path == '/static/style.css':
                return self.serve_file(STATIC / 'style.css', 'text/css; charset=utf-8')
            if url.path == '/api/status':
                return self.json_response(self.manager.state())
            if url.path == '/api/frame':
                args = parse_qs(url.query)
                after = int(args.get('after', ['-1'])[0])
                return self.json_response(self.manager.get_frame(after))
            if url.path == '/api/inspect':
                query = parse_qs(url.query)
                location = os.path.abspath(query.get('path', [''])[0])
                valid = os.path.isfile(location) and os.path.basename(location) == 'data.log'
                return self.json_response({'path': location, 'is_file': valid})
            if url.path == '/api/list':
                query = parse_qs(url.query)
                location = os.path.abspath(query.get('path', [self.browse_root])[0])
                if not os.path.isdir(location):
                    raise ValueError('Directory does not exist: ' + location)
                items = []
                with os.scandir(location) as entries:
                    for entry in entries:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                items.append({'name': entry.name, 'path': entry.path, 'kind': 'dir'})
                            elif entry.name == 'data.log' and entry.is_file():
                                items.append({'name': entry.name, 'path': entry.path, 'kind': 'file'})
                        except (PermissionError, OSError):
                            continue
                items.sort(key=lambda item: (item['kind'] != 'dir', item['name'].casefold()))
                total = len(items)
                return self.json_response({'path': location, 'parent': os.path.dirname(location),
                                           'items': items[:2000], 'truncated': total > 2000})
            if url.path == '/api/search':
                query = parse_qs(url.query)
                location = os.path.abspath(query.get('path', [self.browse_root])[0])
                needle = query.get('q', [''])[0].strip().casefold()
                if not os.path.isdir(location):
                    raise ValueError('Directory does not exist: ' + location)
                # Bounded recursive lookup prevents huge mounted datasets from
                # holding an HTTP worker indefinitely. Search only data.log.
                found = []
                visited = 0
                deadline = time.monotonic() + 7.0
                truncated = False
                def on_walk_error(error):
                    pass
                for dirname, subdirs, filenames in os.walk(location, topdown=True, onerror=on_walk_error,
                                                             followlinks=False):
                    visited += 1
                    subdirs.sort(key=str.casefold)
                    if 'data.log' in filenames:
                        candidate = os.path.join(dirname, 'data.log')
                        if needle in candidate.casefold():
                            found.append({'name': os.path.basename(dirname) + '/data.log',
                                          'path': candidate, 'kind': 'file'})
                    if len(found) >= 150 or visited >= 10000 or time.monotonic() > deadline:
                        truncated = True
                        break
                return self.json_response({'path': location, 'items': found, 'truncated': truncated})
            self.json_response({'error': 'Not found'}, 404)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.json_response({'error': str(exc)}, 400)
        except (ConnectionResetError, BrokenPipeError):
            pass

    def serve_file(self, path, content_type):
        self.write_bytes(path.read_bytes(), content_type)

    def do_POST(self):
        try:
            url = urlsplit(self.path)
            if url.path == '/api/convert':
                return self.convert_recording()
            body = self.read_json()
            if url.path == '/api/open':
                c = dict(DEFAULT)
                c.update(body.get('settings', {}))
                self.manager.restart(c, target=0, paused=False)
            elif url.path == '/api/apply':
                self.manager.restart(body.get('settings'), target=body.get('time'),
                                     paused=body.get('paused'))
            elif url.path == '/api/seek':
                self.manager.restart(target=body['time'], paused=body.get('paused'))
            elif url.path == '/api/restart':
                self.manager.restart(target=0, paused=bool(body.get('paused', False)))
            elif url.path == '/api/control':
                return self.json_response(self.manager.command(body['action'], body.get('value')))
            else:
                return self.json_response({'error': 'Not found'}, 404)
            self.json_response(self.manager.state())
        except (ValueError, KeyError, TypeError, OSError) as exc:
            self.json_response({'error': str(exc)}, 400)

    def convert_recording(self):
        count = int(self.headers.get('Content-Length', '0'))
        if not 0 < count <= 300*1024*1024:
            raise ValueError('Recording too large (max 300 MiB)')
        if not shutil.which('ffmpeg'):
            raise ValueError('ffmpeg is required for MP4 export')
        with tempfile.TemporaryDirectory(prefix='of_gui_export_') as temp:
            src = Path(temp) / 'captured.webm'
            dst = Path(temp) / 'captured.mp4'
            with src.open('wb') as f:
                left = count
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        raise ValueError('Incomplete recording upload')
                    f.write(chunk)
                    left -= len(chunk)
            cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(src),
                   '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                raise ValueError('ffmpeg conversion failed: ' + result.stderr[-300:])
            self.write_bytes(dst.read_bytes(), 'video/mp4', extra={
                'Content-Disposition': 'attachment; filename="optical_flow_playback.mp4"'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--root', default='/data', help='Initial folder for the file browser')
    ap.add_argument('--engine', default=str(HERE / 'build' / 'of_sanity_gui_engine'))
    args = ap.parse_args()
    Handler.manager = ReplayManager(args.engine)
    Handler.browse_root = args.root
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print('Open http://{}:{} in your browser'.format(args.host, args.port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        Handler.manager.close()
        httpd.server_close()

if __name__ == '__main__':
    main()
