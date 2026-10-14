#!/usr/bin/env python3
"""Two fresh guarded servers: image disk KV and real HTTP tool/image flows."""
import argparse
import array
import base64
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = Path('/home/sunghoon/workspace/ds4-exaone/models/YoungAi-DeepSeek-V4.1-Flash')
STEM = 'DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative'
MODEL_ID = 'ds41-pr83-api'
OUTPUTS = 32
MEM_BUDGET_MB = 110000
MIN_CACHE_TOKENS = 8
DISK_MB = 1024
IO_BYTES = 1024 * 1024
HTTP_OK = 200
ROUTES = {'chat': '/v1/chat/completions', 'anthropic': '/v1/messages', 'responses': '/v1/responses'}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(IO_BYTES), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    require(result.returncode == 0, f'{argv[0]} failed: {result.stderr.strip()}')
    return result.stdout.strip()


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def gpu_owners():
    raw = command(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    require(not raw or all(line.strip().isdigit() for line in raw.splitlines()), f'unrecognized GPU owner list: {raw}')
    return [int(line) for line in raw.splitlines()]


def file_identity(path):
    st = path.stat()
    return {'path': str(path), 'dev': st.st_dev, 'inode': st.st_ino, 'bytes': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'ctime_ns': st.st_ctime_ns}


def gguf_vocab(path):
    # Read metadata only. Do not infer vocabulary width from the logits dump.
    with path.open('rb') as source:
        def number(fmt):
            size = struct.calcsize(fmt)
            data = source.read(size)
            require(len(data) == size, 'truncated GGUF metadata')
            return struct.unpack(fmt, data)[0]

        def string():
            size = number('<Q')
            require(size <= 16 * IO_BYTES, 'unbounded GGUF string')
            data = source.read(size)
            require(len(data) == size, 'truncated GGUF string')
            return data

        def skip(kind):
            widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            if kind in widths:
                source.seek(widths[kind], 1)
            elif kind == 8:
                source.seek(number('<Q'), 1)
            elif kind == 9:
                element, count = number('<I'), number('<Q')
                if element in widths:
                    source.seek(widths[element] * count, 1)
                else:
                    require(element == 8 and count <= 1000000, 'unsupported GGUF metadata array')
                    for _ in range(count):
                        skip(element)
            else:
                raise RuntimeError(f'unknown GGUF metadata type {kind}')

        require(source.read(4) == b'GGUF' and number('<I') in (2, 3), 'expected GGUF v2/v3')
        number('<Q')
        count = number('<Q')
        require(count <= 1000000, 'unbounded GGUF metadata count')
        for _ in range(count):
            key, kind = string(), number('<I')
            if key == b'tokenizer.ggml.tokens':
                require(kind == 9 and number('<I') == 8, 'token table must be a string array')
                vocab = number('<Q')
                require(1000 < vocab <= 1000000, f'invalid vocabulary width {vocab}')
                return vocab
            skip(kind)
    raise RuntimeError('GGUF has no tokenizer.ggml.tokens')


def logits(path, vocab):
    size = path.stat().st_size
    require(size > 0 and size % (vocab * 4) == 0, f'invalid logits extent: {size} bytes, V={vocab}')
    bad = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(IO_BYTES), b''):
            words = array.array('I')
            words.frombytes(block)
            if sys.byteorder != 'little':
                words.byteswap()
            bad += sum((word & 0x7f800000) == 0x7f800000 for word in words)
    require(bad == 0, f'{path.name}: {bad} nonfinite logits')
    return {'vocab': vocab, 'rows': size // (vocab * 4), 'bytes': size, 'nonfinite': bad, 'sha256': sha(path)}


def image(surface, data):
    encoded = base64.b64encode(data).decode()
    if surface == 'chat':
        return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}}
    if surface == 'responses':
        return {'type': 'input_image', 'image_url': 'data:image/png;base64,' + encoded}
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': encoded}}


def text(surface, value):
    return {'type': 'input_text' if surface == 'responses' else 'text', 'text': value}


def body(surface, messages, outputs=OUTPUTS):
    request = {'model': MODEL_ID, 'temperature': 0, 'top_p': 1, 'stream': False}
    if surface == 'responses':
        request.update(input=messages, max_output_tokens=outputs, reasoning={'effort': 'none'})
    else:
        request.update(messages=messages, max_tokens=outputs)
        if surface == 'chat':
            request.update(reasoning_effort='none', seed=12345, return_token_ids=True)
        else:
            request['thinking'] = {'type': 'disabled'}
    return request


def response_view(surface, response):
    require(isinstance(response, dict) and not response.get('error'), f'API error: {response}')
    usage = response['usage']
    if surface == 'chat':
        require(response.get('object') == 'chat.completion', 'wrong Chat response object')
        choices = response['choices']
        require(len(choices) == 1, 'expected one Chat choice')
        choice = choices[0]
        require(choice['finish_reason'] in ('stop', 'length', 'tool_calls'), f'invalid Chat finish: {choice}')
        message = choice['message']
        require(message.get('role') == 'assistant', 'Chat role is not assistant')
        return {'prompt': usage['prompt_tokens'], 'completion': usage['completion_tokens'],
                'cached': usage['prompt_tokens_details']['cached_tokens'], 'finish': choice['finish_reason'],
                'text': message.get('content') or '', 'calls': message.get('tool_calls', [])}
    if surface == 'anthropic':
        require(response.get('type') == 'message' and response.get('role') == 'assistant', 'invalid Anthropic message')
        finish = response['stop_reason']
        require(finish in ('end_turn', 'max_tokens', 'tool_use', 'stop_sequence'), f'invalid Anthropic stop: {finish}')
        return {'prompt': usage['input_tokens'] + usage['cache_read_input_tokens'] + usage['cache_creation_input_tokens'],
                'completion': usage['output_tokens'], 'cached': usage['cache_read_input_tokens'], 'finish': finish,
                'text': ''.join(part.get('text', '') for part in response['content'] if part['type'] == 'text'),
                'calls': [part for part in response['content'] if part['type'] == 'tool_use']}
    require(response.get('object') == 'response', 'wrong Responses object')
    require(response.get('status') in ('completed', 'incomplete'), f'invalid Responses status: {response.get("status")}')
    if response['status'] == 'incomplete':
        require(response.get('incomplete_details', {}).get('reason') == 'max_output_tokens', 'unexpected incomplete Responses result')
    return {'prompt': usage['input_tokens'], 'completion': usage['output_tokens'],
            'cached': usage['input_tokens_details']['cached_tokens'], 'finish': response['status'],
            'text': ''.join(part.get('text', '') for item in response['output'] if item.get('type') == 'message'
                            for part in item.get('content', []) if part.get('type') == 'output_text'),
            'calls': [item for item in response['output'] if item.get('type') in ('function_call', 'custom_tool_call')]}


class Server:
    def __init__(self, gate, phase):
        self.gate, self.phase = gate, phase
        self.out = gate.out / phase
        self.out.mkdir()
        self.log_path = self.out / 'server.log'
        self.rows_path = self.out / 'current.f32'
        self.process = None
        self.telemetry_stop = threading.Event()
        self.telemetry_thread = None
        self.log_file = None
        self.image_id = None

    def telemetry(self):
        with (self.out / 'telemetry.jsonl').open('w') as target:
            while not self.telemetry_stop.is_set():
                entry = {'time': time.time()}
                try:
                    entry['gpu'] = command(['nvidia-smi', '--query-gpu=clocks.gr,utilization.gpu,memory.used,power.draw,temperature.gpu', '--format=csv,noheader,nounits'])
                    entry['meminfo'] = {line.split(':')[0]: line.split(':', 1)[1].strip()
                                        for line in Path('/proc/meminfo').read_text().splitlines()
                                        if line.startswith(('MemAvailable:', 'SwapFree:'))}
                except Exception as error:
                    entry['error'] = str(error)
                target.write(json.dumps(entry) + '\n')
                target.flush()
                self.telemetry_stop.wait(1)

    def get(self, path, receipt):
        with OPENER.open(self.gate.url + path, timeout=5) as response:
            data = response.read()
            require(response.status == HTTP_OK, f'GET {path}: HTTP {response.status}')
        (self.out / receipt).write_bytes(data)
        return json.loads(data)

    def start(self):
        require(not port_open(self.gate.args.port), f'port {self.gate.args.port} is already in use; no process was stopped')
        owners = gpu_owners()
        save(self.out / 'owners-before.json', owners)
        require(not owners, f'GPU already owned by {owners}; no process was stopped')
        require(sha(self.gate.binary) == self.gate.binary_sha, 'final server binary changed')
        require([file_identity(path) for path in self.gate.artifacts] == self.gate.identities, 'model/vision artifacts changed')
        args = self.gate.args
        model_root = args.model_root
        native = [str(self.gate.binary), '--cuda', '-m', str(self.gate.artifacts[0]),
                  '--vision-gguf', str(self.gate.artifacts[1]), '--engram-dir', str(model_root / 'deepseek-engram'),
                  '--zchain', str(model_root / (STEM + '-grrb-code_fit_n15360-engine')),
                  '--mem-budget-mb', str(MEM_BUDGET_MB), '--mem-floor-gb', '2', '--max-seqs', '1',
                  '--prefix-reuse', 'exact', '--no-dspark', '--emit-trace', '--host', '127.0.0.1',
                  '--port', str(args.port), '--model-id', MODEL_ID, '--no-update-check',
                  '--kv-disk-dir', str(self.gate.kv), '--kv-disk-space-mb', str(DISK_MB),
                  '--kv-cache-min-tokens', str(MIN_CACHE_TOKENS)]
        guard = [sys.executable, str(ROOT / 'tools/host_memory_guard.py'), '--max-gib', '128', '--high-gib', '128',
                 '--reserve-gib', '2', '--trip-gib', '2', '--psi-full-limit', '0', '--timeout', str(args.server_timeout),
                 '--grace-seconds', '5', '--log', str(self.out / 'guard.jsonl'), '--', *native]
        env = dict(os.environ, DS41_GEN_LOGITS=str(self.rows_path))
        save(self.out / 'command.json', {'argv': guard, 'cwd': str(ROOT),
             'environment': {key: value for key, value in env.items() if key.startswith(('DS4_', 'DS41_', 'CUDA_'))}})
        self.log_file = self.log_path.open('wb', buffering=0)
        self.process = subprocess.Popen(guard, cwd=ROOT, env=env, stdout=self.log_file, stderr=subprocess.STDOUT, start_new_session=True)
        (self.out / 'guard.pid').write_text(str(self.process.pid) + '\n')
        self.telemetry_thread = threading.Thread(target=self.telemetry, daemon=True)
        self.telemetry_thread.start()
        start = time.monotonic()
        while time.monotonic() - start < args.boot_timeout:
            require(self.process.poll() is None, f'{self.phase}: server exited during boot; see {self.log_path}')
            try:
                models = self.get('/v1/models', 'models.json')
                stats = self.get('/v1/stats', 'stats-before.json')
                break
            except (OSError, urllib.error.URLError):
                time.sleep(1)
        else:
            raise RuntimeError(f'{self.phase}: readiness timeout; see {self.log_path}')
        serving = stats['serving']
        effective = serving['effective']
        require(any(item['id'] == MODEL_ID for item in models['data']), 'wrong /v1/models identity')
        require(serving['family'] == 'deepseek4-v41-flash', 'wrong serving family')
        require(effective['disk'] is True and effective['max_seqs'] == 1 and effective['prefix_reuse'] == 'exact'
                and effective['mtp_mode'] == 'off', f'wrong effective configuration: {effective}')
        require(serving['requested']['backend'] == 'cuda', 'backend is not CUDA')
        boot = self.log_path.read_text(errors='replace')
        for required in ('vision tower wired', 'engram rows:', 'zchain mount: gr=', 'V4.1 route:', 'MTP=off',
                         'prefix_reuse=exact', f'KV disk cache {self.gate.kv}', 'engine=open host_vocab=yes'):
            require(required in boot, f'missing readiness evidence: {required}')
        found = re.search(r'placeholder id (\d+)', boot)
        require(found is not None, 'missing vision placeholder id')
        self.image_id = int(found[1])
        save(self.out / 'readiness.json', {'status': 'PASS', 'seconds': time.monotonic() - start,
             'guard_pid': self.process.pid, 'vision_token_id': self.image_id, 'effective': effective,
             'qualification_scope': 'Configuration only; workload checks follow.'})
        print(f'{self.phase}: ready after {time.monotonic() - start:.1f}s', flush=True)

    def close(self):
        if self.process is None:
            if self.log_file is not None:
                self.log_file.close()
            return
        was_alive = self.process.poll() is None
        if was_alive:
            self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            # The scope comes only from this guard's own start record.
            events = [json.loads(line) for line in (self.out / 'guard.jsonl').read_text().splitlines()]
            unit = next(event['unit'] for event in events if event['event'] == 'start')
            require(re.fullmatch(r'ds4-hostguard-\d+-\d+\.scope', unit) is not None, 'invalid task scope identity')
            subprocess.run(['systemctl', '--user', 'kill', '--signal=SIGKILL', '--kill-whom=all', unit], capture_output=True, timeout=5)
            self.process.kill()
            self.process.wait(timeout=5)
            self.gate.errors.append(f'{self.phase}: guard cleanup timed out')
        self.telemetry_stop.set()
        if self.telemetry_thread is not None:
            self.telemetry_thread.join(timeout=12)
        if self.log_file is not None:
            self.log_file.close()
        events = [json.loads(line) for line in (self.out / 'guard.jsonl').read_text().splitlines()]
        bad = [event for event in events if event['event'] in ('refused', 'cleanup_failed')
               or (event['event'] == 'terminate' and event.get('reason') != 'watchdog received a stop signal')]
        if bad or not was_alive:
            self.gate.errors.append(f'{self.phase}: unexpected guard exit/events: {bad or self.process.returncode}')
        deadline = time.monotonic() + 15
        owners = gpu_owners()
        while (port_open(self.gate.args.port) or owners) and time.monotonic() < deadline:
            time.sleep(0.25)
            owners = gpu_owners()
        save(self.out / 'shutdown.json', {'intentional_stop': was_alive, 'guard_exit': self.process.returncode,
             'unexpected_guard_events': bad, 'port_open': port_open(self.gate.args.port), 'gpu_owners_after': owners})
        require(not port_open(self.gate.args.port) and not owners, f'{self.phase}: port/GPU not clear after task cleanup')
        telemetry = [json.loads(line) for line in (self.out / 'telemetry.jsonl').read_text().splitlines()]
        require(any('gpu' in entry for entry in telemetry), f'{self.phase}: no observed GPU telemetry')

    def finish(self):
        stats = self.get('/v1/stats', 'stats-after.json')
        before = json.loads((self.out / 'stats-before.json').read_text())
        routes = {'chat': 'openai_chat_serial', 'anthropic': 'anthropic_messages_serial', 'responses': 'openai_responses_serial'}
        for surface, route in routes.items():
            expected = sum(row['phase'] == self.phase and row['surface'] == surface for row in self.gate.results)
            actual = stats['routes'][route] - before['routes'][route]
            require(actual == expected, f'{self.phase}: route {route} handled {actual}/{expected} requests')

    def call(self, name, surface, request, images, reuse=None, disk='any', exact_outputs=None):
        directory = self.gate.out / 'cases' / name
        directory.mkdir(parents=True)
        data = json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        (directory / 'request.json').write_bytes(data)
        offset = self.log_path.stat().st_size
        started = time.perf_counter()
        result = {'name': name, 'phase': self.phase, 'surface': surface, 'status': 'FAIL', 'expected_images': images,
                  'request_sha256': hashlib.sha256(data).hexdigest(), 'path': ROUTES[surface]}
        try:
            headers = {'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'}
            req = urllib.request.Request(self.gate.url + ROUTES[surface], data, headers)
            try:
                with OPENER.open(req, timeout=self.gate.args.request_timeout) as response:
                    result['http'] = response.status
                    result['headers'] = dict(response.headers)
                    raw = response.read()
            except urllib.error.HTTPError as error:
                result['http'], raw = error.code, error.read()
            finally:
                result['wall_s'] = time.perf_counter() - started
                with self.log_path.open('rb') as source:
                    source.seek(offset)
                    native = source.read().decode(errors='replace')
                (directory / 'native.log').write_text(native)
                if self.rows_path.exists():
                    self.rows_path.replace(directory / 'logits.f32')
            (directory / 'response.raw').write_bytes(raw)
            require(result['http'] == HTTP_OK, f'HTTP {result["http"]}: {raw[:2000].decode(errors="replace")}')
            try:
                response = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as error:
                raise RuntimeError(f'invalid JSON response: {error}; see response.raw') from error
            save(directory / 'response.json', response)
            view = response_view(surface, response)
            result.update(view)
            prompts = [(int(pos), int(token)) for pos, token in re.findall(r'^\[ptok\] (\d+) (-?\d+)$', native, re.M)]
            emitted = [(int(pos), int(token)) for pos, token in re.findall(r'^\[emit\] (\d+) (-?\d+)$', native, re.M)]
            require(prompts and [pos for pos, _ in prompts] == list(range(len(prompts))), 'missing/duplicate/noncontiguous native prompt trace')
            prompt_ids = [token for _, token in prompts]
            ids = [token for _, token in emitted]
            require(ids and [pos for pos, _ in emitted] == list(range(len(prompts), len(prompts) + len(ids))), 'missing/noncontiguous pure emit trace')
            require(all(0 <= token < self.gate.vocab for token in prompt_ids + ids), 'invalid native token ID')
            for file, tokens in [('prompt.ids', prompt_ids), ('tokens.ids', ids)]:
                (directory / file).write_text(''.join(f'{token}\n' for token in tokens))
            require(view['prompt'] == len(prompts), f'API prompt count {view["prompt"]} != native {len(prompts)}')
            require(view['completion'] > 0 and view['completion'] in (len(ids), len(ids) - 1), 'completion/emit count mismatch')
            requested = request.get('max_tokens', request.get('max_output_tokens'))
            require(len(ids) <= requested, 'emitted beyond requested token cap')
            if exact_outputs is not None:
                require(len(ids) == exact_outputs, f'early stop: {len(ids)}/{exact_outputs} emitted')
            if surface == 'chat':
                require(response.get('token_ids') == ids, 'returned token_ids differ from native emit trace')
            image_runs = sum(token == self.image_id and (i == 0 or prompt_ids[i - 1] != self.image_id)
                             for i, token in enumerate(prompt_ids))
            require(image_runs == images, f'image order/layout count {image_runs}/{images}')
            prefill = re.findall(r'\[v41\] prefill (\d+) token ([\d.]+)s', native)
            reused = re.findall(r'KV reuse (\d+)/(\d+) token .*prefill suffix (\d+)', native)
            require(len(prefill) == 1 and len(reused) <= 1, 'request did not run exactly one native generation')
            restored = int(reused[0][0]) if reused else 0
            require(view['cached'] == restored and int(prefill[0][0]) == len(prompts) - restored, 'API/native reuse counts disagree')
            if reused:
                require(int(reused[0][1]) == len(prompts) and int(reused[0][2]) == len(prompts) - restored, 'invalid native restore extent')
            if reuse is not None:
                require(restored == reuse, f'cached tokens {restored}, expected {reuse}')
            disk_loads = re.findall(r'KV disk checkpoint loaded \((\d+) token,', native)
            if disk == 'hit':
                require(disk_loads == [str(len(prompts))] and restored == len(prompts), 'fresh process did not import/restore the whole disk checkpoint')
            elif disk == 'miss':
                require(not disk_loads and restored == 0, 'unexpected image identity cache hit')
            require(not re.search(r'KV checkpoint does not copy|KV disk checkpoint did not load|argmax kernel .* != host|capture failed|rollback check FAIL|\[v41\].*(?:failed|refused)', native), 'native diagnostic failure; see native.log')
            require('[v41] DSpark:' not in native, 'speculation ran under --no-dspark')
            result['logits'] = logits(directory / 'logits.f32', self.gate.vocab)
            require(result['logits']['rows'] == len(ids), f'logits rows {result["logits"]["rows"]} != emitted {len(ids)}')
            result.update(emitted=len(ids), prompt_ids_sha256=sha(directory / 'prompt.ids'), token_ids_sha256=sha(directory / 'tokens.ids'),
                          disk_loaded_tokens=[int(value) for value in disk_loads], image_runs=image_runs,
                          vision_encoded=sum(map(int, re.findall(r'vision tower: (\d+) image\(s\)', native))),
                          vocab=self.gate.vocab, logit_rows=result['logits']['rows'], logits_sha256=result['logits']['sha256'],
                          native_prefill_s=float(prefill[0][1]), timings=response.get('timings'))
            if restored == 0:
                require(result['vision_encoded'] == images, f'cold request encoded {result["vision_encoded"]}/{images} images')
            if disk == 'hit':
                require(result['vision_encoded'] == 0, 'whole disk hit reran the vision tower')
            stats = self.get('/v1/stats', f'{name}.stats.json')
            require(stats['last_request']['speculation_active'] is False, 'stats reports active speculation')
            result['last_request'] = stats['last_request']
            require(view['text'] or view['calls'], 'empty assistant result')
            result['status'] = 'PASS'
            self.gate.results.append(result)
            print(json.dumps({key: result[key] for key in ('name', 'status', 'http', 'prompt', 'cached', 'emitted', 'finish', 'wall_s')}), flush=True)
            return {'response': response, 'result': result, 'prompt': prompt_ids, 'ids': ids, 'dir': directory}
        except Exception as error:
            result['error'] = str(error)
            self.gate.results.append(result)
            raise RuntimeError(f'{name}: {error}; receipts: {directory}') from error
        finally:
            save(directory / 'result.json', result)


class Gate:
    def __init__(self, args):
        self.args, self.out = args, args.out.resolve()
        self.binary = args.binary.resolve()
        self.binary_sha = sha(self.binary)
        self.artifacts = [args.model_root / (STEM + '.gguf'), args.model_root / 'DeepSeek-V4.1-Flash.vision.gguf']
        self.identities = [file_identity(path) for path in self.artifacts]
        self.vocab = gguf_vocab(self.artifacts[0])
        self.kv = self.out / 'kv'
        self.url = f'http://127.0.0.1:{args.port}'
        self.results, self.errors, self.comparisons = [], [], []

    @contextlib.contextmanager
    def server(self, phase):
        server = Server(self, phase)
        try:
            server.start()
            yield server
        finally:
            server.close()

    def disk_snapshot(self, name):
        files = [{'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in sorted(self.kv.glob('*.kv'))]
        save(self.out / f'{name}.disk.json', files)
        return files

    def compare_restart(self, cold, hit):
        check = {'name': 'disk-restart-exact', 'status': 'FAIL',
                 'request_bytes': (cold['dir'] / 'request.json').read_bytes() == (hit['dir'] / 'request.json').read_bytes(),
                 'expanded_prompt': cold['prompt'] == hit['prompt'], 'returned_tokens': cold['response']['token_ids'] == hit['response']['token_ids'],
                 'native_tokens': cold['ids'] == hit['ids'],
                 'all_logits': cold['result']['logits'] == hit['result']['logits'],
                 'returned_choices': cold['response']['choices'] == hit['response']['choices']}
        self.comparisons.append(check)
        require(all(value for key, value in check.items() if key not in ('name', 'status')), f'disk restart parity failed: {check}')
        check['status'] = 'PASS'

    def run(self):
        a = (self.args.inputs / 'image-a.png').read_bytes()
        b = (self.args.inputs / 'image-b.png').read_bytes()
        require(a != b and a[:8] == b[:8] == b'\x89PNG\r\n\x1a\n' and a[16:24] == b[16:24], 'identity fixtures must differ with equal PNG geometry')
        single = body('chat', [{'role': 'user', 'content': [text('chat', 'Disk image identity case. Study this image: '), image('chat', a),
                    text('chat', ' Write a detailed 2000-word essay about its colors and geometry. Begin now and continue at length.')]}])
        pair = body('chat', [{'role': 'user', 'content': [text('chat', 'Disk image ordering case. First: '), image('chat', a),
                    text('chat', ' Second: '), image('chat', b), text('chat', ' Write a detailed 2000-word comparison; continue at length.')]}])
        with self.server('server-a') as server:
            cold = server.call('disk-cold', 'chat', single, 1, reuse=0, disk='miss', exact_outputs=OUTPUTS)
            ordered = server.call('ordered-cold', 'chat', pair, 2, reuse=0, disk='miss', exact_outputs=OUTPUTS)
            require(len(self.disk_snapshot('after-a')) >= 2, 'process A did not export both image checkpoints')
            server.finish()
        require(not self.errors, '; '.join(self.errors))
        with self.server('server-b') as server:
            hit = server.call('disk-restart-hit', 'chat', single, 1, reuse=len(cold['prompt']), disk='hit', exact_outputs=OUTPUTS)
            self.compare_restart(cold, hit)
            changed = copy.deepcopy(single)
            changed['messages'][0]['content'][1] = image('chat', b)
            miss = server.call('changed-image-miss', 'chat', changed, 1, reuse=0, disk='miss', exact_outputs=OUTPUTS)
            require(miss['prompt'] == cold['prompt'], 'changed-image test altered expanded token geometry')
            reordered = copy.deepcopy(pair)
            reordered['messages'][0]['content'][1] = image('chat', b)
            reordered['messages'][0]['content'][3] = image('chat', a)
            miss = server.call('reordered-images-miss', 'chat', reordered, 2, reuse=0, disk='miss', exact_outputs=OUTPUTS)
            require(miss['prompt'] == ordered['prompt'], 'reordered-image test altered expanded token geometry')
            self.protocols(server, a, b)
            server.finish()
            self.disk_snapshot('after-b')
        require(not self.errors, '; '.join(self.errors))
        require(len(self.results) == 11 and all(row['status'] == 'PASS' for row in self.results), 'incomplete HTTP gate')
        require(sha(self.binary) == self.binary_sha, 'server binary changed during the gate')

    def protocols(self, server, a, b):
        # Replay actual model-issued calls; never fabricate a successful call.
        chat_messages = [{'role': 'system', 'content': 'Use the screenshot tool exactly once when asked. After its result, answer directly.'},
                         {'role': 'user', 'content': [text('chat', 'This is the first image. '), image('chat', a),
                          text('chat', ' Call screenshot now, with no arguments, to obtain the second image for comparison. Do not answer until the tool returns.')]}]
        tools = [{'type': 'function', 'function': {'name': 'screenshot', 'description': 'Return the second image for comparison.',
                  'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
        request = body('chat', chat_messages, 192)
        request.update(tools=tools, tool_choice='required')
        first = server.call('chat-image-tool-call', 'chat', request, 1)
        message = first['response']['choices'][0]['message']
        calls = message.get('tool_calls', [])
        require(first['result']['finish'] == 'tool_calls' and len(calls) == 1, 'Chat did not return exactly one complete tool call')
        call = calls[0]
        require(call.get('id') and call.get('type') == 'function' and call['function']['name'] == 'screenshot', 'invalid Chat screenshot call')
        require(json.loads(call['function']['arguments']) == {}, 'screenshot arguments do not match the empty-object schema')
        follow_messages = chat_messages + [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': [
            text('chat', 'Screenshot result: '), image('chat', b), text('chat', ' Compare it with the first image. Answer directly; do not call tools again.')]}]
        follow = body('chat', follow_messages, 64)
        follow.update(tools=tools, tool_choice='auto')
        continued = server.call('chat-tool-image-follow', 'chat', follow, 2)
        require(continued['result']['text'].strip() and not continued['result']['calls'], 'Chat tool follow-up has no final assistant text or called another tool')

        # Exact block shapes from crates/ds4-server/tests/parse.rs.
        use = {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'c1', 'name': 'shot', 'input': {}}]}
        nested = {'type': 'tool_result', 'tool_use_id': 'c1', 'content': [text('anthropic', 'nested:'), image('anthropic', b), text('anthropic', ':end')]}
        outer = image('anthropic', a)
        for name, blocks in [('anthropic-outer-then-tool', [outer, nested]), ('anthropic-tool-then-outer', [nested, outer])]:
            messages = [use, {'role': 'user', 'content': [*blocks, text('anthropic', ' Describe both images briefly. Do not call tools.')]}]
            result = server.call(name, 'anthropic', body('anthropic', messages, 64), 2)
            require(result['result']['text'].strip() and not result['result']['calls'], f'{name}: no final text or unexpected tool use')

        inputs = [{'type': 'message', 'role': 'user', 'content': [text('responses', 'Responses image extension case. Describe the supplied screenshot briefly.')]},
                  {'type': 'function_call', 'call_id': 'r1', 'name': 'shot', 'arguments': '{}'},
                  {'type': 'function_call_output', 'call_id': 'r1', 'output': [text('responses', 'before:'), image('responses', a), text('responses', ':after')]}]
        base = server.call('responses-tool-image', 'responses', body('responses', inputs, 64), 1)
        require(base['result']['text'].strip() and not base['result']['calls'], 'Responses base has no final text or unexpected tool call')
        # Explicit history is required; this tests the documented image-array
        # extension, not a separate previous_response_id session mechanism.
        assistant = {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': base['result']['text']}]}
        extension = inputs + [assistant, {'type': 'function_call', 'call_id': 'r2', 'name': 'shot', 'arguments': '{}'},
                     {'type': 'function_call_output', 'call_id': 'r2', 'output': [text('responses', 'New screenshot: '), image('responses', b),
                      text('responses', ' Compare both images directly. Do not call tools.')]}]
        extended = server.call('responses-tool-image-extension', 'responses', body('responses', extension, 64), 2, reuse=len(base['prompt']))
        require(extended['prompt'][:len(base['prompt'])] == base['prompt'], 'Responses extension did not retain the original expanded prefix')
        require(extended['result']['text'].strip() and not extended['result']['calls'], 'Responses extension has no final text or unexpected tool call')

    def report(self, status):
        save(self.out / 'comparisons.json', self.comparisons)
        fields = ['name', 'phase', 'surface', 'status', 'http', 'prompt', 'cached', 'emitted', 'finish', 'image_runs', 'vision_encoded',
                  'vocab', 'logit_rows', 'logits_sha256', 'native_prefill_s', 'wall_s']
        with (self.out / 'results.csv').open('w') as target:
            writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.results)
        save(self.out / 'report.json', {'status': status, 'errors': self.errors, 'generations': len(self.results),
             'server_loads': len(list(self.out.glob('server-*/guard.pid'))), 'vocab': self.vocab,
             'binary_sha256': self.binary_sha, 'comparisons': self.comparisons,
             'scope': 'Two-load image disk restore plus Chat/Anthropic/Responses tool-image contracts; timings include logits readback, no speed or broad image-quality claim.'})
        with (self.out / 'SHA256SUMS').open('w') as target:
            for path in sorted(self.out.rglob('*')):
                if path.is_file() and path.name != 'SHA256SUMS' and self.kv not in path.parents:
                    target.write(f'{sha(path)}  {path}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('binary', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--model-root', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'scratch/pr83-review/vision-gate/inputs')
    parser.add_argument('--port', type=int, default=8099)
    parser.add_argument('--boot-timeout', type=int, default=900)
    parser.add_argument('--request-timeout', type=int, default=600)
    parser.add_argument('--server-timeout', type=int, default=3600)
    parser.add_argument('--artifact-receipt', type=Path, help='Existing immutable model/Engram/GR verification receipt; copied and hashed, not regenerated')
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    args.model_root, args.inputs = args.model_root.resolve(), args.inputs.resolve()
    require(not args.out.exists() or not any(args.out.iterdir()), 'OUT must be fresh or empty; no files are deleted')
    require(args.binary.is_file() and os.access(args.binary, os.X_OK), 'server binary must exist and be executable')
    for key in ('DS41_NO_ENGRAM', 'DS41_VISION_REENCODE', 'DS41_FWD_TRACE', 'DS41_DUMP_HEAD', 'DS41_FEED_DIGEST'):
        require(key not in os.environ, f'unset conflicting diagnostic {key}')
    for directory in ('deepseek-engram', STEM + '-grrb-code_fit_n15360-engine'):
        require((args.model_root / directory).is_dir(), f'missing {directory}')
    args.out.mkdir(parents=True, exist_ok=True)
    gate = Gate(args)
    gate.kv.mkdir()
    (gate.out / 'driver.py').write_bytes(Path(__file__).read_bytes())
    if args.artifact_receipt:
        (gate.out / 'artifact-receipt.json').write_bytes(args.artifact_receipt.read_bytes())
    config = {'binary': str(gate.binary), 'binary_sha256': gate.binary_sha, 'driver_sha256': sha(Path(__file__)), 'model_artifacts': gate.identities,
              'artifact_identity_scope': 'GGUF stat identities; full model/Engram/GR hashes come from the optional existing artifact receipt.',
              'vocab_from_gguf_metadata': gate.vocab, 'port': args.port, 'disk_dir': str(gate.kv), 'disk_budget_mb': DISK_MB,
              'sampling': 'temperature=0, pure decode; --no-dspark', 'engram': 'enabled', 'overlay': 'GR',
              'clock_policy_mhz': '300-2200 already configured; never changed by this driver',
              'inputs': [dict(file_identity(args.inputs / name), sha256=sha(args.inputs / name)) for name in ('image-a.png', 'image-b.png')],
              'git_head': command(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']), 'command': sys.argv}
    save(gate.out / 'config.json', config)
    diff = subprocess.run(['git', '-C', str(ROOT), 'diff', '--binary'], capture_output=True, check=True).stdout
    (gate.out / 'source.diff').write_bytes(diff)
    status = 'FAIL'
    try:
        gate.run()
        status = 'PASS'
    except BaseException as error:
        gate.errors.append(f'{type(error).__name__}: {error}')
        (gate.out / 'failure.txt').write_text(traceback.format_exc())
        print(f'FAIL: {error}', file=sys.stderr, flush=True)
    finally:
        gate.report(status)
    print(json.dumps({'status': status, 'generations': len(gate.results), 'errors': gate.errors, 'out': str(gate.out)}), flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
