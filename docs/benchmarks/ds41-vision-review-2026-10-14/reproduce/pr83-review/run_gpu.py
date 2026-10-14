import json, pathlib, subprocess, sys, threading, time
out=pathlib.Path(sys.argv[1]); cmd=sys.argv[2:]
out.mkdir(parents=True,exist_ok=False)
(out/'command.json').write_text(json.dumps(cmd,indent=2))
stop=threading.Event()
def watch():
 with (out/'telemetry.jsonl').open('w') as f:
  while not stop.is_set():
   gpu=subprocess.run(['nvidia-smi','--query-gpu=clocks.gr,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
   mem=dict(l.split(':',1) for l in pathlib.Path('/proc/meminfo').read_text().splitlines())
   f.write(json.dumps(dict(time=time.time(),gpu=gpu,available=mem['MemAvailable']))+'\n');f.flush();stop.wait(1)
thread=threading.Thread(target=watch);thread.start()
try:
 with (out/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (out/'exit.json').write_text(json.dumps(dict(exit=r.returncode)))
finally:stop.set();thread.join()
raise SystemExit(r.returncode)
