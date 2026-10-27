import json,os,subprocess,sys,time
from pathlib import Path

ready_log,monitor_log=map(Path,sys.argv[1:3])
assert sys.argv[3]=='--' and len(sys.argv)>4
cores={5,6,7,8,9,15,16,17,18,19}
assert cores<=os.sched_getaffinity(0)
proc=subprocess.Popen(sys.argv[4:])
pinned=False
helpers=set()
with monitor_log.open('a',buffering=1) as out:
 while proc.poll() is None:
  try:
   if not pinned and ready_log.exists() and 'MiniMax graph ctx=' in ready_log.read_text():
    os.sched_setaffinity(proc.pid,{5})
    pinned=True
    out.write(json.dumps({'event':'pinned','time':time.time(),'pid':proc.pid,'main_core':5,'helper_cores':sorted(cores)})+'\n')
   if pinned:
    base=Path(f'/proc/{proc.pid}')
    tasks=list((base/'task').iterdir())
    for task in tasks:
     tid=int(task.name)
     if tid!=proc.pid and tid not in helpers:
      os.sched_setaffinity(tid,cores)
      helpers.add(tid)
    stat=(base/'stat').read_text().split(') ',1)[1].split(); cpu=int(stat[36])
    # Affinity takes effect before the next scheduling turn.
    mask=os.sched_getaffinity(proc.pid); assert mask=={5}
    current=int(Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq').read_text())
    out.write(json.dumps({'event':'sample','time':time.time(),'cpu':cpu,'current_khz':current,'main_allowed':sorted(mask),'helpers':len(tasks)-1})+'\n')
  except (FileNotFoundError,ProcessLookupError):
   pass
  time.sleep(0.5 if pinned else 0.02)
 status=proc.wait()
 out.write(json.dumps({'event':'exit','time':time.time(),'status':status,'pinned':pinned})+'\n')
if not pinned and status==0:
 raise SystemExit('model never reached graph-ready marker')
raise SystemExit(status)
