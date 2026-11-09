"""Independent parent owns only its spawned process group; finite wall/idle budget."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wall-seconds',type=float,default=7200);p.add_argument('--idle-seconds',type=float,default=600)
    p.add_argument('--interval',type=float,default=15);p.add_argument('--no-gpu-sample',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER)
    p.add_argument('--progress-file',type=Path)
    a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:raise ValueError('Missing child command')
    root=a.root;(root/'logs').mkdir(parents=True,exist_ok=True);(root/'evidence').mkdir(exist_ok=True)
    start=time.time();child=subprocess.Popen(cmd,start_new_session=True)
    log=(root/'logs/watchdog.jsonl').open('a',buffering=1)
    writer=None;sample_step=0
    if not a.no_gpu_sample:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(root/'tensorboard/resource_monitor'))
    reason=None
    def terminate(sig,frame):
        nonlocal reason;reason=f'parent_signal_{sig}'
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    while child.poll() is None:
        now=time.time();progress=a.progress_file or root/'evidence/progress.json'
        last=max(start,progress.stat().st_mtime if progress.exists() else start)
        if now-start>a.wall_seconds:reason='wall_time_limit'
        elif now-last>a.idle_seconds:reason='no_progress_timeout'
        event={'time':now,'child_pid':child.pid,'elapsed_seconds':now-start,'progress_age_seconds':now-last}
        if not a.no_gpu_sample:
            try:
                q=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
                event['gpu_csv']=q.stdout.strip();event['gpu_query_rc']=q.returncode
                if writer:
                    for line in q.stdout.splitlines():
                        gpu,util,memory,temp=[float(v.strip()) for v in line.split(',')]
                        for name,value in [('utilization_percent',util),('memory_mib',memory),('temperature_c',temp)]:
                            writer.add_scalar(f'gpu_{int(gpu)}/{name}',value,sample_step)
                    writer.add_scalar('watchdog/progress_age_seconds',now-last,sample_step)
                    writer.flush();sample_step+=1
            except Exception as e:event['gpu_query_error']=str(e)
        log.write(json.dumps(event)+'\n')
        if reason:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
            break
        time.sleep(min(a.interval,a.idle_seconds/2))
    rc=child.wait();final={'event':'watchdog_finished','time':time.time(),'child_returncode':rc,'reason':reason,'released_by_parent_exit':True}
    log.write(json.dumps(final)+'\n');log.close()
    if writer:writer.close()
    (root/'evidence/watchdog_exit.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final),flush=True)
    sys.exit(124 if reason else rc)
if __name__=='__main__':main()
