"""Small, explicit backend contracts. Environment-specific execution is injected.
The smoke backend checks manifest/variant wiring without inventing success.
"""
class SmokeBackend:
    def __init__(self, allow=False): self.allow=allow
    def run(self, episode, variant):
        return {'success':bool(self.allow),'steps':1 if self.allow else 0,'retries':0,'contract_ok':self.allow,'backend':'smoke'}

class CommandBackend:
    """Adapter for an external simulator/robot command.

    The command receives one JSON request file and must write one JSON response
    with success, steps, and optional evidence fields. Nonzero exit is an error.
    """
    def __init__(self, command, workdir=None): self.command=command;self.workdir=workdir
    def run(self, episode, variant):
        import subprocess,tempfile,json,os
        with tempfile.TemporaryDirectory() as d:
            req=os.path.join(d,'request.json');out=os.path.join(d,'response.json')
            json.dump({'episode':episode.__dict__,'variant':variant,'output':out},open(req,'w'))
            p=subprocess.run(self.command+[req],cwd=self.workdir,capture_output=True,text=True)
            if p.returncode: raise RuntimeError(p.stderr[-2000:] or f'command exit {p.returncode}')
            return json.load(open(out))
