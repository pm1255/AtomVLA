"""Pinned full sources, validated size/SHA, resume Range with finite retries."""
import hashlib,json,os,time,requests
from pathlib import Path
from full_common import dump,sha

def download(item,repo,revision,root,progress):
    dest=Path(root)/repo.replace('/','__')/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
    receipt=dest.with_suffix(dest.suffix+'.verified.json')
    if dest.exists() and receipt.exists():
        r=json.loads(receipt.read_text())
        if dest.stat().st_size==item['size'] and r['sha256']==item['lfs']['oid']:return dest
        raise ValueError('Previously verified source size or expected hash changed')
    temp=dest.with_suffix(dest.suffix+'.partial');url=f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{item["path"]}'
    errors=[]
    for attempt in range(3):
        start=temp.stat().st_size if temp.exists() else 0
        try:
            if start<item['size']:
                with requests.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    if start and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):raise ValueError('Invalid resumed Content-Range')
                    if not start and response.status_code not in [200,206]:raise ValueError('Unexpected HTTP status')
                    last=0
                    with temp.open('ab' if start else 'wb') as f:
                        for block in response.iter_content(4*1024*1024):
                            f.write(block);start+=len(block)
                            if start>item['size']:raise ValueError('Download exceeds pinned source size')
                            if time.time()-last>10:
                                dump(progress,{'phase':'download','bytes':start,'total_bytes':item['size'],'source':item['path'],'time':time.time()});last=time.time()
            if temp.stat().st_size!=item['size']:raise ValueError('Incomplete download')
            digest=sha(temp)
            if digest!=item['lfs']['oid']:raise ValueError('Pinned source checksum mismatch')
            temp.replace(dest);dump(receipt,{'repo':repo,'revision':revision,'path':item['path'],'sha256':digest,'bytes':item['size'],'verified_at':time.time()})
            return dest
        except Exception as e:
            errors.append(str(e))
            if 'checksum' in str(e) or 'Range' in str(e) or attempt==2:raise RuntimeError(json.dumps(errors))
            time.sleep(2*(attempt+1))
    raise RuntimeError('download retry budget exhausted')
