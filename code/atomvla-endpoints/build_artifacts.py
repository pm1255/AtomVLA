"""Export inspectable pilot labels, comparison videos and a concise report."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main(a):
    root=Path(a.root); deliver=root/'artifacts';deliver.mkdir(exist_ok=True)
    results=json.loads((root/'libero_results.json').read_text())
    rows=[]
    for path in sorted((root/'libero').glob('*/direction_annotations.json')):
        labels=json.loads(path.read_text())
        for s in labels['segments']:
            rows.append({'episode':path.parent.name,'segment_id':s['segment_id'],'primitive':s['primitive'],
                         'arm':s['arm'],'start_action':s['start'],'end_action_exclusive':s['end'],
                         'coordinate_frame':s['coordinate_frame'],'duration_s':s['duration_s'],
                         **{f'endpoint_{k}_m':s['endpoint_position'][j] for j,k in enumerate('xyz')},
                         **{f'direction_{k}':None if s['net_direction'] is None else s['net_direction'][j] for j,k in enumerate('xyz')},
                         'direction_valid':s['net_direction_valid'],'motion_type':s['motion_type'],
                         'path_length_m':s['path_length_m'],'net_distance_m':s['net_distance_m'],
                         'straightness':s['straightness'],'rotation_excursion_deg':s['rotation_excursion_deg'],
                         'boundary_source':s['boundary_source'],'semantic_verified':s['semantic_verified']})
    with (deliver/'libero_atomic_directions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregate={mode:{'successes':sum(r['conditions'][mode]['final_native_success'] for r in results),'n':len(results)}
               for mode in ['source','raw_action_replay','endpoint_xyz','endpoint_pose']}
    for mode in ['endpoint_xyz','endpoint_pose']:
        aggregate[mode]['mean_of_episode_mean_endpoint_error_m']=float(np.mean([r['conditions'][mode]['mean_endpoint_error_m'] for r in results]))
    aggregate['valid_references']=sum(r.get('valid_reference',False) for r in results)
    aggregate['atomic_segments']=len(rows)
    aggregate['raw_replay_max_position_difference_m']=max(r['conditions']['raw_action_replay']['max_measured_position_difference_m'] for r in results)
    (deliver/'libero_aggregate.json').write_text(json.dumps(aggregate,indent=2))
    if a.video:
        videos(root,deliver)
    print(json.dumps(aggregate,indent=2))


def videos(root,deliver):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    for task in ['put_the_bowl_on_the_plate','put_the_cream_cheese_in_the_bowl']:
        ep=root/'libero'/(task+'__seed11')
        modes=['source','endpoint_xyz','endpoint_pose']
        titles=['Reference demonstration','Terminal XYZ only','Terminal XYZ + orientation']
        frames=[[frame for frame in imageio.get_reader(ep/(mode+'.mp4'))] for mode in modes]
        records=[json.loads((ep/(mode+'_result.json')).read_text()) for mode in modes]
        traces=[np.load(ep/(mode+'_trace.npz'))['positions'] for mode in modes]
        labels=json.loads((ep/'direction_annotations.json').read_text())
        table=['| 段 | 函数 | 控制步区间 | 终点 XYZ（m） | 净平移方向 | 运动记录 |',
               '|---:|---|---|---|---|---|']
        for s in labels['segments']:
            xyz=', '.join(f'{v:.4f}' for v in s['endpoint_position'])
            direction='无有效净方向' if s['net_direction'] is None else ', '.join(f'{v:+.3f}' for v in s['net_direction'])
            table.append(f"| {s['segment_id']} | {s['primitive']} | [{s['start']}, {s['end']}) | ({xyz}) | {direction} | {s['motion_type']} |")
        (deliver/(task+'_atoms.md')).write_text('\n'.join(table)+'\n')
        writer=imageio.get_writer(deliver/(task+'_comparison.mp4'),fps=5,codec='libx264',quality=7,macro_block_size=2)
        final_image=None
        for i in range(max(map(len,frames))):
            canvas=Image.new('RGB',(960,356),'#f3f5f8');draw=ImageDraw.Draw(canvas)
            for j,(mode,title,seq,r,trace) in enumerate(zip(modes,titles,frames,records,traces)):
                x=j*320;frame=Image.fromarray(seq[min(i,len(seq)-1)]).resize((320,256));canvas.paste(frame,(x,35))
                draw.text((x+8,8),title,fill='#111827',font=font)
                t=min(i*4,len(trace)-1);cursor=0;active=r['segment_results'][-1]
                for segment in r['segment_results']:
                    cursor+=segment['steps']
                    if t<cursor:active=segment;break
                d=trace[min(t+4,cursor,len(trace)-1)]-trace[t];norm=np.linalg.norm(d)
                direction='no translation' if norm<=.002 else 'd=('+','.join(f'{v:+.2f}' for v in d/norm)+')'
                draw.text((x+8,296),active['primitive']+'  |  '+direction,fill='#111827',font=font)
                outcome='SUCCESS' if r['final_native_success'] else 'FAILURE'
                draw.text((x+8,323),f't={min(i/5,(len(trace)-1)/20):.1f}s | final: '+outcome,
                          fill='#166534' if r['final_native_success'] else '#991b1b',font=font)
            writer.append_data(np.asarray(canvas));final_image=canvas
        writer.close();final_image.save(deliver/(task+'_comparison_final.png'))
        fig=plt.figure(figsize=(12,4.2))
        for j,(trace,title) in enumerate(zip(traces,titles)):
            ax=fig.add_subplot(1,3,j+1,projection='3d');ax.plot(*trace.T,linewidth=1.7)
            ax.scatter(*trace[0],c='green',s=35);ax.scatter(*trace[-1],c='red',s=35)
            ax.set_title(title);ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.set_zlabel('World Z (m)')
            ax.set_xlim(-.25,.22);ax.set_ylim(-.12,.12);ax.set_zlim(.85,1.25)
        fig.tight_layout();fig.savefig(deliver/(task+'_paths.png'),dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--video',action='store_true')
    main(p.parse_args())
