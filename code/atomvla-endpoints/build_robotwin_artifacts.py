"""Visualize saved RoboTwin rollouts; never executes the environment."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main(root):
    out = root / 'artifacts'
    out.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    modes = ['source', 'endpoint_xyz', 'endpoint_pose']
    titles = ['Reference demonstration', 'Terminal XYZ only', 'Terminal XYZ + orientation']
    for ep in sorted((root / 'robotwin').glob('*')):
        if not (ep / 'direction_annotations.json').exists():
            continue
        labels = json.loads((ep / 'direction_annotations.json').read_text())
        program = json.loads((ep / 'program_xyz.json').read_text())
        active_arm = program[0][0]['arm']
        arm_i = ['left', 'right'].index(active_arm)
        frames, traces, results, segs = [], [], [], []
        for mode in modes:
            reader = imageio.get_reader(ep / mode / 'video.mp4')
            frames.append([frame for frame in reader])
            reader.close()
            traces.append(np.load(ep / mode / 'trace.npz')['positions'])
            results.append(json.loads((ep / mode / 'result.json').read_text()))
            segs.append(json.loads((ep / mode / 'segments.json').read_text()))
        rows = ['| 段 | 执行臂 | 原生函数 | 物理步区间 | 实测终点 XYZ（m） | 净平移方向 | 段内最大转动（°） |',
                '|---:|---|---|---|---|---|---:|']
        for s in labels['segments']:
            if s['arm'] != active_arm:
                continue
            xyz = ', '.join(f'{v:.5f}' for v in s['endpoint_position'])
            d = '无有效净方向' if s['net_direction'] is None else ', '.join(f'{v:+.3f}' for v in s['net_direction'])
            rows.append(f"| {s['segment_id']} | {active_arm} | {s['primitive_by_arm'][active_arm]} | "
                        f"[{s['start']}, {s['end']}) | ({xyz}) | {d} | {s['rotation_excursion_deg']:.2f} |")
        (out / (ep.name + '_atoms.md')).write_text('\n'.join(rows) + '\n')
        length = max(map(len, frames))
        contact_indices = set(np.linspace(0, length - 1, 4).astype(int))
        contacts = []
        with imageio.get_writer(out / (ep.name + '_comparison.mp4'), fps=10,
                                codec='libx264', quality=7, macro_block_size=2) as writer:
            for i in range(length):
                canvas = Image.new('RGB', (1440, 476), '#f3f5f8')
                draw = ImageDraw.Draw(canvas)
                for j, (title, seq, tr, res, segments) in enumerate(zip(titles, frames, traces, results, segs)):
                    frame_i = min(i, len(seq) - 1)
                    t = min(frame_i * 25, len(tr) - 1)
                    segment = next((s for s in segments if s['start'] <= t < s['end']), segments[-1])
                    delta = tr[min(t + 25, segment['end']), arm_i] - tr[t, arm_i]
                    norm = np.linalg.norm(delta)
                    d = 'no translation' if norm <= .002 else 'd=(' + ','.join(f'{x:+.2f}' for x in delta/norm) + ')'
                    x = 480 * j
                    draw.text((x + 10, 10), title, fill='#111827', font=font)
                    canvas.paste(Image.fromarray(seq[frame_i]).resize((480, 360)), (x, 38))
                    kind = segment['primitive_by_arm'][active_arm]
                    draw.text((x + 10, 404), f'{active_arm}/{kind} | {d}', fill='#111827', font=font)
                    success = res['final_native_success']
                    draw.text((x + 10, 437), f't={t/250:.2f}s | final: ' + ('SUCCESS' if success else 'FAILURE'),
                              fill='#166534' if success else '#991b1b', font=font)
                writer.append_data(np.asarray(canvas))
                if i in contact_indices:
                    contacts.append(canvas.copy())
        canvas.save(out / (ep.name + '_comparison_final.png'))
        sheet = Image.new('RGB', (1440, 476 * len(contacts)))
        for j, frame in enumerate(contacts):
            sheet.paste(frame, (0, j * 476))
        sheet.save(out / (ep.name + '_contact_sheet.jpg'), quality=90)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(14, 4.5))
        for j, (tr, title, segments) in enumerate(zip(traces, titles, segs)):
            ax = fig.add_subplot(1, 3, j + 1, projection='3d')
            for idx, s in enumerate(segments):
                xyz = tr[s['start']:s['end'] + 1, arm_i]
                ax.plot(*xyz.T, label=str(idx), linewidth=1.8)
                ax.text(*xyz[-1], str(idx), fontsize=8)
            ax.set_title(title)
            ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)'); ax.set_zlabel('World Z (m)')
            ax.set_xlim(-.35, .35); ax.set_ylim(-.5, .05); ax.set_zlim(.8, 1.2)
        fig.suptitle(ep.name + ' | active arm: ' + active_arm)
        fig.tight_layout()
        fig.savefig(out / (ep.name + '_paths.png'), dpi=150)
        plt.close(fig)
        print(ep.name, 'frames', list(map(len, frames)), 'saved')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    main(Path(p.parse_args().root))
