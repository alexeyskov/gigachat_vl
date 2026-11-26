#!/usr/bin/env python3
"""Print machine-readable progress and manifest paths without credentials."""
from pathlib import Path
import argparse
import json
import shutil


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='datasets/research/pilot-v1')
    parser.add_argument('--log', default='runs/research/logs/data-pilot.log')
    args=parser.parse_args()
    root=Path(args.output).resolve()
    result={'output':str(root), 'complete':(root/'audit.json').exists(),
            'disk_free_gib':round(shutil.disk_usage(root if root.exists() else root.parent).free/1024**3,2)}
    if result['complete']:
        result['audit']=json.loads((root/'audit.json').read_text())
        result['manifests']={p.stem:{'path':str(p),'bytes':p.stat().st_size} for p in sorted(root.glob('*.jsonl'))}
    log=Path(args.log)
    if log.exists():
        events=[]
        for line in log.read_text(errors='replace').splitlines():
            try:
                event=json.loads(line)
                if isinstance(event,dict) and 'event' in event: events.append(event)
            except ValueError: pass
        result['latest_events']=events[-3:]
        result['log_path']=str(log.resolve())
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
