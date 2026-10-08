"""Frozen LAYA replay; observe exact logits without an extra forward pass."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import time
from gate_variants import VARIANTS,variant_input

BASE=Path('/mnt/4t/jev_vla_libero')
ROOT=BASE/'experiments/laya_gate_optimization'


def sha(value):return hashlib.sha256(value).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    out=Path(args.out).resolve()
    assert out.is_relative_to(ROOT/'runs')
    out.mkdir(exist_ok=False)
    import numpy as np
    import torch
    import laya
    from laya.common import build_sequence,_encode_question_text,QTYPES,temp_bucket
    from supervise import allowed
    allowed()
    torch.set_num_threads(4)
    source=json.loads((BASE/'configs/local_gate_sources.json').read_text())['laya']
    assert laya.__version__==source['sdk_version']
    agent=laya.Agent(str(BASE/source['path']),device='cuda',compile=False,fast=False,
        expected_sha256={i['path']:i['sha256'] for i in source['assets']})
    agent.model.eval()
    for p in agent.model.parameters():p.requires_grad_(False)
    assert agent.device.type=='cuda'
    records=json.loads((ROOT/'fixtures/diagnostic_replay.json').read_text())
    manifest=dict(utc=datetime.now(timezone.utc).isoformat(),source=source,
        fixture_sha256=sha((ROOT/'fixtures/diagnostic_replay.json').read_bytes()),
        variants=VARIANTS,device=str(agent.device),frozen=True,sdk=laya.__version__,
        temperature=list(agent.temperature),temperature_by_options=agent.temperature_by_options,
        scripts={p.name:sha(p.read_bytes()) for p in (ROOT/'scripts').glob('*.py')},
        timing_scope='One synchronized SDK call; excludes preflight token audit, includes host SDK work.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    captured=[]
    original_forward=agent._forward
    def observe(batch):
        logits,act=original_forward(batch)
        captured.append(logits.copy())
        return logits,act
    agent._forward=observe
    def evaluate(record,name,warmup=False):
        state,questions,spec=variant_input(name,record['state'])
        q=questions['replanning'];internal={'t':q['type'],'ins':q['instructions'],'crit':q['criteria']}
        sequence=build_sequence(agent.tok,state,internal,max_len=spec['max_len'],head_max_len=spec['head'],
                                return_stats=True,return_truncation_stats=True)
        ids,markers,options,state_stats=sequence
        fullhead=len(_encode_question_text(agent.tok,'choice question: '+q['instructions'],add_special_tokens=False))
        kepthead=markers[0]-2
        audit=dict(tokens=len(ids),head_full=fullhead,head_used=kepthead,head_dropped=fullhead-kepthead,
                   options=options,state=state_stats,fits_512=len(ids)<=512)
        assert not state_stats['truncated'] and options['options_distinct']==len(q['criteria'])
        captured.clear()
        before=getattr(agent,'cpu_fallback_count',0)
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.no_grad():
            raw=agent.system_one(state=state,questions=questions,max_len=spec['max_len'],head_max_len=spec['head'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        assert agent.device.type=='cuda' and getattr(agent,'cpu_fallback_count',0)==before
        assert len(captured)==1
        k=len(q['criteria']);logits=np.asarray(captured[0][0][:k],dtype=np.float64)
        temp=float(agent.temperature_by_options.get(temp_bucket(QTYPES['choice'],k),agent.temperature[QTYPES['choice']]))
        def softmax(z):
            values=np.exp(z-z.max());return values/values.sum()
        probs=softmax(logits/temp)
        returned=raw['answers']['replanning']['probabilities']
        assert max(abs(float(p)-returned[label]) for p,label in zip(probs,q['criteria']))<.00006
        result=dict(id=record['id'],kind=record['kind'],variant=name,warmup=warmup,
            state=state,state_sha256=sha(state.encode()),prompt_sha256=sha(json.dumps(questions,sort_keys=True).encode()),
            token_audit=audit,logits=dict(zip(q['criteria'],logits.tolist())),temperature=temp,
            probabilities=returned,raw_choice=raw['answers']['replanning']['choice'],
            max_probability=max(returned.values()),temperature_one_diagnostic_only=dict(zip(q['criteria'],softmax(logits).tolist())),
            sdk_wall_seconds=elapsed,raw=raw)
        if record['kind']=='developer_replay':
            result.update(case=record['case'],tick=record['tick'])
            if name=='original3':
                assert result['state_sha256']==record['old_state_sha256']
                result['old_probability_max_error']=max(abs(v-record['old_probabilities'][label]) for label,v in returned.items())
        else:result['semantic_label']=record['semantic_label']
        return result
    # Separate warmups per length/schema; never used as experiment observations.
    with (out/'warmup.jsonl').open('w') as stream:
        for name in VARIANTS:
            stream.write(json.dumps(evaluate(records[0],name,True))+'\n');stream.flush()
    with (out/'decisions.jsonl').open('w') as stream:
        for i,record in enumerate(records):
            allowed()
            names=list(VARIANTS)
            names=names[i%len(names):]+names[:i%len(names)]
            for name in names:
                result=evaluate(record,name)
                stream.write(json.dumps(result)+'\n');stream.flush()
            if i%20==0:print(json.dumps(dict(completed=i+1,total=len(records))),flush=True)
    assert not any(p.requires_grad for p in agent.model.parameters()) and not agent.model.training
    (out/'completed.json').write_text(json.dumps(dict(records=len(records),decisions=len(records)*len(VARIANTS),
        frozen=True,cpu_fallback_count=getattr(agent,'cpu_fallback_count',0),utc=datetime.now(timezone.utc).isoformat()),indent=2))
    print('completed',flush=True)


if __name__=='__main__':main()
