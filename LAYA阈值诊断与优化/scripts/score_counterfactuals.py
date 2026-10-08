"""Frozen LAYA logits and option features; outcomes never enter the model input."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,signal,time,traceback
from gate_variants import variant_input
from supervise import allowed
BASE=Path('/mnt/4t/jev_vla_libero')
ROOT=BASE/'experiments/laya_gate_optimization'
VARIANTS=['numbers3','binary_short','binary_compact']


def digest(value):return hashlib.sha256(value).hexdigest()


def parameter_digest(model,torch):
    result=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        result.update(name.encode());result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--runs',required=True)
    p.add_argument('--states',default='0,1,2');p.add_argument('--variants',default=','.join(VARIANTS));args=p.parse_args()
    variants=args.variants.split(',');assert len(set(variants))==len(variants) and set(variants)<=set(VARIANTS+['binary_window','binary_window3'])
    out=Path(args.out).resolve();assert out.is_relative_to(ROOT/'runs');out.mkdir(exist_ok=False)
    states={int(s) for s in args.states.split(',')};records={};sources={}
    schedule=json.loads((ROOT/'fixtures/counterfactual_schedule.json').read_text())
    expected={r['id']:r for r in schedule if r['state'] in states}
    for name in args.runs.split(','):
        assert name.isascii() and name.replace('_','').isalnum()
        folder=ROOT/'runs'/name
        assert json.loads((folder/'supervisor.json').read_text())['status']=='completed'
        manifest=json.loads((folder/'data/manifest.json').read_text())
        assert manifest['frozen_parameters_unchanged']
        for line in (folder/'data/records.jsonl').read_text().splitlines():
            record=json.loads(line)
            if record['state'] not in states:continue
            assert record['id'] not in records,'Duplicate counterfactual ID'
            assert record['branches']['replan']['baseline_suffix_exact']
            assert digest(record['state_text'].encode())==record['state_sha256']
            assert all(record[k]==v for k,v in expected[record['id']].items())
            records[record['id']]=record;sources[record['id']]=name
    assert set(records)==set(expected),'Incomplete predeclared state split'
    import numpy as np
    import torch
    import laya
    from laya.common import build_sequence,_encode_question_text,QTYPES,temp_bucket
    torch.set_num_threads(4);allowed()
    source=json.loads((BASE/'configs/local_gate_sources.json').read_text())['laya']
    assert laya.__version__==source['sdk_version']
    agent=laya.Agent(str(BASE/source['path']),device='cuda',compile=False,fast=False,
        expected_sha256={i['path']:i['sha256'] for i in source['assets']})
    agent.model.eval()
    for parameter in agent.model.parameters():parameter.requires_grad_(False)
    assert agent.device.type=='cuda'
    before=parameter_digest(agent.model,torch)
    raw_capture=[];feature_capture=[]
    forward=agent._forward
    def observe(batch):
        logits,actions=forward(batch);raw_capture.append(logits.copy());return logits,actions
    agent._forward=observe
    def feature_hook(module,inputs,output):feature_capture.append(output.detach().float().cpu().numpy().copy())
    hook=agent.model.scorer[2].register_forward_hook(feature_hook)
    weight=agent.model.scorer[-1].weight.detach().float().cpu().numpy().copy()
    np.savez_compressed(out/'original_readout.npz',weight=weight)
    manifest=dict(started_utc=datetime.now(timezone.utc).isoformat(),states=sorted(states),cases=len(records),
        variants=variants,source_runs=sorted(set(sources.values())),source=source,source_record_runs=sources,
        source_schedule_sha256=digest((ROOT/'fixtures/counterfactual_schedule.json').read_bytes()),
        scripts={p.name:digest(p.read_bytes()) for p in Path(__file__).parent.glob('*.py')},
        frozen_laya_before=before,no_optimizer=True,
        feature_definition='Replan option minus continue option activation after the existing scorer GELU, before its last linear layer',
        timing_scope='Offline synchronized scoring including feature capture; not deployment latency',
        third_option_note='numbers3 retains the old question schema with corrected numbers; binary probabilities are an explicitly labeled projection for diagnosis only')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    features=[];row_ids=[];rows=[]
    def evaluate(record,variant):
        text=record['state_text']
        if variant in ['binary_window','binary_window3']:
            from window_state import from_counterfactual
            text=from_counterfactual(record,BASE/'experiments/long_chunk_laya/attempt_v1/smol/main',compact=variant=='binary_window3')
        state,questions,spec=variant_input(variant,text)
        q=questions['replanning'];labels=list(q['criteria'])
        assert labels[:2]==['replan','continue']
        internal=dict(t=q['type'],ins=q['instructions'],crit=q['criteria'])
        ids,markers,options,stats=build_sequence(agent.tok,state,internal,max_len=spec['max_len'],head_max_len=spec['head'],
            return_stats=True,return_truncation_stats=True)
        fullhead=len(_encode_question_text(agent.tok,'choice question: '+q['instructions'],add_special_tokens=False))
        assert fullhead==markers[0]-2 and not stats['truncated'] and options['options_distinct']==len(labels)
        raw_capture.clear();feature_capture.clear();fallback=getattr(agent,'cpu_fallback_count',0)
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.no_grad():
            raw=agent.system_one(state=state,questions=questions,max_len=spec['max_len'],head_max_len=spec['head'])
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        assert agent.device.type=='cuda' and getattr(agent,'cpu_fallback_count',0)==fallback
        assert len(raw_capture)==len(feature_capture)==1
        logits=np.asarray(raw_capture[0][0][:len(labels)],dtype=np.float64)
        hidden=feature_capture[0][0]
        assert hidden.shape==(len(labels),weight.shape[1]) and np.isfinite(hidden).all()
        temp=float(agent.temperature_by_options.get(temp_bucket(QTYPES['choice'],len(labels)),agent.temperature[QTYPES['choice']]))
        probabilities=np.exp(logits/temp-np.max(logits/temp));probabilities/=probabilities.sum()
        returned=raw['answers']['replanning']['probabilities']
        assert max(abs(float(probabilities[i])-returned[label]) for i,label in enumerate(labels))<.00006
        delta=float(logits[0]-logits[1]);p_replan=float(1/(1+np.exp(-delta/temp)))
        item=dict(id=record['id'],variant=variant,task=record['task'],state=record['state'],horizon=record['horizon'],
            tick=record['tick'],split=record['split'],labels=record['labels'],
            outcomes={c:dict(success=r['success'],vla_calls=r['vla_calls']) for c,r in record['branches'].items()},
            raw_logits=dict(zip(labels,logits.tolist())),raw=raw,temperature=temp,raw_logit_delta=delta,
            binary_probabilities={'replan':p_replan,'continue':1-p_replan},
            projected_action_choice='replan' if p_replan>=.5 else 'continue',
            model_state=state,model_state_sha256=digest(state.encode()),
            token_audit=dict(tokens=len(ids),head_dropped=0,state=stats),sdk_and_capture_seconds=elapsed)
        return item,(hidden[0]-hidden[1]).astype(np.float32)
    try:
        first=records[sorted(records)[0]]
        with (out/'warmup.jsonl').open('w') as stream:
            for variant in variants:
                item,_=evaluate(first,variant);stream.write(json.dumps(item)+'\n')
        with (out/'scores.jsonl').open('w') as stream:
            for i,cid in enumerate(sorted(records)):
                allowed()
                for variant in variants:
                    item,feature=evaluate(records[cid],variant);item['feature_row']=len(features)
                    features.append(feature);row_ids.append(cid+'|'+variant);rows.append(item)
                    stream.write(json.dumps(item)+'\n');stream.flush()
                if i%20==0:print(json.dumps(dict(completed_cases=i+1,planned_cases=len(records))),flush=True)
        np.savez_compressed(out/'features.npz',features=np.asarray(features),row_ids=np.asarray(row_ids))
        after=parameter_digest(agent.model,torch);assert before==after
        manifest.update(completed_utc=datetime.now(timezone.utc).isoformat(),rows=len(rows),
                        frozen_laya_after=after,frozen_parameters_unchanged=True,
                        feature_dimensions=int(np.asarray(features).shape[1]))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        (out/'status.json').write_text(json.dumps(dict(phase='completed',completed=len(records),planned=len(records),rows=len(rows))))
    except BaseException as error:
        np.savez_compressed(out/'partial_features.npz',features=np.asarray(features),row_ids=np.asarray(row_ids))
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error),traceback=traceback.format_exc(),completed_rows=len(rows)),indent=2))
        raise
    finally:hook.remove()


def interrupted(number,frame):raise SystemExit(f'Owned worker interrupted by signal {number}')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,interrupted)
    main()
