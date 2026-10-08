"""Raw trajectory/accounting invariants; this is not a test of research efficacy."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
import numpy as np
from calibration_math import predict as calibrated_predict
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'长动作块补充实验'
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
folder=ROOT/'raw'/args.run/'data'
manifest=json.loads((folder/'manifest.json').read_text())
delay=manifest['logical_delay_steps']
rows=[json.loads(l) for l in (folder/'episodes.jsonl').read_text().splitlines()]
checks=Counter();failures=[];cases={};paired_initials={};paired_first_chunks={}
def check(value,kind,context):
    checks[kind]+=1
    if not value:failures.append(dict(kind=kind,context=context))
def sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
check(manifest['freeze_before']==manifest['freeze_after'],'frozen VLA',args.run)
check(manifest['freeze_after']['state_dict_sha256']=='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82','same checkpoint',args.run)
check(len(rows)==manifest['planned'],'completed schedule',args.run)
schedule=json.loads((folder/'schedule.json').read_text(encoding='utf-8'))
actual=[dict(task=r['task_id'],state=r['state_id'],candidate=r['candidate_id'],horizon=r['horizon']) for r in rows]
check(actual==schedule,'complete immutable execution order',args.run)
check(len({(r['candidate_id'],r['task_id'],r['state_id'],r['horizon']) for r in rows})==len(rows),
      'unique candidate initial horizon cells',args.run)
bundle_path=folder/'laya_ipc/model_bundle.json';bundle=None
if bundle_path.exists():
    bundle=json.loads(bundle_path.read_text())
    check(hashlib.sha256(bundle_path.read_bytes()).hexdigest()==manifest['laya_ready']['model_bundle_sha256'],'frozen readout bundle',args.run)
    stopped=manifest['readout_service_shutdown']
    check(stopped['frozen_laya_before']==stopped['frozen_laya_after']==bundle['laya_fingerprint'] and stopped['frozen_laya_unchanged'],
          'frozen LAYA before/after adapted deployment',args.run)
    check(bundle['state2_used_for_fitting'] is False,'state2 excluded from readout fitting',args.run)
for r in rows:
    cid=r['candidate_id'];h=r['horizon'];tid=r['task_id'];sid=r['state_id']
    case=folder/cid/f't{tid:02d}_s{sid:02d}_h{h}_{r["method"]}'
    tag=f'{cid}/t{tid}s{sid}h{h}'
    check(r['logical_delay_steps']==delay,'declared logical delay',tag)
    alignment_mode=r['candidate'].get('alignment_mode','current_state')
    noise_clock=r['candidate'].get('noise_clock','prediction')
    check(r['alignment_mode']==alignment_mode and r['noise_clock']==noise_clock,'declared alignment and noise clock',tag)
    trace=np.load(case/'trace.npz');initial=np.load(case/'initial_state.npy')
    calls=json.loads((case/'calls.json').read_text());gate=json.loads((case/'gate.json').read_text())
    timeline=json.loads((case/'timeline.json').read_text())
    check(sha(initial)==r['initial_state_sha256'],'initial fingerprint',tag)
    check(sha(trace['actions'])==r['executed_action_sha256'],'executed fingerprint',tag)
    check(len(trace['actions'])==len(trace['physics'])==len(timeline)==r['steps'],'physical/action length',tag)
    check(len(calls)==r['vla_calls'] and len(gate)==r['gate_calls'],'call totals',tag)
    prediction_seconds=sum(c['predict_seconds'] for c in calls)
    gate_seconds=sum(g['full_gate_seconds'] for g in gate)
    check(np.isfinite(prediction_seconds) and prediction_seconds>0 and
          abs(prediction_seconds-r['prediction_seconds'])<1e-9,'VLA time equals per-call receipts',tag)
    check(np.isfinite(gate_seconds) and gate_seconds>=0 and
          abs(gate_seconds-r['gate_seconds'])<1e-9,'gate time equals full per-decision receipts',tag)
    check(r['episode_wall_seconds']>=prediction_seconds+gate_seconds,
          'serial service fits measured episode wall',tag)
    check(r['native_generated_rows']==len(calls)*h and
          r['unused_generated_rows']==len(calls)*h-r['steps'],'generated used unused accounting',tag)
    if r['method']=='laya_first':check(len(gate)==int(r['steps']>round(.3*h)),'first-only gate count',tag)
    check(bool(timeline[-1]['success'])==r['success'],'terminal success',tag)
    check(all(np.isfinite(trace[n]).all() for n in ['actions','physics']),'finite trajectory',tag)
    chunks={c['call_id']:np.load(case/f'chunk_{c["call_id"]:03d}.npz')['actions'] for c in calls}
    initial_key=(tid,sid);chunk_key=(tid,sid,h)
    if initial_key in paired_initials:
        check(np.array_equal(initial,paired_initials[initial_key]),'within-stage paired initial exact',tag)
    else:paired_initials[initial_key]=initial.copy()
    if chunk_key in paired_first_chunks:
        check(np.array_equal(chunks[0],paired_first_chunks[chunk_key]),'within-stage first chunk exact',tag)
    else:paired_first_chunks[chunk_key]=chunks[0].copy()
    check(sum(len(c['executed_rows']) for c in calls)==r['steps'],'row accounting',tag)
    for c in calls:
        check(chunks[c['call_id']].shape==(h,7) and np.isfinite(chunks[c['call_id']]).all(),'native finite chunk',tag)
        seed_tick=c['request_tick']-delay if r['method']=='vlash_style_k5' and c['call_id'] and noise_clock=='image' else c['request_tick']
        check(c['noise_seed']==20261007+tid*100000+sid*1000+seed_tick and c['noise_seed_tick']==seed_tick,'declared noise mapping',tag)
        check(c['nfe']==10 and c['unused_rows']==h-len(c['executed_rows']),'NFE and unused rows',tag)
        if c['call_id'] and c.get('actual_delivery_tick') is not None:
            if r['method']=='vlash_style_k5':
                state_tick=c['request_tick'] if alignment_mode=='current_state' else c['request_tick']-delay
                check(c['actual_delivery_tick']==c['request_tick'] and c['state_tick']==state_tick and c['image_tick']==c['request_tick']-delay,
                      'VLASH declared image/proprio clocks',tag)
                check(c['current_state_alignment']==(alignment_mode=='current_state') and c['executed_rows'][0]==0,'VLASH declared proprio mode starts row zero',tag)
            else:
                check(c['actual_delivery_tick']==c['request_tick']+delay and c['expired_prefix_rows']==delay,'declared-delay delivery',tag)
        if c['trigger']=='mandatory_finite_queue':
            check(timeline[c['request_tick']]['queue_remaining']==delay-1,'finite queue guard',tag)
        if c['trigger']=='laya_replan':
            check(any(g['tick']==c['request_tick'] and g['choice']=='replan' for g in gate),'gate to VLA request',tag)
        if c['call_id'] and r['method']=='smol70':
            check(c['request_tick']-calls[c['call_id']-1]['request_tick']==round(.3*h),'Smol70 exact request period',tag)
        if c['call_id'] and r['method']=='fixed_r60':
            check(c['request_tick']-calls[c['call_id']-1]['request_tick']==2*round(.3*h),'fixed 0.6H request period',tag)
        if c['call_id'] and r['method']=='fixed_interval':
            check(c['request_tick']-calls[c['call_id']-1]['request_tick']==r['gate_interval'],'fixed absolute request interval',tag)
        if c['call_id'] and r['method']=='laya_skip1':
            prior=calls[c['call_id']-1]['request_tick'];interval=round(.3*h)
            check(c['request_tick']-prior in {interval,2*interval},'single skip request window',tag)
            if c['request_tick']-prior==2*interval:
                check(any(g['tick']==prior+interval and g['choice']=='continue' for g in gate),
                      'single skip recovery follows continue',tag)
        if c['call_id'] and r['method']=='laya_first':
            expected=round(.3*h)*(2 if c['call_id']==1 and gate[0]['choice']=='continue' else 1)
            check(c['request_tick']-calls[c['call_id']-1]['request_tick']==expected,'first-only intervention then exact Smol70 cadence',tag)
    for tick,row in enumerate(timeline):
        check(row['tick']==tick and np.array_equal(trace['actions'][tick],chunks[row['active_call']][row['action_row']]),'executed source row',tag)
    for g in gate:
        probs=g['probabilities'];tick=g['tick'];call=calls[timeline[tick]['active_call']]
        check(set(probs)=={'replan','continue'} and g['minimum_probability']==0,'binary argmax contract',tag)
        model_id=g.get('readout_model_id')
        check(model_id==r['candidate'].get('readout_model_id'),'configured readout identity',tag)
        if model_id is None:
            check(g['choice']==g['raw_choice'],'unadapted choice preserved',tag)
        else:
            check(bundle is not None and model_id in bundle['entries'],'saved readout coefficients',tag)
            entry=bundle['entries'][model_id]
            features=np.asarray([g['feature_delta']]) if g['feature_delta'] is not None else None
            expected=float(calibrated_predict(entry['model'],np.array([g['raw_logit_delta']]),features,np.array([h]))[0])
            check(abs(expected-probs['replan'])<1e-10,'independent deployment readout calculation',tag)
            check(g['raw_probabilities']==g['raw']['answers']['replanning']['probabilities'],'original LAYA probabilities retained',tag)
            check(g['choice']==('replan' if expected>=.5 else 'continue'),'adapted two-choice argmax',tag)
        check(probs[g['choice']]>=max(probs.values())-.0002,'higher probability selected',tag)
        if r['method'] in ['laya_skip1','laya_first']:
            check(g['plan_age']==round(.3*h),'single skip decision exactly at first request opportunity',tag)
        else:check(tick%r['gate_interval']==0 and g['plan_age']>=r['minimum_age'],'configured check eligibility',tag)
        check(g['remaining']==timeline[tick]['queue_remaining']+1 and g['plan_age']==tick-call['request_tick'],'actual queue and age',tag)
        check(hashlib.sha256(g['model_state'].encode()).hexdigest()==g['state_sha256'],'model input fingerprint',tag)
        check(g['token_audit']['head_dropped']==0 and not g['token_audit']['state']['truncated'],'complete input',tag)
        check(g['variant']==r['candidate']['variant'],'specified input variant',tag)
        check(f'after {delay} additional logical step' in g['state'],'declared delay in gate input',tag)
        check(g['full_gate_seconds']>=g['ipc_outer_seconds']>=g['sdk_synchronized_wall_ms']/1000>0,
              'gate cost includes IPC and synchronized LAYA',tag)
    reference=OLD/f'raw/attempt_v1/smol/main/t{tid:02d}_s{sid:02d}_h{h}_smol70'
    if reference.exists():
        check(np.array_equal(initial,np.load(reference/'initial_state.npy')),'old paired initial exact',tag)
        check(np.array_equal(chunks[0],np.load(reference/'chunk_000.npz')['actions']),'old first chunk exact',tag)
    cases[(cid,tid,sid,h)]=(r,trace)
for (cid,tid,sid,h),(r,trace) in cases.items():
    if cid=='full5' and ('queue_only',tid,sid,h) in cases:
        q,qt=cases[('queue_only',tid,sid,h)]
        check(r['gate_choices']=={'continue':r['gate_calls']},'full5 action collapse',f't{tid}s{sid}h{h}')
        check(np.array_equal(trace['actions'],qt['actions']) and np.array_equal(trace['physics'],qt['physics']),
              'full5 equals queue-only trajectory',f't{tid}s{sid}h{h}')
check(manifest['total_vla_calls']==manifest['qualification_vla_calls']+sum(r['vla_calls'] for r in rows),'total VLA accounting',args.run)
check(manifest['total_gate_calls']==manifest['warmup_gate_calls']+sum(r['gate_calls'] for r in rows),'total LAYA accounting',args.run)
result=dict(run=args.run,episodes=len(rows),checks=sum(checks.values()),by_kind=dict(checks),errors=len(failures),failures=failures,
            scope='raw mechanism/accounting invariants, not policy quality',
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'checks'/f'{args.run}_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(run=args.run,episodes=len(rows),checks=sum(checks.values()),errors=len(failures),failures=failures[:5])))
assert not failures
