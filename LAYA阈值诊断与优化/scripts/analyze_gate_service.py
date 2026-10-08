"""Account for the development gate overhead and an explicitly idealized break-even model."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    all_gates=[];by_h={h:[] for h in [50,100,200]};sources={}
    pooled_path=ROOT/'analysis/single_skip_development_pooled.json'
    pooled=json.loads(pooled_path.read_text(encoding='utf-8'))['groups']
    for name in ['single_skip_v1','single_skip_confirmation_v1']:
        folder=ROOT/'raw'/name/'data'
        assert json.loads((folder.parent/'supervisor.json').read_text(encoding='utf-8'))['status']=='completed'
        for line in (folder/'episodes.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['candidate_id']!='window3_skip1':continue
            p=folder/'window3_skip1'/f"t{row['task_id']:02d}_s{row['state_id']:02d}_h{row['horizon']}_laya_skip1"/'gate.json'
            gates=json.loads(p.read_text(encoding='utf-8'));assert len(gates)==row['gate_calls']
            all_gates.extend(gates);by_h[row['horizon']].extend(gates)
            sources[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    timing={}
    for key in ['full_gate_seconds','ipc_outer_seconds','token_audit_wall_ms','sdk_synchronized_wall_ms','service_prewrite_wall_ms']:
        values=np.array([g[key] for g in all_gates]);scale=1000 if key.endswith('_seconds') else 1
        timing[key]=dict(n=len(values),mean_ms=float(values.mean()*scale),median_ms=float(np.median(values)*scale),
                         p95_ms=float(np.quantile(values,.95)*scale))
    groups={}
    for h,gates in by_h.items():
        r=pooled[f'window3_skip1_h{h}'];assert len(gates)==r['gate_calls']
        v=r['vla_seconds']/r['vla_calls'];g=r['gate_seconds']/len(gates)
        continues=sum(x['choice']=='continue' for x in gates);frequency=continues/len(gates)
        groups[str(h)]=dict(gates=len(gates),continues=continues,observed_continue_frequency=frequency,
            mean_vla_ms=v*1000,mean_full_gate_ms=g*1000,idealized_break_even_continue_frequency=g/v,
            idealized_steady_state_cost_ratio=(1+g/v)/(1+frequency),
            observed_cost_ratio=r['comparisons']['smol70']['total_service_cost_ratio'],
            actual_successes=r['successes'],total_service_seconds=r['total_service_seconds'])
    result=dict(timing=timing,by_horizon=groups,source_gate_sha256=sources,
        pooled_source_sha256=hashlib.sha256(pooled_path.read_bytes()).hexdigest(),
        break_even_assumptions='Idealized long stationary episode, each I/2I cycle has one VLA and one gate, unchanged success and stopping time. Ratio=(1+c_gate/c_vla)/(1+p_continue). Boundary/terminal effects and policy-induced trajectory changes are excluded.',
        warning='Observed choice frequency is not a calibrated probability of safe continuation; this accounting model establishes no success guarantee.')
    (ROOT/'analysis/gate_service_development.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(timing=timing,by_horizon=groups)))


if __name__=='__main__':main()
