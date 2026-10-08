"""Lock all d=1 validation candidates, starts, source bytes and shard order before running."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
from local_policy import allowed

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    allowed()
    name='validation_d1_v1';folder=ROOT/'protocol'/name
    states=[12,10,14,11,13]
    assert not folder.exists(),'Validation definition is immutable once frozen'
    source_names=['run_closed_loop.py','controller_runtime.py','window_state.py','binary_service.py','gate_variants.py','supervise.py']
    locked_source={n:sha(ROOT/'scripts'/n) for n in source_names}
    candidates={
        'smol70':dict(method='smol70',role='native Smol70 reference'),
        'naive_k5':dict(method='naive_k5',role='frequency-matched reference for VLASH mechanism'),
        'vlash_style_k5':dict(method='vlash_style_k5',role='same frozen model VLASH-style mechanism, without offset training'),
        'fixed45':dict(method='fixed_interval',interval=45,role='selected fixed-period attribution control'),
        'compact_skip1':dict(method='laya_skip1',variant='binary_compact',interval_ratio=.3,role='input-window ablation'),
        'window3_skip1':dict(method='laya_skip1',variant='binary_window3',interval_ratio=.3,role='primary LAYA candidate'),
        'window3_first':dict(method='laya_first',variant='binary_window3',interval_ratio=.3,horizons=[50,100],role='secondary limited first-request policy'),
        'compact_minr30_i5':dict(method='laya',variant='binary_compact',interval=5,minimum_age_ratio=.3,horizons=[100],role='secondary H100 quality/cost endpoint'),
    }
    evidence=['analysis/single_skip_development_pooled.json','analysis/fixed_interval_matrix.json',
              'analysis/pooled_development.json','analysis/first_intervention_development.json',
              'analysis/first_intervention_state2_diagnostic.json','STATISTICS_AMENDMENT_01.md',
              'checks/delay_d1_qualification_v1_exact_reproduction.json']
    qualification=json.loads((ROOT/evidence[-1]).read_text(encoding='utf-8'))
    assert qualification['errors']==0 and qualification['episodes']==9
    configs=[]
    for index,sid in enumerate(states):
        run=f'validation_d1_s{sid}_v1';filename=f'{run}.json';path=ROOT/'configs'/filename
        assert not path.exists(),'Never overwrite a frozen cohort shard'
        cfg=dict(tasks=list(range(10)),states=[sid],horizons=[50,100,200],logical_delay_steps=1,
            schedule_seed=202610087,cohort_id=name,cohort_role='Paired validation independent of this round candidate selection; earlier project exposure is not excluded',
            candidates=candidates,locked_source_sha256=locked_source)
        planned=sum(1 for t in cfg['tasks'] for h in cfg['horizons'] for c in candidates.values()
                    if h in c.get('horizons',cfg['horizons']))
        assert planned==210
        configs.append((path,cfg,dict(run=run,config=filename,state=sid,episodes=planned,order=index)))
    folder.mkdir(parents=True)
    shards=[]
    for path,cfg,item in configs:
        path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8',newline='\n')
        shards.append(dict(**item,config_sha256=sha(path)))
    definition=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),cohort_id=name,
        tasks=list(range(10)),states_in_execution_order=states,horizons=[50,100,200],
        episodes_per_method_horizon=50,total_episodes=1050,candidates=candidates,shards=shards,
        primary_candidate='window3_skip1',primary_comparators=['smol70','fixed45'],
        locked_execution_source_sha256=locked_source,selection_evidence_sha256={n:sha(ROOT/n) for n in evidence},
        vla_sha256='b296dfca9e977fbe06d0f4a7971dbcfa368954e168c32f9c61295107f00eba82',
        minimum_probability=0,binary_argmax=True,readout_adaptation=False,logical_delay_steps=1,
        statistics='Primary descriptive hierarchical paired bootstrap; retain fixed-task and task-only sensitivities. No equivalence or uncorrected significance claim.',
        qualification_note='Nine real d=1 episodes exactly reproduced prior trajectories. Afterwards only the preflight source-byte lock was added to the runner.',
        no_adaptation_during_cohort=True,no_early_stopping_for_favorable_results=True,
        interpretation='Main policies span all three H. Two secondary policies have explicitly restricted H. Official pi0.5 remains a separate completed model reference.')
    (folder/'definition.json').write_text(json.dumps(definition,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(cohort=name,episodes=1050,shards=shards,definition_sha256=sha(folder/'definition.json'))))


if __name__=='__main__':main()
