"""Reuse the pinned service lifecycle; every actual decision is a two-choice argmax."""
from pathlib import Path
import hashlib,json,sys,time
from gate_variants import variant_input,questions_for,VARIANTS
BASE=Path('/mnt/4t/jev_vla_libero')
ROOT=BASE/'experiments/laya_gate_optimization'
sys.path.insert(0,str(BASE/'experiments/long_chunk_laya/scripts'))
import laya_service as legacy
from laya.common import _encode_question_text

legacy.EXPERIMENT_ROOT=ROOT
legacy.MINIMUM_PROBABILITY=0.0
legacy.LABELS={'continue','replan'}
legacy.QUESTIONS=questions_for('short2')


def evaluate(agent,torch,build_sequence,request,received_ns):
    name=request['variant']
    assert name in VARIANTS and VARIANTS[name]['schema'].endswith('2')
    state,questions,spec=variant_input(name,request['state'])
    q=questions['replanning'];internal={'t':q['type'],'ins':q['instructions'],'crit':q['criteria']}
    started=time.monotonic_ns()
    ids,markers,options,truncation=build_sequence(agent.tok,state,internal,max_len=spec['max_len'],
        head_max_len=spec['head'],return_stats=True,return_truncation_stats=True)
    fullhead=len(_encode_question_text(agent.tok,'choice question: '+q['instructions'],add_special_tokens=False))
    head_dropped=fullhead-(markers[0]-2)
    assert head_dropped==0 and not truncation['truncated'] and options['options_distinct']==2
    audit_end=time.monotonic_ns()
    before=getattr(agent,'cpu_fallback_count',0)
    assert agent.device.type=='cuda'
    torch.cuda.synchronize();prediction_start=time.monotonic_ns()
    with torch.no_grad():
        raw=agent.system_one(state=state,questions=questions,max_len=spec['max_len'],head_max_len=spec['head'])
    torch.cuda.synchronize();prediction_end=time.monotonic_ns()
    assert agent.device.type=='cuda' and getattr(agent,'cpu_fallback_count',0)==before
    choice,raw_choice,probabilities=legacy.checked_answer(raw)
    assert choice==raw_choice and choice in {'continue','replan'}
    end=time.monotonic_ns()
    return dict(status='ok',request_id=request['request_id'],warmup=bool(request.get('warmup',False)),
        variant=name,choice=choice,raw_choice=raw_choice,probabilities=probabilities,minimum_probability=0.0,
        decision_rule='two-choice argmax; fixed option order; exact ties retain SDK first-option choice',
        usage=raw.get('usage',{}),raw=raw,model_state=state,
        token_audit=dict(tokens=len(ids),options=options,state=truncation,head_dropped=head_dropped,
                         max_len=spec['max_len'],head_max_len=spec['head']),
        state_sha256=legacy.digest_bytes(state.encode()),
        prompt_sha256=legacy.digest_bytes(legacy.canonical_bytes(questions)),
        clock=dict(received_monotonic_ns=received_ns,prediction_start_monotonic_ns=prediction_start,
                   prediction_end_monotonic_ns=prediction_end,response_prewrite_monotonic_ns=end,
                   response_prewrite_utc=legacy.utc_now()),
        token_audit_wall_ms=(audit_end-started)/1e6,sdk_synchronized_wall_ms=(prediction_end-prediction_start)/1e6,
        service_prewrite_wall_ms=(end-received_ns)/1e6,cpu_fallback_count=before)


base_atomic=legacy.atomic_json
def atomic(path,value):
    if path.name=='ready.json':
        value.update(binary_argmax=True,variants=VARIANTS,
            wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            variants_source_sha256=hashlib.sha256(Path(__file__).with_name('gate_variants.py').read_bytes()).hexdigest())
    return base_atomic(path,value)

legacy.evaluate=evaluate
legacy.atomic_json=atomic
if __name__=='__main__':legacy.main()
