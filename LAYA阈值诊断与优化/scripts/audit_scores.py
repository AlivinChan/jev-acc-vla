"""Audit saved logits, option features, model inputs, labels and complete split coverage."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from calibration_data import ROOT,load_runs
from gate_variants import variant_input


def main():
    p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args()
    folder=ROOT/'raw'/args.run/'data'
    manifest=json.loads((folder/'manifest.json').read_text())
    rows,features,provenance=load_runs([args.run],set(manifest['states']),manifest['variants'])
    weight=np.load(folder/'original_readout.npz',allow_pickle=False)['weight'].ravel().astype(float)
    sources={}
    for name in manifest['source_runs']:
        path=ROOT/'raw'/name/'data/records.jsonl'
        for line in path.read_text().splitlines():
            record=json.loads(line)
            if record['state'] in manifest['states']:
                assert record['id'] not in sources
                sources[record['id']]=record
    checks=0;errors=[];reconstruction=[];probability_errors=[]
    def check(condition,label):
        nonlocal checks
        checks+=1
        if not condition:errors.append(label)
    check(np.isfinite(features).all(),'finite features')
    check(features.shape==(len(rows),1024),'feature matrix shape')
    check(manifest['no_optimizer'] is True,'no optimizer')
    check(provenance['laya_fingerprint']=='ae9a05c819a0ab99493067336fd9f5c186dce48265e75ef027978f2597dfac71','pinned frozen LAYA')
    for index,row in enumerate(rows):
        tag=row['id']+'|'+row['variant'];record=sources[row['id']]
        text=record['state_text']
        if row['variant'] in ['binary_window','binary_window3']:
            from window_state import from_counterfactual
            text=from_counterfactual(record,ROOT.parent/'长动作块补充实验/raw/attempt_v1/smol/main',compact=row['variant']=='binary_window3')
        state,questions,spec=variant_input(row['variant'],text)
        check(state==row['model_state'],tag+' exact input transformation')
        check(row['labels']==record['labels'],tag+' labels')
        check(all(row[k]==record[k] for k in ['task','state','horizon','tick','split']),tag+' identity')
        check(row['token_audit']['head_dropped']==0 and not row['token_audit']['state']['truncated'],tag+' token preservation')
        names=list(questions['replanning']['criteria']);logits=np.array([row['raw_logits'][n] for n in names])
        expected=np.exp(logits/row['temperature']-np.max(logits/row['temperature']));expected/=expected.sum()
        actual=row['raw']['answers']['replanning']['probabilities']
        error=max(abs(expected[i]-actual[n]) for i,n in enumerate(names));probability_errors.append(error)
        check(error<.00006,tag+' SDK softmax')
        delta=float(logits[0]-logits[1]);binary=float(1/(1+np.exp(-delta/row['temperature'])))
        check(delta==row['raw_logit_delta'],tag+' logit subtraction')
        check(abs(binary-row['binary_probabilities']['replan'])<1e-12,tag+' binary probability')
        check(abs(sum(row['binary_probabilities'].values())-1)<1e-12,tag+' binary normalization')
        check(row['projected_action_choice']==('replan' if binary>=.5 else 'continue'),tag+' argmax')
        # The live final linear layer is BF16. Saved activations/readout are float32 copies;
        # independently redoing their dot product need not reproduce the rounded BF16 logits.
        error=abs(float(features[index]@weight)-delta);reconstruction.append(error)
        bound=.004*(abs(logits[0])+abs(logits[1]))+1e-5
        check(error<=bound,tag+' readout consistency within BF16 output rounding bound')
        if row['variant']!='numbers3':
            check(set(actual)=={'continue','replan'},tag+' true binary choices')
            check(row['raw']['answers']['replanning']['choice']==row['projected_action_choice'],tag+' SDK choice')
    result=dict(run=args.run,cases=provenance['unique_cases'],score_rows=len(rows),checks=checks,errors=len(errors),failures=errors,
                maximum_probability_rounding_error=max(probability_errors),maximum_feature_logit_reconstruction_error=max(reconstruction),
                feature_logit_reconstruction_note='FP64 reconstruction versus rounded BF16 live outputs; all checked against per-row BF16 relative rounding bound',
                frozen_laya_sha256=provenance['laya_fingerprint'])
    (ROOT/'checks'/f'{args.run}_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result));assert not errors


if __name__=='__main__':main()
