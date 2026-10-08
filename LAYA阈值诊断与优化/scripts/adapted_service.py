"""Original binary LAYA plus an explicitly separate, frozen CPU readout; no VLA here."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import binary_service as binary
from deployment_readout import Readout
from score_counterfactuals import parameter_digest

ROOT=binary.ROOT
bundle=None;adapters={};captures=[];hidden=[];agent_ref=None;before_hash=None;torch_ref=None
capture_enabled=False;capture_features=False
base_evaluate=binary.evaluate;base_atomic=binary.legacy.atomic_json


def initialize(agent,torch):
    global agent_ref,before_hash,torch_ref
    if agent_ref is not None:
        assert agent_ref is agent
        return
    agent_ref=agent;torch_ref=torch
    before_hash=parameter_digest(agent.model,torch)
    assert before_hash==bundle['laya_fingerprint']
    assert agent.dtype_for(1)==torch.bfloat16
    assert all(not p.requires_grad and p.grad is None for p in agent.model.parameters())
    original=agent._forward
    def observe(batch):
        logits,actions=original(batch)
        if capture_enabled:captures.append(logits.copy())
        return logits,actions
    agent._forward=observe
    def hook(module,inputs,output):
        if capture_features:hidden.append(output.detach().float().cpu().numpy().copy())
    agent.model.scorer[2].register_forward_hook(hook)


def evaluate(agent,torch,build_sequence,request,received_ns):
    global capture_enabled,capture_features
    initialize(agent,torch)
    model_id=request.get('readout_model_id')
    if model_id is None:
        response=base_evaluate(agent,torch,build_sequence,request,received_ns)
        response.update(readout_model_id=None,model_bundle_sha256=bundle['bundle_sha256'])
        return response
    assert model_id in bundle['entries']
    entry=bundle['entries'][model_id];adapter=adapters[model_id]
    assert request['variant']==entry['variant'] and entry['target']=='utility'
    captures.clear();hidden.clear();capture_enabled=True
    capture_features=adapter.kind=='frozen_feature_linear'
    try:response=base_evaluate(agent,torch,build_sequence,request,received_ns)
    finally:capture_enabled=False;capture_features=False
    assert len(captures)==1 and captures[0].shape[0]==1
    logits=captures[0][0][:2];delta=float(logits[0]-logits[1]);feature=None
    if adapter.kind=='frozen_feature_linear':
        assert len(hidden)==1 and hidden[0].shape==(1,2,1024)
        feature=(hidden[0][0][0]-hidden[0][0][1]).astype(np.float32)
    decision=adapter.predict(delta,feature)
    response.update(raw_probabilities=response['probabilities'],probabilities=decision['probabilities'],choice=decision['choice'],
        readout_model_id=model_id,model_bundle_sha256=bundle['bundle_sha256'],raw_logit_delta=delta,
        deployment_logit=decision['logit'],feature_delta=feature.tolist() if feature is not None else None,
        decision_rule='Two-choice argmax of a separately fitted frozen readout; original LAYA output retained in raw',
        fitted_readout=True,laya_state_dict_sha256=before_hash)
    end=time.monotonic_ns();response['clock']['response_prewrite_monotonic_ns']=end
    response['clock']['response_prewrite_utc']=binary.legacy.utc_now()
    response['service_prewrite_wall_ms']=(end-received_ns)/1e6
    return response


def atomic(path,value):
    if path.name=='ready.json':
        value.update(readout_models=list(adapters),model_bundle_sha256=bundle['bundle_sha256'],
            separate_frozen_readout=True,adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if path.name=='stopped.json':
        assert agent_ref is not None
        after=parameter_digest(agent_ref.model,torch_ref)
        assert before_hash==after
        assert all(not p.requires_grad and p.grad is None for p in agent_ref.model.parameters())
        value.update(frozen_laya_before=before_hash,frozen_laya_after=after,frozen_laya_unchanged=True,
                     no_optimizer=True,model_bundle_sha256=bundle['bundle_sha256'])
    return base_atomic(path,value)


def main():
    global bundle,adapters
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--bundle',required=True)
    args=p.parse_args();path=Path(args.bundle).resolve()
    assert path.is_relative_to(ROOT/'runs')
    value=path.read_bytes();bundle=json.loads(value);bundle['bundle_sha256']=hashlib.sha256(value).hexdigest()
    assert bundle['state2_used_for_fitting'] is False
    adapters={name:Readout(entry['model']) for name,entry in bundle['entries'].items()}
    assert adapters
    binary.legacy.evaluate=evaluate;binary.legacy.atomic_json=atomic
    return binary.legacy.run(Path(args.folder).resolve())


if __name__=='__main__':main()
