"""Same measured IPC path for explicitly labeled readout adaptations and raw controls."""
from pathlib import Path
import hashlib,json,subprocess,time
import controller_runtime as runtime
BASE=runtime.BASE;ROOT=runtime.ROOT;atomic=runtime.atomic
SCRIPT_ROOT=Path(__file__).resolve().parent


class ReadoutClient(runtime.LayaClient):
    def __init__(self,folder,bundle_path):
        self.folder=folder;folder.mkdir();(folder/'requests').mkdir();(folder/'responses').mkdir()
        value=Path(bundle_path).read_bytes();self.bundle=json.loads(value)
        self.bundle_sha=hashlib.sha256(value).hexdigest();(folder/'model_bundle.json').write_bytes(value)
        self.log=(folder/'service.log').open('w')
        self.process=subprocess.Popen([str(BASE/'envs/laya/bin/python'),str(SCRIPT_ROOT/'adapted_service.py'),
            '--folder',str(folder),'--bundle',str(folder/'model_bundle.json')],stdout=self.log,stderr=subprocess.STDOUT)
        self.index=0;self.variant='binary_compact';self.readout_model_id=None;self.closed=False;started=time.monotonic()
        try:
            while not (folder/'ready.json').exists():
                if self.process.poll() is not None:raise RuntimeError('Adapted LAYA failed to load')
                if time.monotonic()-started>180:raise TimeoutError('Adapted LAYA loading timeout')
                time.sleep(.02)
            self.ready=json.loads((folder/'ready.json').read_text())
            assert self.ready['threshold']==0 and self.ready['frozen'] and self.ready['separate_frozen_readout']
            assert self.ready['model_bundle_sha256']==self.bundle_sha
            qualification=[]
            for fixture in self.bundle['qualification_fixtures']:
                self.variant=fixture['variant'];self.readout_model_id=fixture['readout_model_id']
                response=self.predict(fixture['state'],warmup=True)
                assert response['raw_logit_delta']==fixture['expected_raw_logit_delta']
                difference=abs(response['probabilities']['replan']-fixture['expected_probability_replan'])
                assert difference<1e-10 and response['choice']==fixture['expected_choice']
                qualification.append(dict(id=fixture['id'],model=self.readout_model_id,probability_error=difference,
                                          raw_logit_delta_exact=True,choice_exact=True))
            atomic(folder/'readout_qualification.json',qualification)
        except BaseException:
            self.close();raise

    def predict(self,state,warmup=False):
        index=f'{self.index:07d}';self.index+=1;started=time.perf_counter()
        atomic(self.folder/'requests'/f'{index}.json',dict(request_id=index,state=state,warmup=warmup,
            variant=self.variant,readout_model_id=self.readout_model_id))
        path=self.folder/'responses'/f'{index}.json'
        while not path.exists():
            if self.process.poll() is not None:raise RuntimeError('Adapted LAYA service died')
            if time.perf_counter()-started>60:raise TimeoutError('Adapted LAYA request timed out')
            time.sleep(.001)
        response=json.loads(path.read_text());assert response['status']=='ok',response
        assert response['variant']==self.variant and response['readout_model_id']==self.readout_model_id
        assert response['model_bundle_sha256']==self.bundle_sha
        probabilities=response['probabilities'];assert set(probabilities)=={'replan','continue'}
        assert response['choice'] in probabilities and probabilities[response['choice']]>=max(probabilities.values())-1e-12
        if self.readout_model_id is None:assert response['choice']==response['raw_choice']
        response['ipc_outer_seconds']=time.perf_counter()-started
        return response

    def close(self):
        if self.closed:return
        super().close();self.closed=True
