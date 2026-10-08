"""Paired hierarchical and fixed-task bootstrap for balanced episode cohorts."""
import numpy as np
from analyze_closed_loop import key,cost


def uncertainty(a,b,replicates=10000):
    amap={key(r):r for r in a};bmap={key(r):r for r in b}
    assert len(amap)==len(a) and len(bmap)==len(b)
    assert set(amap)==set(bmap),'Require a complete paired cohort, not an intersection'
    tasks=sorted({k[0] for k in amap});sizes=[];blocks=[]
    for t in tasks:
        keys=sorted(k for k in amap if k[0]==t);sizes.append(len(keys));block=[]
        for k in keys:
            aa,bb=amap[k],bmap[k]
            assert aa['initial_state_sha256']==bb['initial_state_sha256']
            block.append([cost(aa),cost(bb),int(aa['success'])-int(bb['success']),1])
        blocks.append(block)
    assert len(set(sizes))==1 and sizes[0]>0,'Balanced declared starts per task required'
    values=np.asarray(blocks,dtype=np.float64);nt,ns=values.shape[:2]
    assert np.isfinite(values).all() and (values[:,:,1]>0).all()
    rng=np.random.default_rng(20261008)
    task_draws=rng.integers(0,nt,(replicates,nt))
    state_draws=rng.integers(0,ns,(replicates,nt,ns))
    samples={
        'hierarchical_task_and_initial':values[task_draws[:,:,None],state_draws].sum(axis=(1,2)),
        'fixed_tasks_resample_initials':values[np.arange(nt)[None,:,None],state_draws].sum(axis=(1,2)),
    }
    results={}
    for name,sample in samples.items():
        ratios=sample[:,0]/sample[:,1];deltas=sample[:,2]/sample[:,3]
        results[name]=dict(cost_ratio_95=np.quantile(ratios,[.025,.975]).tolist(),
                          success_delta_95=np.quantile(deltas,[.025,.975]).tolist())
    return dict(n=len(amap),tasks=nt,initials_per_task=ns,replicates=replicates,seed=20261008,
        intervals=results,interpretation='Descriptive paired bootstrap; does not establish equivalence or non-inferiority')
