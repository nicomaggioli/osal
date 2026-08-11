#!/usr/bin/env python3
"""Fast causal probe of the spot-basis averaging mechanism on 192 markets."""
from __future__ import annotations
import json
import continuous_spot_basis_dev as c


def compact(r):
    return {k:v for k,v in r.items() if k != 'records'}


def main():
    spot=c.load_spot(['2026-07-26','2026-07-27','2026-07-28'])
    meta_by=c.load_metadata(); rows_by=c.load_dense(meta_by)
    metas=sorted((m for m in meta_by.values() if m.ticker in rows_by),key=lambda m:m.open_s)
    if len(metas)!=192: raise RuntimeError(f'expected 192 markets, got {len(metas)}')
    results=[]; cache={}
    for window,mult in [(60,.75),(120,.60),(120,.75),(120,.90),(240,.75)]:
        opps={m.ticker:c.build_opportunities(m,rows_by[m.ticker],spot,window,mult) for m in metas}
        cache[(window,mult)]=opps
        for latency in [3,5]:  # one-second completed-bar observation + at least two seconds routing
            for participation in [.05,.10,.25]:
                for pmin in [.75,.80,.85,.90]:
                    for edge_min in [.02,.05]:
                        for max_ask in [.90,.95]:
                            p=(latency,participation,45,600,pmin,edge_min,max_ask)
                            r=c.evaluate(p,metas,rows_by,opps)
                            r['model']={'sigma_window':window,'sigma_multiplier':mult}
                            results.append(r)
    ranked=sorted(results,key=lambda r:(r['fixed_3h_lcb95'],r['after_top10_mean'],r['total_pnl']),reverse=True)
    robust=[r for r in ranked if r['trades']>=30 and min(r['chronological_thirds'] or [-1])>0
            and min(r['day_means'].values() or [-1])>0 and r['after_top10_mean']>0
            and r['after_strongest_3h_block_mean']>0 and r['top1_positive_share']<=.15
            and r['top5_positive_share']<=.50]
    chosen=robust[0] if robust else ranked[0]
    model=chosen['model']; p=chosen['params']
    tup=(p['latency_s'],p['participation'],p['min_seconds_to_close'],p['max_seconds_to_close'],p['p_min'],p['edge_min'],p['max_ask'])
    best=c.evaluate(tup,metas,rows_by,cache[(model['sigma_window'],model['sigma_multiplier'])]); best['model']=model
    out={'round':'continuous spot-basis causal probe','opened_days':sorted(c.TARGET_DAYS),'unopened_day':'2026-07-29',
         'market_count':len(metas),'candidate_count':len(results),'robust_count':len(robust),'best':best,
         'top50_robust':[compact(r) for r in robust[:50]],'top50_all':[compact(r) for r in ranked[:50]]}
    (c.OUT/'continuous_spot_basis_probe.json').write_text(json.dumps(out,indent=2,sort_keys=True))
    print(json.dumps(compact(best),indent=2,sort_keys=True))

if __name__=='__main__': main()
