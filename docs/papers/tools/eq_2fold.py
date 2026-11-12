#!/usr/bin/env python3
"""2-fold cross-validated re-score of GS-EQ / SP-EQ r1/r2/r3 from existing per-view rows (CPU only).

Fold A = even positions of each scene's sorted test-view names, fold B = odd positions.
For each fold F (held-out) the cap is picked on the other fold (dev) with the registered rule
(smallest cap whose 3-seed mean LPIPS-VGG <= competitor 3-seed mean, dev views), then scored on F.
Registered tally: per scene, ratio = mean over 3 paired groups of ours/competitor time at the cap;
FASTER if ratio < 0.95 and LPIPS holds on >= 2/3 seeds; SLOWER if > 1.05 or no cap reaches the
competitor's LPIPS (dev); else TIE.  Row AHEAD if FASTER >= 4 and no SLOWER; PARITY if FASTER >= SLOWER; else BEHIND.
"""
import json, os, re, sys, collections, statistics as st

CAPS = [300, 500, 800, 1200]
# Data root: SPLAT_DATA (default ~/data/splat). EQ_VERIFIED: printf pattern of the verify.py outputs to compare.
DATA = os.environ.get('SPLAT_DATA') or os.path.join(os.path.expanduser('~'), 'data', 'splat')
RES = os.path.join(DATA, 'runs/perf/results.jsonl')
GSEQ = os.path.join(DATA, 'runs/perf/fal/gseq/rows-all.jsonl')
VERIFIED = os.environ.get('EQ_VERIFIED') or os.path.join(os.environ.get('TMPDIR', '/tmp'), 'v-%s.json')


def load_speq(tag, sha, timekey):
    """returns {scene: {group: {arm: (time, {view: lpips})}}} ; arm 'comp' or cap int"""
    out = collections.defaultdict(dict)
    for l in open(RES):
        r = json.loads(l)
        lab = r['engine'].get('label') or ''
        if lab == f'spirula-v2026.10.6-{tag}':
            arm = 'comp'
        else:
            m = re.fullmatch(rf'{tag}-(\d+)k-{sha}', lab)
            if not m:
                continue
            arm = int(m.group(1))
        g = r['pair_id']
        t = r.get(timekey)
        assert t is not None, (lab, timekey)
        pv = {p['image']: p['lpips_vgg'] for p in r['per_image']}
        assert len(pv) == r['n_scored'] == r['views_expected'], lab
        d = out[r['scene']].setdefault(g, {})
        assert arm not in d, (lab, g)
        d[arm] = (t, pv)
    return out


def load_gseq():
    out = collections.defaultdict(dict)
    for l in open(GSEQ):
        r = json.loads(l)
        gs = r['gseq']
        arm = 'comp' if gs['arm'] == 'gsplat-mcmc1m' else int(re.fullmatch(r'ours-(\d+)k', gs['arm']).group(1))
        g = f"s{gs['seed']}"
        pv = {p['image']: p['lpips_vgg'] for p in r['per_image']}
        assert len(pv) == r['n_scored'] == r['views_expected']
        d = out[r['scene']].setdefault(g, {})
        assert arm not in d
        d[arm] = (gs['train_s'], pv)
    return out


def mean(xs):
    return sum(xs) / len(xs)


def analyse(name, data, comp_name, tkey):
    flags = []
    res = {'name': name, 'scenes': {}}
    for scene in sorted(data):
        groups = data[scene]
        assert len(groups) == 3, (scene, len(groups))
        gids = sorted(groups)
        for g in gids:
            assert set(groups[g]) == set(['comp'] + CAPS), (scene, g, groups[g].keys())
        # view-set check across all arms and groups
        vsets = {(g, a): frozenset(groups[g][a][1]) for g in gids for a in groups[g]}
        allv = set(vsets.values())
        if len(allv) != 1:
            flags.append(f'{name} {scene}: view sets differ across arms/groups ({sorted(len(v) for v in allv)})')
        views = sorted(set.intersection(*[set(v) for v in allv]))
        foldv = {'A': views[0::2], 'B': views[1::2]}

        def lp(g, a, vs):
            pv = groups[g][a][1]
            return mean([pv[v] for v in vs])

        def t(g, a):
            return groups[g][a][0]

        def score(cap, vs):
            """held-out scoring of a cap on views vs"""
            if cap is None:
                return dict(cap=None, verdict='SLOWER', why='no cap reaches comp LPIPS on dev')
            ours_g = [lp(g, cap, vs) for g in gids]
            comp_g = [lp(g, 'comp', vs) for g in gids]
            holds = [o <= c for o, c in zip(ours_g, comp_g)]
            ratios = [t(g, cap) / t(g, 'comp') for g in gids]
            ratio = mean(ratios)
            nh = sum(holds)
            mean_holds = mean(ours_g) <= mean(comp_g)
            if ratio < 0.95 and nh >= 2:
                v = 'FASTER'
            elif ratio > 1.05:
                v = 'SLOWER'
            else:
                v = 'TIE'
            # strict variant: held-out 3-seed mean must also hold, else "LPIPS not met" -> SLOWER
            vs_strict = v if mean_holds else 'SLOWER'
            return dict(cap=cap, ours_lp=mean(ours_g), comp_lp=mean(comp_g), margin=mean(comp_g) - mean(ours_g),
                        holds=holds, nh=nh, mean_holds=mean_holds, ratio=ratio, ratios=ratios,
                        verdict=v, strict=vs_strict)

        def pick(vs):
            c = mean([lp(g, 'comp', vs) for g in gids])
            for cap in CAPS:
                if mean([lp(g, cap, vs) for g in gids]) <= c:
                    return cap
            return None

        sres = {'n_views': len(views), 'nA': len(foldv['A']), 'nB': len(foldv['B'])}
        # in-sample (reproduction)
        sres['insample'] = score(pick(views), views)
        sres['fixed1200'] = score(1200, views)
        for held, dev in (('A', 'B'), ('B', 'A')):
            cap = pick(foldv[dev])
            sres[held] = score(cap, foldv[held])
            sres[held]['dev_cap'] = cap
            sres[held]['fixed1200'] = score(1200, foldv[held])
        res['scenes'][scene] = sres
    res['flags'] = flags
    return res


def tally(vs):
    c = collections.Counter(vs)
    f, s, ti = c['FASTER'], c['SLOWER'], c['TIE']
    row = 'AHEAD' if f >= 4 and s == 0 else ('PARITY' if f >= s else 'BEHIND')
    return f'{f}/{ti}/{s} {row}'


def combined(a, b):
    return a if a == b else 'DISAGREE'


def combined_tally(vs):
    c = collections.Counter(vs)
    f, s, ti, d = c['FASTER'], c['SLOWER'], c['TIE'], c['DISAGREE']
    # disagreeing scenes do not count as FASTER; conservatively they are not SLOWER either -> shown separately
    row = 'AHEAD' if f >= 4 and s == 0 else ('PARITY' if f >= s else 'BEHIND')
    return f'{f}/{ti}/{s} (+{d} disagree) {row}'


def fmt(x, n=3):
    return f'{x:.{n}f}'.lstrip('0') if x is not None and x < 1 else (f'{x:.{n}f}' if x is not None else '-')


def report(res, verified, md):
    name = res['name']
    md.append(f"\n## {name}\n")
    md.append(f"Time clock: {res['clock']}. Competitor: {res['comp']}. Views per scene (A/B): " +
              ', '.join(f"{s} {v['nA']}/{v['nB']}" for s, v in res['scenes'].items()) + '.\n')
    md.append('In-sample reproduction (all views) vs verified JSON: ' + '; '.join(
        f"{s} {v['insample']['cap']}k {fmt(v['insample']['ratio'])} {v['insample']['verdict']}"
        f" [verified {verified.get(s, ('?', '?', '?'))[0]} {fmt(verified.get(s, (0, 0, 0))[1]) if s in verified else '?'} {verified.get(s, ('?', '?', '?'))[2]}]"
        for s, v in res['scenes'].items()) + '\n')
    md.append('| scene | fold (held-out) | cap picked on dev | ours LPIPS held-out | comp LPIPS held-out | mean holds | seeds hold | ratio | verdict (registered) | verdict (strict) | fixed-1.2M: holds / ratio / verdict |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for s, v in res['scenes'].items():
        for f in 'AB':
            r = v[f]
            fx = r['fixed1200']
            if r['cap'] is None:
                md.append(f"| {s} | {f} | none | - | - | - | - | - | SLOWER | SLOWER | {fx['nh']}/3 / {fmt(fx['ratio'])} / {fx['verdict']} |")
                continue
            md.append(f"| {s} | {f} | {r['cap']}k | {r['ours_lp']:.4f} | {r['comp_lp']:.4f} | {'yes' if r['mean_holds'] else 'NO'} ({r['margin']:+.4f}) | "
                      f"{''.join('Y' if h else 'n' for h in r['holds'])} {r['nh']}/3 | {fmt(r['ratio'])} | {r['verdict']} | {r['strict']} | "
                      f"{fx['nh']}/3 ({'mean ok' if fx['mean_holds'] else 'MEAN FAIL'}) / {fmt(fx['ratio'])} / {fx['verdict']} |")
    ta = [v['A']['verdict'] for v in res['scenes'].values()]
    tb = [v['B']['verdict'] for v in res['scenes'].values()]
    sa = [v['A']['strict'] for v in res['scenes'].values()]
    sb = [v['B']['strict'] for v in res['scenes'].values()]
    comb = [combined(a, b) for a, b in zip(ta, tb)]
    combs = [combined(a, b) for a, b in zip(sa, sb)]
    fxa = [v['A']['fixed1200']['verdict'] for v in res['scenes'].values()]
    fxb = [v['B']['fixed1200']['verdict'] for v in res['scenes'].values()]
    fxall = [v['fixed1200']['verdict'] for v in res['scenes'].values()]
    ins = [v['insample']['verdict'] for v in res['scenes'].values()]
    md.append('')
    md.append(f"- In-sample (registered, same views): {tally(ins)}")
    md.append(f"- Fold A held-out (registered rule): {tally(ta)}; strict: {tally(sa)}")
    md.append(f"- Fold B held-out (registered rule): {tally(tb)}; strict: {tally(sb)}")
    md.append(f"- Combined (scene counts only if both folds agree): {combined_tally(comb)} -> per scene: " +
              ', '.join(f'{s} {c}' for s, c in zip(res['scenes'], comb)))
    md.append(f"- Combined strict: {combined_tally(combs)}")
    md.append(f"- Fixed 1.2M: all views {tally(fxall)}; fold A {tally(fxa)}; fold B {tally(fxb)}")
    res['summary'] = dict(ins=tally(ins), A=tally(ta), B=tally(tb), comb=combined_tally(comb), combs=combined_tally(combs),
                          sA=tally(sa), sB=tally(sb), fx=tally(fxall), fxA=tally(fxa), fxB=tally(fxb))
    if res['flags']:
        md.append('- FLAGS: ' + '; '.join(res['flags']))


def verified_of(key):
    d = json.load(open(VERIFIED % key))
    out = {}
    if key == 'gs-eq':
        for s, v in d['per_scene'].items():
            out[s] = (v['budget_k'], v['ratio'], v['verdict'])
    elif key == 'sp-eq':
        for s, v in d['per_scene'].items():
            out[s] = (v['lpips_match_budget'], v['lpips_match_ratio'], v['verdict'])
    else:
        for s, v in d['primary_train_s']['per_scene'].items():
            out[s] = (v['budget'], v['ratio'], v['verdict'])
    return out


def main(outmd):
    md = ['# EQ claims: 2-fold cross-validated re-score',
          '',
          'Script: eq_2fold.py (same directory). CPU only, existing per-view rows. Folds: per scene, test views sorted by name;'
          ' fold A = even positions, fold B = odd. For held-out fold A the cap is picked on fold B (dev) and vice versa,'
          ' with the registered rule (smallest cap with 3-seed mean LPIPS-VGG <= competitor 3-seed mean on dev views).'
          ' Held-out scoring uses the registered tally: FASTER if mean paired time ratio < 0.95 and LPIPS holds on >= 2/3 seeds'
          ' (seed i: ours seed-i LPIPS <= competitor paired run i LPIPS, on held-out views); SLOWER if ratio > 1.05 or no cap'
          ' reaches the competitor on dev; else TIE. Row AHEAD if FASTER >= 4 and no SLOWER; PARITY if FASTER >= SLOWER.'
          ' "Strict" additionally requires the held-out 3-seed mean LPIPS to hold, otherwise the scene is "LPIPS not met" = SLOWER'
          ' (the registered rule only checks the mean at selection time, so this is the held-out analogue).'
          ' Time ratios are per run (not per view) and do not change with the fold; only the picked cap can change.'
          ' Tallies are written FASTER/TIE/SLOWER.']
    claims = [
        ('GS-EQ (L40S, vs gsplat-MCMC 1M default, feb21b9)', load_gseq(), 'gsplat-MCMC 1M', 'train_s', 'gs-eq', GSEQ),
        ('SP-EQ r1 (097c3fd, vs Spirula default)', load_speq('speq', '097c3fd', 'wall_s'), 'Spirula v2026.10.6', 'wall_s (registered r1 clock; r1 ours rows have no train_s)', 'sp-eq', RES),
        ('SP-EQ r2 (8122037, vs Spirula default)', load_speq('speq2', '8122037', 'train_s'), 'Spirula v2026.10.6', 'train_s (00:17 amendment)', 'sp-eq-r2', RES),
        ('SP-EQ r3 (b6dcf37, vs Spirula default)', load_speq('speq3', 'b6dcf37', 'train_s'), 'Spirula v2026.10.6', 'train_s', 'sp-eq-r3', RES),
    ]
    allres = []
    for name, data, comp, clock, vkey, src in claims:
        res = analyse(name, data, comp, clock)
        res['comp'], res['clock'] = comp, clock + f'; source {src}'
        report(res, verified_of(vkey), md)
        allres.append(res)
    md.append('\n## Summary\n')
    md.append('| claim | in-sample (registered) | fold A | fold B | combined (both folds agree) | combined strict | fixed 1.2M (all views) |')
    md.append('|---|---|---|---|---|---|---|')
    for r in allres:
        s = r['summary']
        md.append(f"| {r['name']} | {s['ins']} | {s['A']} | {s['B']} | {s['comb']} | {s['combs']} | {s['fx']} |")
    open(outmd, 'w').write('\n'.join(md) + '\n')
    json.dump([{k: v for k, v in r.items()} for r in allres], open(outmd.replace('.md', '.json'), 'w'), indent=1, default=str)
    print('\n'.join(md))


if __name__ == '__main__':
    main(sys.argv[1])
