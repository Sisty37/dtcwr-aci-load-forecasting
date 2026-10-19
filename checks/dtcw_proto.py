import numpy as np, math

class DTCWR:
    """Reference prototype of the v5 pseudocode (h=1 indexing: score s_t available at time t)."""
    def __init__(s, W_S, W_L, k, H, c, mu_z, sigma0, sd_z=1.0, zlo=-np.inf, eps=1e-6):
        s.W_S,s.W_L,s.k,s.H,s.c,s.mu,s.s0,s.eps=W_S,W_L,k,H,c,mu_z,sigma0,eps
        s.sd,s.zlo=sd_z,zlo; s.post_raw=[]; s.paused=False
        s.Gp=s.Gm=0.0; s.zp=s.zm=None   # last-zero timestamps for G+ and G-
        s.anchor=-10**18; s.t_alarm=-10**18; s.log=[]
    def update(s, t, score):
        raw=math.log(score/s.s0+s.eps)
        if s.paused:                         # re-baselining after an alarm
            s.post_raw.append(raw)
            if len(s.post_raw)>=s.W_S:
                s.mu=float(np.mean(s.post_raw)); s.paused=False
                s.Gp=s.Gm=0.0; s.zp=s.zm=t
            return
        z=max((raw-s.mu)/s.sd, s.zlo)
        s.Gp=max(0.0,s.Gp+z-s.k); s.Gm=max(0.0,s.Gm-z-s.k)
        if s.Gp==0.0: s.zp=t
        if s.Gm==0.0: s.zm=t
        if s.zp is None: s.zp=t
        if s.zm is None: s.zm=t
        fire_p, fire_m = s.Gp>s.H, s.Gm>s.H
        if fire_p or fire_m:
            if t-s.t_alarm>=s.c:
                # triggering statistic's last-zero time; if both fire take the larger excess
                trig = 'p' if (fire_p and (not fire_m or s.Gp>=s.Gm)) else 'm'
                tau_hat = (s.zp if trig=='p' else s.zm)+1   # first sample after last zero
                s.anchor=tau_hat; s.t_alarm=t; s.log.append(('alarm',t,trig,tau_hat))
                # re-baseline reference on post-change data (scores tau_hat..t are re-used)
                s.post_raw=[]; s.paused=True
            else:
                s.log.append(('cooldown_ignored',t))
            s.Gp=s.Gm=0.0; s.zp=s.zm=t
    def window_idx(s, t, avail_first):
        """indices of scores in window at time t (scores avail_first..t)."""
        n_avail=t-avail_first+1
        n_post=max(0, t-max(s.anchor,avail_first)+1) if s.anchor<=t else 0
        n=min(s.W_L, max(s.W_S, n_post), n_avail)
        return np.arange(t-n+1, t+1), n_post

def sim(changes, T=6000, seed=0):
    r=np.random.default_rng(seed); scale=np.ones(T)
    for (tau,f) in changes: scale[tau:]=f
    return np.abs(r.normal(0,1,T))*scale

def run(scores, **kw):
    # calibrate mu_z, sigma0 on first 1000 (validation-like)
    val=scores[:1000]; s0=np.median(val)/0.6745
    lz=np.log(val/s0+1e-6); mu=lz.mean(); sd=lz.std()
    zs=(lz-mu)/sd; zlo=np.quantile(zs,0.05)
    m=DTCWR(mu_z=mu, sigma0=s0, sd_z=sd, zlo=zlo, **kw); rec=[]
    for t in range(1000,len(scores)):
        m.update(t,scores[t]); idx,npost=m.window_idx(t,0); rec.append((t,idx,npost,m.anchor))
    return m,rec

kw=dict(W_S=48,W_L=720,k=0.5,H=6,c=24)
# Test 1: single variance shift at 2000 -> anchor near 2000; membership rules
m,rec=run(sim([(2000,3.0)]),**kw)
alarms=[a for a in m.log if a[0]=='alarm']; first=alarms[0]
assert 2000<=first[1]<=2100, first
assert abs(first[3]-2000)<=15, ("anchor error",first)
a=first[3]
for t,idx,npost,anc in rec:
    if anc!=a or t<first[1]: continue
    pre=int(np.sum(idx<a))
    if npost < kw['W_S']: assert pre==kw['W_S']-npost, (t,pre,npost)
    else: assert pre==0, (t,pre,npost)
    if t>first[1]+800: break
print("T1 single shift: alarm at",first[1],"anchor",a,"(true 2000); membership rules OK; #alarms total",len(alarms))

# Test 2: alarm inside cooldown must not move anchor
m2=DTCWR(W_S=48,W_L=720,k=0.5,H=6,c=1000,mu_z=0.0,sigma0=1.0)
for t in range(0,300): m2.update(t,1.0)            # z≈0: quiet
for t in range(300,320): m2.update(t,np.e**3)      # big burst -> alarm
anc=m2.anchor; ta=m2.t_alarm
for t in range(320,420): m2.update(t,1.0)          # quiet; re-baseline completes
for t in range(420,460): m2.update(t,np.e**4)      # second burst, still inside cooldown (c=1000)
assert m2.anchor==anc and m2.t_alarm==ta and any(x[0]=='cooldown_ignored' for x in m2.log)
print("T2 cooldown: anchor unchanged at",anc,"after in-cooldown alarm(s): OK")

# Test 3: repeated alarms -> old post-change residuals drop out once new post-change count >= W_S
m3,rec3=run(sim([(2000,3.0),(3500,0.5)]),**kw)
al=[x for x in m3.log if x[0]=='alarm']
second=[x for x in al if x[1]>3400][0]; a2=second[3]
assert abs(a2-3500)<=40, second
for t,idx,npost,anc in rec3:
    if anc!=a2: continue
    if npost>=kw['W_S']: assert np.all(idx>=a2), t
print("T3 repeated alarm: 2nd alarm",second[1],"anchor",a2,"(true 3500); no pre-anchor scores once n_post>=W_S: OK")

# Test 4: separate last-zero bookkeeping (downward shift triggers G-)
m4,_=run(sim([(2500,0.3)]),**kw)
d=[x for x in m4.log if x[0]=='alarm' and x[1]>2400][0]
assert d[2]=='m' and abs(d[3]-2500)<=30, d
print("T4 downward shift: triggered by G-, anchor",d[3],"(true 2500): OK")

# False alarm rate on stationary stream
m5,_=run(sim([] ,T=21000,seed=3),**kw)
print("False alarms per 1000 steps on stationary stream:", round(len([x for x in m5.log if x[0]=='alarm'])/20,3))
