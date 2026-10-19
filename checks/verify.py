import numpy as np, math
rng=np.random.default_rng(0)

# ---------- 1. Quantile convention ----------
def Qhat(window, p):
    if p <= 0: return -np.inf
    if p > 1: return np.inf
    n=len(window); k=math.ceil(p*(n+1))
    if k>n: return np.inf
    return np.sort(window)[k-1]

# monotonicity check
w=rng.exponential(size=50)
ps=np.linspace(-0.2,1.2,2001); qs=[Qhat(w,p) for p in ps]
qa=np.array(qs); assert all(qa[i]<=qa[i+1] for i in range(len(qa)-1)), "non-monotone"
print("Qhat non-decreasing on [-0.2,1.2]: OK; Qhat(0)=",Qhat(w,0),"Qhat(1)=",Qhat(w,1))

# ---------- 2. Interleaved delayed ACI bound ----------
def run(T,h,gamma,alpha,a1,adversarial,seed):
    r=np.random.default_rng(seed)
    # scores: heavy tail + abrupt shifts + autocorrelation
    z=np.zeros(T+h); e=r.standard_t(2,size=T+h)
    for t in range(1,T+h): z[t]=0.9*z[t-1]+e[t]
    s=np.abs(z)*np.where(np.arange(T+h)>T//2, 5,1)
    A=np.full(h,a1); errs=np.zeros(T); amin=np.inf; amax=-np.inf
    hist=[]  # scores available
    for t in range(T):            # origin t, target t+h
        j=t%h
        # outcomes revealed at time t: targets <= t -> origins <= t-h
        if t-h>=0:
            # update copy (t-h)%h == j with err of origin t-h  (its previous use)
            A[j]=A[j]+gamma*(alpha-errs[t-h])
        amin=min(amin,A[j]); amax=max(amax,A[j])
        avail=s[:t+1] if t>=0 else s[:0]
        # arbitrary data-dependent window (random length, possibly adversarially tiny)
        L=int(r.integers(1,200)) if adversarial else 200
        win=avail[-L:] if len(avail)>0 else np.array([])
        if len(win)==0: q=np.inf if 1-A[j]>0 else -np.inf
        else: q=Qhat(win,1-A[j])
        errs[t]= float(s[t+h]>q)
    C=max(a1,1-a1)+gamma
    gap=abs(errs.mean()-alpha); bound=h*C/(gamma*T)
    return gap,bound,amin,amax
worst=0
for h in [1,24]:
  for gamma in [0.005,0.02,0.1]:
    for adv in [False,True]:
      for seed in range(20):
        T=int(rng.integers(3000,6000))
        gap,b,amin,amax=run(T,h,gamma,0.1,0.1,adv,seed)
        assert gap<=b+1e-12, (h,gamma,adv,seed,gap,b)
        assert amin>=-gamma-1e-12 and amax<=1+gamma+1e-12
        worst=max(worst,gap/b)
print("Interleaved delayed-ACI bound hC/(gamma T) held in all 240 runs; max gap/bound =",round(worst,3))

# ---------- 3. KL Monte Carlo cross-check ----------
from scipy import stats
q=stats.norm.ppf(0.95); a=0.1
def klb(p): return p*np.log(p/a)+(1-p)*np.log((1-p)/(1-a))
N=2_000_000
for k in [1.5,2,3]:
    x=np.abs(rng.normal(0,k,N))
    # log-likelihood ratio of folded N(0,k^2) vs folded N(0,1)
    llr=-np.log(k)-x**2/(2*k*k)+x**2/2
    p=np.mean(x>q)
    print(f"var x{k}: MC KL magnitude={llr.mean():.3f}  KL indicator={klb(p):.3f}  ratio={llr.mean()/klb(p):.2f}")
for d in [1,2,3]:
    x=np.abs(rng.normal(d,1,N))
    f1=stats.norm.pdf(x-d)+stats.norm.pdf(x+d); f0=2*stats.norm.pdf(x)
    llr=np.log(f1/f0); p=np.mean(x>q)
    print(f"mean +{d}: MC KL magnitude={llr.mean():.3f}  KL indicator={klb(p):.3f}  ratio={llr.mean()/klb(p):.2f}")
