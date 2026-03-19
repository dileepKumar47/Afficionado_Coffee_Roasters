# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG & STYLES ─────────────────────────────────────────────────────
st.set_page_config(page_title="Afficionado Coffee",page_icon="☕",layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#FAF5EC;color:#1C1008;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1C1008,#3D1F0A);border-right:2px solid #C8722A;}
[data-testid="stSidebar"] *{color:#F5ECD7 !important;}
h1,h2,h3{font-family:'Playfair Display',serif;color:#C8722A;}
h2{border-bottom:2px solid #C8722A;padding-bottom:4px;}
.kpi{background:linear-gradient(135deg,#3D1F0A,#1C1008);border-radius:12px;padding:16px 18px;border-left:4px solid #C8722A;margin-bottom:8px;}
.kpi .val{font-size:1.65rem;font-weight:700;font-family:'Playfair Display',serif;color:#C8722A;}
.kpi .lbl{font-size:0.71rem;text-transform:uppercase;letter-spacing:0.07em;color:#E8D5B0;margin-top:3px;}
.kpi .dlt{font-size:0.8rem;margin-top:4px;} .pos{color:#7ECB7A;} .neg{color:#E88080;}
.badge{display:inline-block;background:#C8722A;color:white;border-radius:20px;padding:3px 14px;font-size:0.78rem;font-weight:600;text-transform:uppercase;margin-bottom:10px;}
.fc-box{background:white;border-radius:10px;padding:10px 14px;margin-bottom:8px;box-shadow:0 1px 6px rgba(0,0,0,0.07);border-top:3px solid #C8722A;}
.fc-lbl{font-size:0.7rem;text-transform:uppercase;color:#8B7355;letter-spacing:0.06em;}
.fc-val{font-size:1.1rem;font-weight:700;color:#3D1F0A;font-family:'Playfair Display',serif;}
</style>""",unsafe_allow_html=True)

# ─── CHART THEME & COLOR PALETTE ──────────────────────────────────────────────
SC={"Lower Manhattan":"#C8722A","Hell's Kitchen":"#5B8DB8","Astoria":"#5C8C5A","All Stores":"#8B7355"}
LY=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="#FFF8F0",font=dict(family="DM Sans",color="#1C1008"),
        margin=dict(l=20,r=20,t=40,b=20),hoverlabel=dict(bgcolor="#3D1F0A",font_color="#F5ECD7",font_size=13),
        legend=dict(bgcolor="#FFF8F0",bordercolor="#C8722A",borderwidth=1,font=dict(color="#000000")))
def F(t=""):
    f=go.Figure(); f.update_layout(title=dict(text=t,font=dict(size=15,family="Playfair Display",color="#C8722A")),**LY); return f

# ─── DATA LOADING & FEATURE ENGINEERING ───────────────────────────────────────
@st.cache_data(show_spinner="Brewing insights... ☕")
def load(path):
    df=pd.read_csv(path).sort_values("transaction_id").reset_index(drop=True)
    df["revenue"]=df["transaction_qty"]*df["unit_price"]
    df["date"]=pd.to_datetime("2025-01-01")+pd.to_timedelta((df.index.to_numpy()*181//len(df)).clip(0,180),unit="D")
    df["hour"]=df["transaction_time"].str.split(":").str[0].astype(int)
    df["day_of_week"]=df["date"].dt.dayofweek; df["day_name"]=df["date"].dt.day_name()
    df["month"]=df["date"].dt.month; df["month_name"]=df["date"].dt.strftime("%b")
    return df

# ─── DAILY AGGREGATION WITH 7-DAY ROLLING MEAN ────────────────────────────────
@st.cache_data
def build_daily(df):
    d=(df.groupby(["date","store_location"]).agg(revenue=("revenue","sum"),transactions=("transaction_id","count"),qty=("transaction_qty","sum"))
       .reset_index().sort_values(["store_location","date"]).reset_index(drop=True))
    parts=[]
    for s in d["store_location"].unique():
        g=d[d["store_location"]==s].copy().sort_values("date").reset_index(drop=True)
        g["roll7"]=g["revenue"].rolling(7).mean(); parts.append(g)
    return pd.concat(parts,ignore_index=True)

# ─── FORECASTING MODELS ───────────────────────────────────────────────────────
def naive(s,h):    return np.full(h,s.iloc[-1])
def ma(s,h,w=7):   return np.full(h,s.rolling(w,min_periods=1).mean().iloc[-1])
def ema(s,h,a=.3): return np.full(h,s.ewm(alpha=a,adjust=False).mean().iloc[-1])
def lin(s,h):
    from sklearn.linear_model import LinearRegression
    x=np.arange(len(s)).reshape(-1,1); y=s.values; m=LinearRegression().fit(x,y)
    p=m.predict(np.arange(len(s),len(s)+h).reshape(-1,1)); std=(y-m.predict(x)).std()
    return p,p-1.96*std,p+1.96*std
def gbr(s,h):
    from sklearn.ensemble import GradientBoostingRegressor
    L=7; sv=s.values
    if len(sv)<L+5: p=np.full(h,sv.mean()); return p,p-sv.std(),p+sv.std()
    X=[sv[i-L:i] for i in range(L,len(sv))]; y=sv[L:]; X,y=np.array(X),np.array(y)
    m=GradientBoostingRegressor(n_estimators=120,learning_rate=.08,max_depth=3,random_state=42).fit(X,y)
    hist=list(sv[-L:]); ps=[]
    for _ in range(h): p=m.predict(np.array(hist[-L:]).reshape(1,-1))[0]; ps.append(p); hist.append(p)
    p=np.array(ps); std=(y-m.predict(X)).std()
    return p,p-1.96*std,p+1.96*std
def sarima(s,h):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        if len(s)<30: return lin(s,h)
        fit=SARIMAX(s.values,order=(1,1,1),seasonal_order=(1,1,1,7),enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        fc=fit.get_forecast(h); ci=fc.conf_int(); return fc.predicted_mean,ci[:,0],ci[:,1]
    except: return lin(s,h)
# ─── MODEL EVALUATION METRICS (MAE, RMSE, MAPE) ───────────────────────────────
def evm(a,p):
    a,p=np.array(a,float),np.array(p,float)
    return {"MAE":round(np.mean(np.abs(a-p)),2),"RMSE":round(np.sqrt(np.mean((a-p)**2)),2),"MAPE(%)":round(np.mean(np.abs((a-p)/(a+1e-9)))*100,2)}

# ─── SIDEBAR CONTROLS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ Afficionado\nCoffee Roasters\n---")
    PATH=st.text_input("Dataset path","Afficionado_Coffee_Roasters.csv")
    ST=st.selectbox("🏪 Store",["All Stores","Lower Manhattan","Hell's Kitchen","Astoria"])
    HD=st.slider("📅 Forecast days",1,30,7)
    MC=st.radio("🤖 Model",["Gradient Boosting","SARIMA","Exp Smoothing","Moving Average","Naive","Linear Trend","Compare All"])
    MT=st.radio("📊 Metric",["Revenue ($)","Transactions"])
    SCI=st.checkbox("Confidence intervals",True); SSC=st.checkbox("Scenario analysis",False)
    st.caption("Afficionado Coffee · v1.0")

# ─── LOAD & FILTER DATA ───────────────────────────────────────────────────────
try: raw=load(PATH); dly=build_daily(raw)
except FileNotFoundError: st.error(f"❌ File not found: `{PATH}`"); st.stop()

def fi(df): return df if ST=="All Stores" else df[df["store_location"]==ST]
fr=fi(raw)
agg=(fi(dly).groupby("date").agg(revenue=("revenue","sum"),transactions=("transactions","sum"))
     .reset_index().sort_values("date").reset_index(drop=True))
agg["roll7"]=agg["revenue"].rolling(7).mean()

# ─── PAGE HEADER & KPI CARDS ──────────────────────────────────────────────────
st.markdown("<h1>☕ Afficionado Coffee Roasters</h1><p style='color:#8B7355;margin-top:-8px;'>Forecasting & Peak Demand Dashboard</p>",unsafe_allow_html=True)
st.markdown(f'<div class="badge">📍 {ST} · {HD}-Day Forecast</div>',unsafe_allow_html=True)

tr=agg["revenue"].sum(); tt=agg["transactions"].sum(); ar=agg["revenue"].mean()
pr=agg["revenue"].max(); pd_=agg.loc[agg["revenue"].idxmax(),"date"].strftime("%d %b")
m1=agg[agg["date"].dt.month==1]["revenue"].sum(); m2=agg[agg["date"].dt.month==2]["revenue"].sum()
mom=((m2-m1)/(m1+1e-9))*100
def kpi(v,l,d=None,p=True):
    dk=f'<div class="dlt {"pos" if p else "neg"}">{"▲" if p else "▼"} {d}</div>' if d else ""
    return f'<div class="kpi"><div class="val">{v}</div><div class="lbl">{l}</div>{dk}</div>'
c1,c2,c3,c4,c5=st.columns(5)
with c1: st.markdown(kpi(f"${tr:,.0f}","Total Revenue"),unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{tt:,}","Transactions"),unsafe_allow_html=True)
with c3: st.markdown(kpi(f"${ar:,.0f}","Avg Daily Rev"),unsafe_allow_html=True)
with c4: st.markdown(kpi(f"${pr:,.0f}",f"Peak ({pd_})"),unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{mom:+.1f}%","Jan→Feb",d=f"{abs(mom):.1f}%",p=mom>=0),unsafe_allow_html=True)
st.markdown("<br>",unsafe_allow_html=True)

# ─── DASHBOARD TABS ───────────────────────────────────────────────────────────
T=st.tabs(["📈 Sales Forecast","🔥 Peak Demand","🏪 Store Comparison","📦 Categories","🤖 Model Eval","📊 EDA"])

# ─── TAB 1: SALES FORECAST ────────────────────────────────────────────────────
with T[0]:
    st.markdown("## Sales & Revenue Forecast")
    mc="revenue" if "Revenue" in MT else "transactions"; ml="Revenue ($)" if "Revenue" in MT else "Transactions"
    s=agg[mc].astype(float).ffill(); dt=agg["date"]
    fd=pd.date_range(dt.iloc[-1]+pd.Timedelta(days=1),periods=HD)
    with st.spinner("Forecasting..."):
        lo=hi=None
        if MC=="Naive": ps=naive(s,HD)
        elif MC=="Moving Average": ps=ma(s,HD)
        elif MC=="Exp Smoothing": ps=ema(s,HD)
        elif MC=="Linear Trend": ps,lo,hi=lin(s,HD)
        elif MC=="Gradient Boosting": ps,lo,hi=gbr(s,HD)
        elif MC=="SARIMA": ps,lo,hi=sarima(s,HD)
        else: ps,lo,hi=gbr(s,HD)
        ps=np.maximum(np.array(ps,float),0)
        if lo is not None: lo,hi=np.maximum(lo,0),np.maximum(hi,0)
    f1=F(f"{ml} Forecast · {ST}")
    f1.add_trace(go.Scatter(x=dt,y=s,name="Historical",line=dict(color=SC.get(ST,"#C8722A"),width=2)))
    f1.add_trace(go.Scatter(x=dt,y=s.rolling(7).mean(),name="7-Day MA",line=dict(color="#8B7355",width=1.5,dash="dot"),opacity=0.8))
    f1.add_trace(go.Scatter(x=fd,y=ps,name=f"Forecast ({MC})",line=dict(color="#E07B39",width=2.5,dash="dash")))
    if SCI and lo is not None:
        f1.add_trace(go.Scatter(x=list(fd)+list(fd[::-1]),y=list(hi)+list(lo[::-1]),fill="toself",fillcolor="rgba(200,114,42,0.15)",line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
    f1.add_vline(x=str(dt.iloc[-1]),line_dash="dot",line_color="#8B7355")
    f1.add_annotation(x=str(dt.iloc[-1]),y=1,yref="paper",text="Forecast Start",showarrow=False,xanchor="left",font=dict(color="#8B7355",size=12))
    f1.update_layout(height=400,xaxis_title="Date",yaxis_title=ml,legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(f1,width='stretch')
    fmt=(lambda v:f"${v:,.0f}") if "Revenue" in MT else (lambda v:f"{int(v):,}")
    ct,cs=st.columns([3,2])
    with ct:
        st.markdown("#### 📋 Forecast Table")
        df2=pd.DataFrame({"Date":fd.strftime("%a, %d %b"),f"Forecast {ml}":[fmt(p) for p in ps]})
        if lo is not None: df2["Lower"]=[fmt(v) for v in lo]; df2["Upper"]=[fmt(v) for v in hi]
        st.dataframe(df2,width='stretch',hide_index=True)
    with cs:
        st.markdown("#### 📌 Forecast Summary")
        for lb,vl in [("Mean Sales",fmt(ps.mean())),("Min Sales",fmt(ps.min())),("Max Sales",fmt(ps.max())),("Total Sales",fmt(ps.sum()))]:
            st.markdown(f'<div class="fc-box"><div class="fc-lbl">{lb}</div><div class="fc-val">{vl}</div></div>',unsafe_allow_html=True)
    if SSC:
        st.markdown("### 🔮 Scenario Analysis")
        fs=F("Best / Base / Worst Case")
        for lb,yv,col in [("Best (+20%)",ps*1.2,"#5C8C5A"),("Base",ps,"#C8722A"),("Worst (-20%)",ps*.8,"#C04A3A")]:
            fs.add_trace(go.Scatter(x=fd,y=yv,name=lb,line=dict(color=col,width=2)))
        fs.update_layout(height=300); st.plotly_chart(fs,width='stretch')

# ─── TAB 2: PEAK DEMAND ───────────────────────────────────────────────────────
with T[1]:
    st.markdown("## Peak Demand Detection")
    hp=(fr.groupby(["day_of_week","hour"])["revenue"].sum().reset_index().pivot(index="day_of_week",columns="hour",values="revenue").fillna(0))
    hp.index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(hp)]
    fh=go.Figure(go.Heatmap(z=hp.values,x=[f"{h}:00" for h in hp.columns],y=hp.index,colorscale=[[0,"#FAF5EC"],[0.5,"#C8722A"],[1,"#3D1F0A"]],hovertemplate="Day:%{y} Hour:%{x}<br>$%{z:,.0f}<extra></extra>"))
    fh.update_layout(title=dict(text="Revenue Heatmap (Day × Hour)",font=dict(size=15,family="Playfair Display",color="#C8722A")),height=290,**LY,xaxis_title="Hour")
    st.plotly_chart(fh,width='stretch')
    c1,c2=st.columns(2)
    with c1:
        ah=fr.groupby("hour")["revenue"].mean().reset_index(); fb=F("Avg Revenue by Hour")
        fb.add_trace(go.Bar(x=ah["hour"],y=ah["revenue"],marker=dict(color=ah["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#3D1F0A"]],showscale=False)))
        ph=int(ah.loc[ah["revenue"].idxmax(),"hour"])
        fb.add_vline(x=ph,line_color="#E07B39",line_dash="dot",annotation_text=f"Peak:{ph}:00",annotation_font_color="#E07B39")
        fb.update_layout(height=270,xaxis_title="Hour",yaxis_title="Avg Revenue ($)"); st.plotly_chart(fb,width='stretch')
    with c2:
        dw=fr.groupby("day_name")["revenue"].sum().reset_index()
        dw["day_name"]=pd.Categorical(dw["day_name"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ordered=True)
        dw=dw.sort_values("day_name"); fd2=F("Revenue by Day of Week")
        fd2.add_trace(go.Bar(x=dw["day_name"],y=dw["revenue"],marker_color=["#C8722A" if d in ["Saturday","Sunday"] else "#8B7355" for d in dw["day_name"]]))
        fd2.update_layout(height=270,yaxis_title="Revenue ($)"); st.plotly_chart(fd2,width='stretch')
    rh=raw.groupby(["store_location","hour"])["transaction_id"].count().reset_index().rename(columns={"transaction_id":"txn"})
    fr2=F("Rush Hour by Store")
    for store,g in rh.groupby("store_location"):
        fr2.add_trace(go.Scatter(x=g["hour"],y=g["txn"],name=store,mode="lines+markers",line=dict(color=SC.get(store,"#888"),width=2.5),marker=dict(size=5)))
    fr2.update_layout(height=290,xaxis_title="Hour",yaxis_title="Transactions"); st.plotly_chart(fr2,width='stretch')

# ─── TAB 3: STORE COMPARISON ──────────────────────────────────────────────────
with T[2]:
    st.markdown("## Store Comparison")
    c1,c2=st.columns(2)
    with c1:
        sr=raw.groupby("store_location")["revenue"].sum().reset_index()
        fp=go.Figure(go.Pie(labels=sr["store_location"],values=sr["revenue"],hole=0.45,marker=dict(colors=["#C8722A","#5B8DB8","#5C8C5A"])))
        fp.update_layout(title=dict(text="Revenue Share",font=dict(size=15,family="Playfair Display",color="#C8722A")),height=300,**LY); st.plotly_chart(fp,width='stretch')
    with c2:
        mn=raw.groupby(["month_name","store_location"])["revenue"].sum().reset_index()
        mn["month_name"]=pd.Categorical(mn["month_name"],["Jan","Feb","Mar","Apr","May","Jun"],ordered=True); mn=mn.sort_values("month_name")
        fm=F("Monthly Revenue per Store")
        for store,g in mn.groupby("store_location"): fm.add_trace(go.Bar(x=g["month_name"],y=g["revenue"],name=store,marker_color=SC.get(store,"#888")))
        fm.update_layout(barmode="group",height=300,xaxis_title="Month",yaxis_title="Revenue ($)"); st.plotly_chart(fm,width='stretch')
    fl=F("Daily Revenue (7-Day MA) by Store")
    for store,g in dly.groupby("store_location"):
        fl.add_trace(go.Scatter(x=g.sort_values("date")["date"],y=g.sort_values("date")["roll7"],name=store,line=dict(color=SC.get(store,"#888"),width=2.5)))
    fl.update_layout(height=320,xaxis_title="Date",yaxis_title="Revenue ($)",legend=dict(orientation="h",y=-0.18)); st.plotly_chart(fl,width='stretch')
    st.dataframe(pd.DataFrame([{"Store":s,"Total Rev ($)":f"{dly[dly['store_location']==s]['revenue'].sum():,.0f}",
        "Avg Daily ($)":f"{dly[dly['store_location']==s]['revenue'].mean():,.0f}",
        "Peak Day ($)":f"{dly[dly['store_location']==s]['revenue'].max():,.0f}",
        "Total Txn":f"{dly[dly['store_location']==s]['transactions'].sum():,}"}
        for s in ["Lower Manhattan","Hell's Kitchen","Astoria"]]),width='stretch',hide_index=True)

# ─── TAB 4: CATEGORY ANALYSIS ─────────────────────────────────────────────────
with T[3]:
    st.markdown("## Category Analysis")
    c1,c2=st.columns(2)
    with c1:
        cr=fr.groupby("product_category")["revenue"].sum().sort_values(ascending=False).reset_index(); fc1=F("Revenue by Category")
        fc1.add_trace(go.Bar(x=cr["revenue"],y=cr["product_category"],orientation="h",marker=dict(color=cr["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#3D1F0A"]])))
        fc1.update_layout(height=320,yaxis=dict(autorange="reversed"),xaxis_title="Revenue ($)"); st.plotly_chart(fc1,width='stretch')
    with c2:
        cq=fr.groupby("product_category")["transaction_qty"].sum().sort_values(ascending=False).reset_index(); fc2=F("Units Sold by Category")
        fc2.add_trace(go.Bar(x=cq["transaction_qty"],y=cq["product_category"],orientation="h",marker=dict(color=cq["transaction_qty"],colorscale=[[0,"#E8D5B0"],[1,"#5B8DB8"]])))
        fc2.update_layout(height=320,yaxis=dict(autorange="reversed"),xaxis_title="Units Sold"); st.plotly_chart(fc2,width='stretch')
    tp=fr.groupby("product_detail")["revenue"].sum().sort_values(ascending=False).head(15).reset_index(); ft=F("Top 15 Products by Revenue")
    ft.add_trace(go.Bar(x=tp["revenue"],y=tp["product_detail"],orientation="h",marker=dict(color=tp["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#C8722A"]])))
    ft.update_layout(height=400,yaxis=dict(autorange="reversed"),xaxis_title="Revenue ($)"); st.plotly_chart(ft,width='stretch')

# ─── TAB 5: MODEL EVALUATION ──────────────────────────────────────────────────
with T[4]:
    st.markdown("## Model Evaluation")
    ev_s=agg["revenue"].ffill(); SP=14; tr2=ev_s.iloc[:-SP]; te=ev_s.iloc[-SP:]; td=agg["date"].iloc[-SP:]
    with st.spinner("Evaluating models..."):
        res={}
        for n,p in [("Naive",naive(tr2,SP)),("Moving Avg",ma(tr2,SP)),("Exp Smooth",ema(tr2,SP))]: res[n]={"preds":p,**evm(te,p)}
        for n,fn in [("Linear Trend",lin),("Gradient Boost",gbr),("SARIMA",sarima)]:
            try: p,_,_=fn(tr2,SP); res[n]={"preds":p,**evm(te,p)}
            except: pass
    mdf=pd.DataFrame([{"Model":k,"MAE":v["MAE"],"RMSE":v["RMSE"],"MAPE(%)":v["MAPE(%)"]} for k,v in res.items()]).sort_values("RMSE").reset_index(drop=True)
    st.success(f"🏆 Best: **{mdf.iloc[0]['Model']}** (lowest RMSE)")
    st.dataframe(mdf, width='stretch', hide_index=True)    
    fe=F("Actual vs Predicted"); fe.add_trace(go.Scatter(x=td,y=te.values,name="Actual",line=dict(color="#1C1008",width=3)))
    pal=["#C8722A","#5B8DB8","#5C8C5A","#8B7355","#E07B39","#9B59B6"]
    for i,(n,v) in enumerate(res.items()): fe.add_trace(go.Scatter(x=td,y=v["preds"],name=n,line=dict(color=pal[i%len(pal)],width=1.8,dash="dash")))
    fe.update_layout(height=340,xaxis_title="Date",yaxis_title="Revenue ($)",legend=dict(orientation="h",y=-0.2)); st.plotly_chart(fe,width='stretch')
    fmape=F("MAPE by Model (lower = better)")
    fmape.add_trace(go.Bar(x=mdf["MAPE(%)"],y=mdf["Model"],orientation="h",marker=dict(color=mdf["MAPE(%)"],colorscale=[[0,"#5C8C5A"],[0.5,"#C8722A"],[1,"#C04A3A"]])))
    fmape.update_layout(height=260,xaxis_title="MAPE (%)",yaxis=dict(autorange="reversed")); st.plotly_chart(fmape,width='stretch')

# ─── TAB 6: EDA & TRENDS ──────────────────────────────────────────────────────
with T[5]:
    st.markdown("## EDA & Trends")
    fe2=F("Daily Revenue with 7-Day MA")
    for store,g in dly.groupby("store_location"):
        g=g.sort_values("date")
        fe2.add_trace(go.Scatter(x=g["date"],y=g["revenue"],name=store+" (raw)",line=dict(color=SC.get(store,"#888"),width=1),opacity=0.35))
        fe2.add_trace(go.Scatter(x=g["date"],y=g["roll7"],name=store+" (7-MA)",line=dict(color=SC.get(store,"#888"),width=2.5)))
    fe2.update_layout(height=330,xaxis_title="Date",yaxis_title="Revenue ($)"); st.plotly_chart(fe2,width='stretch')
    c1,c2=st.columns(2)
    with c1:
        fd3=F("Revenue Distribution by Store")
        for store,g in dly.groupby("store_location"): fd3.add_trace(go.Histogram(x=g["revenue"],name=store,opacity=0.7,nbinsx=30,marker_color=SC.get(store,"#888")))
        fd3.update_layout(barmode="overlay",height=290,xaxis_title="Daily Revenue ($)"); st.plotly_chart(fd3,width='stretch')
    with c2:
        dw2=agg.copy(); dw2["dow"]=dw2["date"].dt.day_name()
        da=dw2.groupby("dow")["revenue"].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index()
        fw=F("Avg Revenue by Day of Week")
        fw.add_trace(go.Bar(x=da["dow"],y=da["revenue"],marker_color=["#C8722A" if d in ["Friday","Saturday"] else "#C8A87A" for d in da["dow"]]))
        fw.update_layout(height=290,yaxis_title="Avg Revenue ($)"); st.plotly_chart(fw,width='stretch')
    with st.expander("🔍 Raw Data Explorer"):
        n=st.slider("Rows",10,500,100)
        cols=st.multiselect("Columns",raw.columns.tolist(),default=["date","store_location","product_category","transaction_qty","unit_price","revenue"])
        if cols: st.dataframe(fr[cols].head(n),width='stretch')
    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1: st.metric("Total Records",f"{len(raw):,}")
    with c2: st.metric("Date Range","Jan–Jun 2025")
    with c3: st.metric("Stores",raw["store_location"].nunique())

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div style="text-align:center;color:#8B7355;font-size:0.78rem;">☕ Afficionado Coffee Roasters · 2025 · Streamlit · Plotly · scikit-learn</div>',unsafe_allow_html=True)
