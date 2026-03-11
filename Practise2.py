import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Afficionado Coffee", page_icon="☕", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#FAF5EC;color:#1C1008;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1C1008,#3D1F0A);border-right:2px solid #C8722A;}
[data-testid="stSidebar"] *{color:#F5ECD7 !important;}
h1,h2,h3{font-family:'Playfair Display',serif;color:#C8722A;}
h2{border-bottom:2px solid #C8722A;padding-bottom:4px;}
.kpi{background:linear-gradient(135deg,#3D1F0A,#1C1008);border-radius:12px;padding:16px 18px;
     border-left:4px solid #C8722A;margin-bottom:8px;}
.kpi .val{font-size:1.65rem;font-weight:700;font-family:'Playfair Display',serif;color:#C8722A;}
.kpi .lbl{font-size:0.71rem;text-transform:uppercase;letter-spacing:0.07em;color:#E8D5B0;margin-top:3px;}
.kpi .dlt{font-size:0.8rem;margin-top:4px;}
.pos{color:#7ECB7A;} .neg{color:#E88080;}
.badge{display:inline-block;background:#C8722A;color:white;border-radius:20px;
       padding:3px 14px;font-size:0.78rem;font-weight:600;text-transform:uppercase;margin-bottom:10px;}
.fc-box{background:white;border-radius:10px;padding:10px 14px;margin-bottom:8px;
        box-shadow:0 1px 6px rgba(0,0,0,0.07);border-top:3px solid #C8722A;}
.fc-lbl{font-size:0.7rem;text-transform:uppercase;color:#8B7355;letter-spacing:0.06em;}
.fc-val{font-size:1.1rem;font-weight:700;color:#3D1F0A;font-family:'Playfair Display',serif;}
</style>""", unsafe_allow_html=True)

COLORS = {"Lower Manhattan":"#C8722A","Hell's Kitchen":"#5B8DB8","Astoria":"#5C8C5A","All Stores":"#8B7355"}

# Headings: Caramel #C8722A | Legend text: Black #000000
LAY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFF8F0",
           font=dict(family="DM Sans", color="#1C1008"), margin=dict(l=20,r=20,t=40,b=20),
           hoverlabel=dict(bgcolor="#3D1F0A", font_color="#F5ECD7", font_size=13),
           legend=dict(bgcolor="#FFF8F0", bordercolor="#C8722A", borderwidth=1,
                       font=dict(color="#000000")))

def fig(title=""):
    f = go.Figure()
    f.update_layout(
        title=dict(text=title, font=dict(size=15, family="Playfair Display", color="#C8722A")),
        **LAY)
    return f

# ── Data ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Brewing insights... ☕")
def load_data(path):
    df = pd.read_csv(path).sort_values("transaction_id").reset_index(drop=True)
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    idx = (df.index.to_numpy() * 181 // len(df)).clip(0, 180)
    df["date"]        = pd.to_datetime("2025-01-01") + pd.to_timedelta(idx, unit="D")
    df["hour"]        = df["transaction_time"].str.split(":").str[0].astype(int)
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"]    = df["date"].dt.day_name()
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%b")
    return df

@st.cache_data
def build_daily(df):
    d = (df.groupby(["date","store_location"])
           .agg(revenue=("revenue","sum"), transactions=("transaction_id","count"), qty=("transaction_qty","sum"))
           .reset_index().sort_values(["store_location","date"]).reset_index(drop=True))
    parts = []
    for s in d["store_location"].unique():
        g = d[d["store_location"]==s].copy().sort_values("date").reset_index(drop=True)
        g["roll7"] = g["revenue"].rolling(7).mean()
        parts.append(g)
    return pd.concat(parts, ignore_index=True)

# ── Forecast models ───────────────────────────────────────────────
def naive_fc(s,h):      return np.full(h, s.iloc[-1])
def ma_fc(s,h,w=7):     return np.full(h, s.rolling(w,min_periods=1).mean().iloc[-1])
def ema_fc(s,h,a=0.3):  return np.full(h, s.ewm(alpha=a,adjust=False).mean().iloc[-1])

def linear_fc(s,h):
    from sklearn.linear_model import LinearRegression
    x=np.arange(len(s)).reshape(-1,1); y=s.values
    m=LinearRegression().fit(x,y)
    p=m.predict(np.arange(len(s),len(s)+h).reshape(-1,1))
    std=(y-m.predict(x)).std()
    return p, p-1.96*std, p+1.96*std

def gbr_fc(s,h):
    from sklearn.ensemble import GradientBoostingRegressor
    L=7; sv=s.values
    if len(sv)<L+5: p=np.full(h,sv.mean()); return p,p-sv.std(),p+sv.std()
    X=[sv[i-L:i] for i in range(L,len(sv))]; y=sv[L:]
    X,y=np.array(X),np.array(y)
    m=GradientBoostingRegressor(n_estimators=120,learning_rate=0.08,max_depth=3,random_state=42).fit(X,y)
    hist=list(sv[-L:]); preds=[]
    for _ in range(h):
        p=m.predict(np.array(hist[-L:]).reshape(1,-1))[0]; preds.append(p); hist.append(p)
    p=np.array(preds); std=(y-m.predict(X)).std()
    return p, p-1.96*std, p+1.96*std

def sarima_fc(s,h):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        if len(s)<30: return linear_fc(s,h)
        fit=SARIMAX(s.values,order=(1,1,1),seasonal_order=(1,1,1,7),
                    enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
        fc=fit.get_forecast(h); ci=fc.conf_int()
        return fc.predicted_mean, ci[:,0], ci[:,1]
    except: return linear_fc(s,h)

def eval_model(a,p):
    a,p=np.array(a,float),np.array(p,float)
    return {"MAE":round(np.mean(np.abs(a-p)),2),
            "RMSE":round(np.sqrt(np.mean((a-p)**2)),2),
            "MAPE(%)":round(np.mean(np.abs((a-p)/(a+1e-9)))*100,2)}

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☕ Afficionado\nCoffee Roasters")
    st.markdown("---")
    DATA_PATH      = st.text_input("Dataset path", "Afficionado_Coffee_Roasters.csv")
    selected_store = st.selectbox("🏪 Store", ["All Stores","Lower Manhattan","Hell's Kitchen","Astoria"])
    horizon_days   = st.slider("📅 Forecast days", 1, 30, 7)
    model_choice   = st.radio("🤖 Model", ["Gradient Boosting","SARIMA","Exp Smoothing","Moving Average","Naive","Linear Trend","Compare All"])
    metric_toggle  = st.radio("📊 Metric", ["Revenue ($)","Transactions"])
    show_ci        = st.checkbox("Confidence intervals", True)
    show_scenarios = st.checkbox("Scenario analysis", False)
    st.caption("Afficionado Coffee · v1.0")

# ── Load data ─────────────────────────────────────────────────────
try:
    raw_df   = load_data(DATA_PATH)
    daily_df = build_daily(raw_df)
except FileNotFoundError:
    st.error(f"❌ File not found: `{DATA_PATH}`"); st.stop()

def filt(df): return df if selected_store=="All Stores" else df[df["store_location"]==selected_store]
filtered_raw   = filt(raw_df)
filtered_daily = filt(daily_df)
agg_daily = (filt(daily_df).groupby("date")
             .agg(revenue=("revenue","sum"),transactions=("transactions","sum"))
             .reset_index().sort_values("date").reset_index(drop=True))
agg_daily["roll7"] = agg_daily["revenue"].rolling(7).mean()

# ── Header & KPIs ─────────────────────────────────────────────────
st.markdown("<h1>☕ Afficionado Coffee Roasters</h1><p style='color:#8B7355;margin-top:-8px;'>Forecasting & Peak Demand Dashboard</p>", unsafe_allow_html=True)
st.markdown(f'<div class="badge">📍 {selected_store} · {horizon_days}-Day Forecast</div>', unsafe_allow_html=True)

total_rev=agg_daily["revenue"].sum(); total_txn=agg_daily["transactions"].sum()
avg_rev=agg_daily["revenue"].mean(); peak_rev=agg_daily["revenue"].max()
peak_day=agg_daily.loc[agg_daily["revenue"].idxmax(),"date"].strftime("%d %b")
m1=agg_daily[agg_daily["date"].dt.month==1]["revenue"].sum()
m2=agg_daily[agg_daily["date"].dt.month==2]["revenue"].sum()
mom=((m2-m1)/(m1+1e-9))*100

def kpi(val,lbl,dlt=None,pos=True):
    d=f'<div class="dlt {"pos" if pos else "neg"}">{"▲" if pos else "▼"} {dlt}</div>' if dlt else ""
    return f'<div class="kpi"><div class="val">{val}</div><div class="lbl">{lbl}</div>{d}</div>'

c1,c2,c3,c4,c5=st.columns(5)
with c1: st.markdown(kpi(f"${total_rev:,.0f}","Total Revenue"), unsafe_allow_html=True)
with c2: st.markdown(kpi(f"{total_txn:,}","Transactions"), unsafe_allow_html=True)
with c3: st.markdown(kpi(f"${avg_rev:,.0f}","Avg Daily Rev"), unsafe_allow_html=True)
with c4: st.markdown(kpi(f"${peak_rev:,.0f}",f"Peak ({peak_day})"), unsafe_allow_html=True)
with c5: st.markdown(kpi(f"{mom:+.1f}%","Jan→Feb",dlt=f"{abs(mom):.1f}%",pos=mom>=0), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

tabs = st.tabs(["📈 Sales Forecast","🔥 Peak Demand","🏪 Store Comparison","📦 Categories","🤖 Model Eval","📊 EDA"])

# ═══════════════════════════════════════════════
# TAB 1 — SALES FORECAST
# ═══════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Sales & Revenue Forecast")
    mcol   = "revenue" if "Revenue" in metric_toggle else "transactions"
    mlabel = "Revenue ($)" if "Revenue" in metric_toggle else "Transactions"
    series = agg_daily[mcol].astype(float).ffill()
    dates  = agg_daily["date"]
    fdates = pd.date_range(dates.iloc[-1]+pd.Timedelta(days=1), periods=horizon_days)

    with st.spinner("Forecasting..."):
        lo=hi=None
        if   model_choice=="Naive":             preds=naive_fc(series,horizon_days)
        elif model_choice=="Moving Average":    preds=ma_fc(series,horizon_days)
        elif model_choice=="Exp Smoothing":     preds=ema_fc(series,horizon_days)
        elif model_choice=="Linear Trend":      preds,lo,hi=linear_fc(series,horizon_days)
        elif model_choice=="Gradient Boosting": preds,lo,hi=gbr_fc(series,horizon_days)
        elif model_choice=="SARIMA":            preds,lo,hi=sarima_fc(series,horizon_days)
        else:                                   preds,lo,hi=gbr_fc(series,horizon_days)
        preds=np.maximum(np.array(preds,float),0)
        if lo is not None: lo,hi=np.maximum(lo,0),np.maximum(hi,0)

    f1=fig(f"{mlabel} Forecast · {selected_store}")
    f1.add_trace(go.Scatter(x=dates,y=series,name="Historical",line=dict(color=COLORS.get(selected_store,"#C8722A"),width=2)))
    f1.add_trace(go.Scatter(x=dates,y=series.rolling(7).mean(),name="7-Day MA",
                            line=dict(color="#8B7355",width=1.5,dash="dot"),opacity=0.8))
    f1.add_trace(go.Scatter(x=fdates,y=preds,name=f"Forecast ({model_choice})",
                            line=dict(color="#E07B39",width=2.5,dash="dash")))
    if show_ci and lo is not None:
        f1.add_trace(go.Scatter(x=list(fdates)+list(fdates[::-1]),y=list(hi)+list(lo[::-1]),
                                fill="toself",fillcolor="rgba(200,114,42,0.15)",
                                line=dict(color="rgba(0,0,0,0)"),name="95% CI"))
    f1.add_vline(x=str(dates.iloc[-1]), line_dash="dot", line_color="#8B7355")
    f1.add_annotation(x=str(dates.iloc[-1]), y=1, yref="paper", text="Forecast Start",
                      showarrow=False, xanchor="left", font=dict(color="#8B7355", size=12))
    f1.update_layout(height=400,xaxis_title="Date",yaxis_title=mlabel,legend=dict(orientation="h",y=-0.15))
    st.plotly_chart(f1, use_container_width=True)

    fmt=(lambda v:f"${v:,.0f}") if "Revenue" in metric_toggle else (lambda v:f"{int(v):,}")
    ct,cs=st.columns([3,2])
    with ct:
        st.markdown("#### 📋 Forecast Table")
        fc_df=pd.DataFrame({"Date":fdates.strftime("%a, %d %b"),
                             f"Forecast {mlabel}":[fmt(p) for p in preds]})
        if lo is not None: fc_df["Lower"]=[fmt(v) for v in lo]; fc_df["Upper"]=[fmt(v) for v in hi]
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with cs:
        st.markdown("#### 📌 Forecast Summary")
        for lbl,val in [("Mean Sales",fmt(preds.mean())),("Min Sales",fmt(preds.min())),
                         ("Max Sales",fmt(preds.max())),("Total Sales",fmt(preds.sum()))]:
            st.markdown(f'<div class="fc-box"><div class="fc-lbl">{lbl}</div><div class="fc-val">{val}</div></div>',
                        unsafe_allow_html=True)

    if show_scenarios:
        st.markdown("### 🔮 Scenario Analysis")
        fs=fig("Best / Base / Worst Case")
        fs.add_trace(go.Scatter(x=fdates,y=preds*1.2,name="Best (+20%)",line=dict(color="#5C8C5A",width=2)))
        fs.add_trace(go.Scatter(x=fdates,y=preds,name="Base",line=dict(color="#C8722A",width=2.5)))
        fs.add_trace(go.Scatter(x=fdates,y=preds*0.8,name="Worst (-20%)",line=dict(color="#C04A3A",width=2)))
        fs.update_layout(height=300); st.plotly_chart(fs, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 — PEAK DEMAND
# ═══════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Peak Demand Detection")
    hr_pivot=(filtered_raw.groupby(["day_of_week","hour"])["revenue"].sum().reset_index()
              .pivot(index="day_of_week",columns="hour",values="revenue").fillna(0))
    hr_pivot.index=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(hr_pivot)]
    fh=go.Figure(go.Heatmap(z=hr_pivot.values,x=[f"{h}:00" for h in hr_pivot.columns],y=hr_pivot.index,
                             colorscale=[[0,"#FAF5EC"],[0.5,"#C8722A"],[1,"#3D1F0A"]],
                             hovertemplate="Day:%{y}  Hour:%{x}<br>$%{z:,.0f}<extra></extra>"))
    fh.update_layout(title=dict(text="Revenue Heatmap (Day × Hour)",font=dict(size=15,family="Playfair Display",color="#C8722A")),
                     height=290,**LAY,xaxis_title="Hour")
    st.plotly_chart(fh, use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        avg_h=filtered_raw.groupby("hour")["revenue"].mean().reset_index()
        fb=fig("Avg Revenue by Hour")
        fb.add_trace(go.Bar(x=avg_h["hour"],y=avg_h["revenue"],
                            marker=dict(color=avg_h["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#3D1F0A"]],showscale=False)))
        ph=int(avg_h.loc[avg_h["revenue"].idxmax(),"hour"])
        fb.add_vline(x=ph,line_color="#E07B39",line_dash="dot",annotation_text=f"Peak:{ph}:00",annotation_font_color="#E07B39")
        fb.update_layout(height=270,xaxis_title="Hour",yaxis_title="Avg Revenue ($)")
        st.plotly_chart(fb, use_container_width=True)
    with c2:
        dow=filtered_raw.groupby("day_name")["revenue"].sum().reset_index()
        dow["day_name"]=pd.Categorical(dow["day_name"],["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],ordered=True)
        fd=fig("Revenue by Day of Week")
        fd.add_trace(go.Bar(x=dow.sort_values("day_name")["day_name"],y=dow.sort_values("day_name")["revenue"],
                            marker_color=["#C8722A" if d in ["Saturday","Sunday"] else "#8B7355" for d in dow.sort_values("day_name")["day_name"]]))
        fd.update_layout(height=270,yaxis_title="Revenue ($)")
        st.plotly_chart(fd, use_container_width=True)

    rh=(raw_df.groupby(["store_location","hour"])["transaction_id"].count()
        .reset_index().rename(columns={"transaction_id":"txn"}))
    fr=fig("Rush Hour by Store")
    for store,g in rh.groupby("store_location"):
        fr.add_trace(go.Scatter(x=g["hour"],y=g["txn"],name=store,mode="lines+markers",
                                line=dict(color=COLORS.get(store,"#888"),width=2.5),marker=dict(size=5)))
    fr.update_layout(height=290,xaxis_title="Hour",yaxis_title="Transactions")
    st.plotly_chart(fr, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 — STORE COMPARISON
# ═══════════════════════════════════════════════
with tabs[2]:
    st.markdown("## Store Comparison")
    c1,c2=st.columns(2)
    with c1:
        sr=raw_df.groupby("store_location")["revenue"].sum().reset_index()
        fp=go.Figure(go.Pie(labels=sr["store_location"],values=sr["revenue"],hole=0.45,
                             marker=dict(colors=["#C8722A","#5B8DB8","#5C8C5A"])))
        fp.update_layout(title=dict(text="Revenue Share",font=dict(size=15,family="Playfair Display",color="#C8722A")),
                         height=300,**LAY)
        st.plotly_chart(fp, use_container_width=True)
    with c2:
        mon=raw_df.groupby(["month_name","store_location"])["revenue"].sum().reset_index()
        mon["month_name"]=pd.Categorical(mon["month_name"],["Jan","Feb","Mar","Apr","May","Jun"],ordered=True)
        mon=mon.sort_values("month_name")
        fm=fig("Monthly Revenue per Store")
        for store,g in mon.groupby("store_location"):
            fm.add_trace(go.Bar(x=g["month_name"],y=g["revenue"],name=store,marker_color=COLORS.get(store,"#888")))
        fm.update_layout(barmode="group",height=300,xaxis_title="Month",yaxis_title="Revenue ($)")
        st.plotly_chart(fm, use_container_width=True)

    fl=fig("Daily Revenue (7-Day MA) by Store")
    for store,g in daily_df.groupby("store_location"):
        fl.add_trace(go.Scatter(x=g.sort_values("date")["date"],y=g.sort_values("date")["roll7"],
                                name=store,line=dict(color=COLORS.get(store,"#888"),width=2.5)))
    fl.update_layout(height=320,xaxis_title="Date",yaxis_title="Revenue ($)",legend=dict(orientation="h",y=-0.18))
    st.plotly_chart(fl, use_container_width=True)

    rows=[{"Store":s,"Total Rev ($)":f"{daily_df[daily_df['store_location']==s]['revenue'].sum():,.0f}",
           "Avg Daily ($)":f"{daily_df[daily_df['store_location']==s]['revenue'].mean():,.0f}",
           "Peak Day ($)":f"{daily_df[daily_df['store_location']==s]['revenue'].max():,.0f}",
           "Total Txn":f"{daily_df[daily_df['store_location']==s]['transactions'].sum():,}"}
          for s in ["Lower Manhattan","Hell's Kitchen","Astoria"]]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# TAB 4 — CATEGORIES
# ═══════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Category Analysis")
    c1,c2=st.columns(2)
    with c1:
        cr=filtered_raw.groupby("product_category")["revenue"].sum().sort_values(ascending=False).reset_index()
        fc1=fig("Revenue by Category")
        fc1.add_trace(go.Bar(x=cr["revenue"],y=cr["product_category"],orientation="h",
                             marker=dict(color=cr["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#3D1F0A"]])))
        fc1.update_layout(height=320,yaxis=dict(autorange="reversed"),xaxis_title="Revenue ($)")
        st.plotly_chart(fc1, use_container_width=True)
    with c2:
        cq=filtered_raw.groupby("product_category")["transaction_qty"].sum().sort_values(ascending=False).reset_index()
        fc2=fig("Units Sold by Category")
        fc2.add_trace(go.Bar(x=cq["transaction_qty"],y=cq["product_category"],orientation="h",
                             marker=dict(color=cq["transaction_qty"],colorscale=[[0,"#E8D5B0"],[1,"#5B8DB8"]])))
        fc2.update_layout(height=320,yaxis=dict(autorange="reversed"),xaxis_title="Units Sold")
        st.plotly_chart(fc2, use_container_width=True)

    tp=filtered_raw.groupby("product_detail")["revenue"].sum().sort_values(ascending=False).head(15).reset_index()
    ft=fig("Top 15 Products by Revenue")
    ft.add_trace(go.Bar(x=tp["revenue"],y=tp["product_detail"],orientation="h",
                        marker=dict(color=tp["revenue"],colorscale=[[0,"#E8D5B0"],[1,"#C8722A"]])))
    ft.update_layout(height=400,yaxis=dict(autorange="reversed"),xaxis_title="Revenue ($)")
    st.plotly_chart(ft, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 5 — MODEL EVALUATION
# ═══════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Model Evaluation")
    ev=agg_daily["revenue"].ffill(); SPLIT=14
    train=ev.iloc[:-SPLIT]; test=ev.iloc[-SPLIT:]; tdates=agg_daily["date"].iloc[-SPLIT:]

    with st.spinner("Evaluating models..."):
        res={}
        for name,pred in [("Naive",naive_fc(train,SPLIT)),("Moving Avg",ma_fc(train,SPLIT)),("Exp Smooth",ema_fc(train,SPLIT))]:
            res[name]={"preds":pred,**eval_model(test,pred)}
        for name,fn in [("Linear Trend",linear_fc),("Gradient Boost",gbr_fc),("SARIMA",sarima_fc)]:
            try: p,_,_=fn(train,SPLIT); res[name]={"preds":p,**eval_model(test,p)}
            except: pass

    mdf=(pd.DataFrame([{"Model":k,"MAE":v["MAE"],"RMSE":v["RMSE"],"MAPE(%)":v["MAPE(%)"]}
                        for k,v in res.items()]).sort_values("RMSE").reset_index(drop=True))
    st.success(f"🏆 Best: **{mdf.iloc[0]['Model']}** (lowest RMSE)")
    st.dataframe(mdf.style.background_gradient(cmap="YlOrBr",subset=["MAE","RMSE","MAPE(%)"]),
                 use_container_width=True, hide_index=True)

    fe=fig("Actual vs Predicted")
    fe.add_trace(go.Scatter(x=tdates,y=test.values,name="Actual",line=dict(color="#1C1008",width=3)))
    pal=["#C8722A","#5B8DB8","#5C8C5A","#8B7355","#E07B39","#9B59B6"]
    for i,(name,v) in enumerate(res.items()):
        fe.add_trace(go.Scatter(x=tdates,y=v["preds"],name=name,line=dict(color=pal[i%len(pal)],width=1.8,dash="dash")))
    fe.update_layout(height=340,xaxis_title="Date",yaxis_title="Revenue ($)",legend=dict(orientation="h",y=-0.2))
    st.plotly_chart(fe, use_container_width=True)

    fm2=fig("MAPE by Model (lower = better)")
    fm2.add_trace(go.Bar(x=mdf["MAPE(%)"],y=mdf["Model"],orientation="h",
                         marker=dict(color=mdf["MAPE(%)"],colorscale=[[0,"#5C8C5A"],[0.5,"#C8722A"],[1,"#C04A3A"]])))
    fm2.update_layout(height=260,xaxis_title="MAPE (%)",yaxis=dict(autorange="reversed"))
    st.plotly_chart(fm2, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 6 — EDA
# ═══════════════════════════════════════════════
with tabs[5]:
    st.markdown("## EDA & Trends")
    fe2=fig("Daily Revenue with 7-Day MA")
    for store,g in daily_df.groupby("store_location"):
        g=g.sort_values("date")
        fe2.add_trace(go.Scatter(x=g["date"],y=g["revenue"],name=store+" (raw)",
                                 line=dict(color=COLORS.get(store,"#888"),width=1),opacity=0.35))
        fe2.add_trace(go.Scatter(x=g["date"],y=g["roll7"],name=store+" (7-MA)",
                                 line=dict(color=COLORS.get(store,"#888"),width=2.5)))
    fe2.update_layout(height=330,xaxis_title="Date",yaxis_title="Revenue ($)")
    st.plotly_chart(fe2, use_container_width=True)

    c1,c2=st.columns(2)
    with c1:
        fdist=fig("Revenue Distribution by Store")
        for store,g in daily_df.groupby("store_location"):
            fdist.add_trace(go.Histogram(x=g["revenue"],name=store,opacity=0.7,nbinsx=30,
                                          marker_color=COLORS.get(store,"#888")))
        fdist.update_layout(barmode="overlay",height=290,xaxis_title="Daily Revenue ($)")
        st.plotly_chart(fdist, use_container_width=True)
    with c2:
        dow2=agg_daily.copy(); dow2["dow"]=dow2["date"].dt.day_name()
        da=(dow2.groupby("dow")["revenue"].mean()
            .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).reset_index())
        fw=fig("Avg Revenue by Day of Week")
        fw.add_trace(go.Bar(x=da["dow"],y=da["revenue"],
                            marker_color=["#C8722A" if d in ["Friday","Saturday"] else "#C8A87A" for d in da["dow"]]))
        fw.update_layout(height=290,yaxis_title="Avg Revenue ($)")
        st.plotly_chart(fw, use_container_width=True)

    with st.expander("🔍 Raw Data Explorer"):
        n=st.slider("Rows",10,500,100)
        cols=st.multiselect("Columns",raw_df.columns.tolist(),
                             default=["date","store_location","product_category","transaction_qty","unit_price","revenue"])
        if cols: st.dataframe(filtered_raw[cols].head(n), use_container_width=True)

    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1: st.metric("Total Records",f"{len(raw_df):,}")
    with c2: st.metric("Date Range","Jan–Jun 2025")
    with c3: st.metric("Stores",raw_df["store_location"].nunique())

st.markdown("---")
st.markdown('<div style="text-align:center;color:#8B7355;font-size:0.78rem;">☕ Afficionado Coffee Roasters · 2025 · Streamlit · Plotly · scikit-learn</div>',
            unsafe_allow_html=True)