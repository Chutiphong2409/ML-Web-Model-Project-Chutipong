
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report, mean_absolute_error,
                             mean_squared_error, r2_score, silhouette_score)

ROOT = Path(__file__).parent
st.set_page_config(page_title="ML Final Project | Chutipong", page_icon="🤖",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root {
  --ink:#18352f; --forest:#174f43; --leaf:#2f806b;
  --mint:#dcefe7; --paper:#f7f8f2; --coral:#e8795b; --gold:#e8b65d;
}
.stApp { background:#000000; color: #ffffff; }
[data-testid="stHeader"] { background:rgba(247,248,242,.92); }
[data-testid="stSidebar"] {
  background:linear-gradient(165deg,#e2f0e8 0%,#d3e7dc 58%,#c4dfd2 100%);
  border-right:1px solid #c7ddd1;
}
[data-testid="stSidebar"] * { color:#183b32 !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
  padding:.55rem .7rem; border-radius:11px; margin:2px 0;
  transition:background .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
  background:rgba(47,128,107,.12);
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
  background:#fff; box-shadow:0 3px 12px rgba(24,70,57,.08);
  font-weight:750;
}
.block-container { max-width:1260px; padding-top:2rem; padding-bottom:3rem; }
.hero {
  background:linear-gradient(118deg,#173f36 0%,#236b58 72%,#39856d 100%);
  border-radius:10px 34px 10px 34px; padding:34px 38px; color:#fff;
  margin:0 0 26px 0; position:relative; overflow:hidden;
  box-shadow:0 14px 30px rgba(28,76,61,.13);
  border-left:7px solid var(--gold);
}
.hero:after {
  content:""; position:absolute; right:35px; bottom:-72px;
  width:190px; height:190px; border:1px solid rgba(255,255,255,.22);
  border-radius:42% 58% 55% 45%; transform:rotate(28deg);
  box-shadow:0 0 0 22px rgba(255,255,255,.035),0 0 0 44px rgba(255,255,255,.025);
}
.hero .eyebrow {
  display:inline-block; color:#f8e4b5; border:1px solid rgba(248,228,181,.48);
  background:rgba(255,255,255,.07); padding:6px 12px; border-radius:7px;
  font-size:11px; letter-spacing:1px; margin-bottom:15px; font-weight:750;
}
.hero h1 {
  color:white; font-size:clamp(1.55rem,3vw,2.35rem); line-height:1.25;
  margin:0 0 10px 0; font-weight:800; max-width:850px;
}
.hero p { color:#e0f0e8; margin:0; font-size:.94rem; max-width:850px; }
h1,h2,h3 { color:#1b493d; letter-spacing:-.02em; }
h2 { margin-top:1.2rem; }
div[data-testid="stMetric"] {
  background:#fff; border:1px solid #e2e9e1; border-top:3px solid #6ba88e;
  padding:16px 17px; border-radius:5px 16px 5px 16px;
  box-shadow:0 5px 16px rgba(35,73,59,.045);
}
div[data-testid="stMetric"] label { color:#638075 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#174f43; }
div[data-testid="stTabs"] button { font-weight:700; color:#547166; }
div[data-testid="stTabs"] button[aria-selected="true"] {
  color:#1c6a55; border-bottom-color:#e8795b;
}
div[data-testid="stForm"] {
  background:#fff; border:1px solid #e0e9df; border-radius:16px 5px 16px 5px;
  padding:20px; box-shadow:0 6px 20px rgba(35,73,59,.04);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background:linear-gradient(100deg,#1c6a55,#3d9276); color:white; border:0;
  border-radius:8px 8px 8px 2px; min-height:2.7rem; font-weight:750; width:100%;
  box-shadow:0 4px 10px rgba(28,106,85,.14);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  border:0; color:white; filter:brightness(1.08); transform:translateY(-1px);
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border:1px solid #e1e9e0; border-radius:12px; overflow:hidden;
}
[data-testid="stAlert"] { border-radius:10px; }
[data-testid="stExpander"] {
  background:#fff; border:1px solid #e2e9e1; border-radius:12px;
}
.small-muted { color:#71877d; font-size:.82rem; }
hr { border-color:#dce7dd; }
</style>
""", unsafe_allow_html=True)

# ---------- Reproducible teaching datasets ----------
@st.cache_data
def build_data():
    rng = np.random.default_rng(42)

    # Heart-risk demonstration data with human-readable feature ranges.
    n = 1100
    age = rng.integers(25, 80, n)
    sex = rng.integers(0, 2, n)
    chest = rng.integers(0, 4, n)
    bp = np.clip(rng.normal(128, 18, n), 85, 200)
    chol = np.clip(rng.normal(225, 42, n), 120, 420)
    maxhr = np.clip(205 - .85*age + rng.normal(0, 13, n), 70, 205)
    exercise = rng.integers(0, 2, n)
    oldpeak = np.clip(rng.gamma(1.5, .65, n), 0, 6)
    logit = (-2.4 + .035*(age-45) + .012*(chol-200) + .025*(bp-120)
             - .025*(maxhr-140) + .55*exercise + .5*(chest==3) + .35*oldpeak)
    prob = 1/(1+np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    heart = pd.DataFrame({"Age":age,"Sex":sex,"ChestPainType":chest,
        "RestingBP":bp.round(0),"Cholesterol":chol.round(0),"MaxHR":maxhr.round(0),
        "ExerciseAngina":exercise,"Oldpeak":oldpeak.round(2),"HeartRisk":y})

    wine = load_wine(as_frame=True)
    wine_df = wine.frame.copy()

    # Synthetic churn data; no real customer records.
    n=1300
    tenure=rng.integers(1,73,n)
    calls=rng.poisson(2.2,n)
    fee=np.clip(rng.normal(650,180,n),150,1500)
    contract=rng.choice([1,2,3],n,p=[.55,.25,.20])
    late=rng.poisson(1.0,n)
    usage=np.clip(rng.normal(35,14,n),1,100)
    age_c=rng.integers(18,76,n)
    churn_prob=1/(1+np.exp(-(-1.8-.035*tenure+.42*calls+.0015*(fee-500)
                             -.45*(contract==3)+.28*late)))
    churn_y=(rng.random(n)<churn_prob).astype(int)
    churn=pd.DataFrame({"Tenure":tenure,"SupportCalls":calls,"MonthlyFee":fee.round(0),
        "ContractPlan":contract,"LatePayments":late,"UsageGB":usage.round(1),
        "Age":age_c,"Churn":churn_y})

    # Housing-style regression dataset (synthetic, not live market prices).
    n=1500
    income=np.clip(rng.lognormal(1.2,.45,n),.5,12)
    house_age=rng.integers(1,52,n)
    rooms=np.clip(rng.normal(5.5,1.5,n),2,12)
    bedrooms=np.clip(rooms*rng.uniform(.15,.28,n),1,5)
    population=rng.integers(100,6000,n)
    occupancy=np.clip(rng.normal(3,1,n),1,8)
    lat=rng.uniform(32,42,n); lon=rng.uniform(-124,-114,n)
    value=(55000+income*42000+rooms*8500-house_age*500
           -population*7-occupancy*1500+(lat-32)*1800+(lon+124)*650
           +rng.normal(0,25000,n))
    housing=pd.DataFrame({"MedInc":income.round(2),"HouseAge":house_age,
        "AveRooms":rooms.round(2),"AveBedrms":bedrooms.round(2),
        "Population":population,"AveOccup":occupancy.round(2),
        "Latitude":lat.round(2),"Longitude":lon.round(2),
        "HouseValue":np.clip(value,50000,650000).round(0)})

    # Student performance demonstration data.
    n=1200
    study=np.clip(rng.normal(3.8,1.7,n),0,10)
    attendance=np.clip(rng.normal(82,12,n),35,100)
    assignment=np.clip(rng.normal(72,16,n),0,100)
    quiz=np.clip(rng.normal(68,17,n),0,100)
    sleep=np.clip(rng.normal(6.8,1.2,n),3,10)
    participation=rng.integers(1,6,n)
    online=rng.integers(0,2,n)
    extra=rng.integers(0,2,n)
    score=study*4.2+attendance*.22+assignment*.18+quiz*.20+sleep*1.2+participation*1.5+online*2+extra*2
    pass_y=(score>48).astype(int)
    student=pd.DataFrame({"StudyHours":study.round(1),"Attendance":attendance.round(0),
        "AssignmentScore":assignment.round(0),"QuizScore":quiz.round(0),
        "SleepHours":sleep.round(1),"Participation":participation,
        "OnlineResources":online,"ExtraClass":extra,"Pass":pass_y})

    return heart, wine_df, churn, housing, student, wine.target_names

heart, wine_df, churn, housing, student, wine_names = build_data()
iris = load_iris(as_frame=True)

def hero(eyebrow, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)

def split_metrics(model, X, y, scale=False):
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model.fit(Xtr,ytr)
    pred=model.predict(Xte)
    return model, Xte, yte, pred

def classification_result(y, pred):
    a,b,c=st.columns(3)
    a.metric("Accuracy",f"{accuracy_score(y,pred)*100:.2f}%")
    b.metric("F1-score",f"{f1_score(y,pred,average='weighted')*100:.2f}%")
    c.metric("ข้อมูลทดสอบ",f"{len(y):,} รายการ")
    with st.expander("ดู Confusion Matrix และรายงานผล"):
        st.dataframe(pd.DataFrame(confusion_matrix(y,pred)),use_container_width=True)
        st.code(classification_report(y,pred,zero_division=0))

def show_data_tab(df, note):
    st.caption(note)
    st.dataframe(df.head(20),use_container_width=True)
    st.caption(f"จำนวนข้อมูลทั้งหมด {len(df):,} แถว • {len(df.columns)} คอลัมน์")

# ---------- Sidebar ----------
with st.sidebar:
    photo=ROOT/"assets"/"student_photo.jpg"
    if photo.exists():
        st.image(str(photo),width=112)
    st.markdown("**นายชุติพงศ์ ทองคำ**")
    st.caption("รหัสนักศึกษา 664245053")
    st.caption("โครงงาน Machine Learning")
    st.markdown("## 🌿 ML Studio")
    st.divider()
    pages=["หน้าหลัก","K-Nearest Neighbor","Decision Tree","Support Vector Machine",
           "K-Means","Regression","Random Forest","สรุปโครงงาน"]
    page=st.radio("เมนูหลัก",pages,label_visibility="collapsed")
    st.divider()
    st.markdown("---")
    

# ---------- Home ----------
if page=="หน้าหลัก":
    hero("THE DATA LAB · FINAL PROJECT","ทดลองคิด วิเคราะห์ และทำนายด้วย Machine Learning",
         "พื้นที่ทดลอง 6 เทคนิคการเรียนรู้ของเครื่อง ตั้งแต่การจำแนก การจัดกลุ่ม ไปจนถึงการพยากรณ์")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("เทคนิค ML","6")
    c2.metric("ชุดข้อมูล","6")
    c3.metric("โมเดล","Classification / Clustering")
    c4.metric("เครื่องมือ","Python + Streamlit")
    st.subheader("ภาพรวมบทเรียน")
    cards=[
      ("01","K-Nearest Neighbor","ทำนายความเสี่ยงโรคหัวใจจากข้อมูลตัวอย่าง","Classification"),
      ("02","Decision Tree","จำแนกชนิดไวน์จากคุณสมบัติทางเคมี","Classification"),
      ("03","Support Vector Machine","วิเคราะห์แนวโน้มยกเลิกบริการ","Classification"),
      ("04","K-Means","จัดกลุ่มดอก Iris โดยไม่ใช้ป้ายกำกับ","Clustering"),
      ("05","Regression","ประมาณราคาบ้านจากคุณลักษณะของบ้าน","Regression"),
      ("06","Random Forest","ทำนายแนวโน้มสอบผ่านจากพฤติกรรมการเรียน","Ensemble"),
    ]
    for start in range(0,6,3):
        cols=st.columns(3)
        for col,item in zip(cols,cards[start:start+3]):
            with col:
                with st.container(border=True):
                    st.caption(f"MODEL {item[0]} · {item[3]}")
                    st.markdown(f"### {item[1]}")
                    st.write(item[2])
    st.info("หมายเหตุ: ชุดข้อมูลโรคหัวใจ ลูกค้า ราคาบ้าน และผลการเรียนเป็นข้อมูลสังเคราะห์เพื่อการศึกษา ไม่ใช่ข้อมูลจริงหรือใช้ตัดสินใจในชีวิตจริง")

# ---------- KNN ----------
elif page=="K-Nearest Neighbor":
    hero("MODEL 01 · CLASSIFICATION","ทำนายความเสี่ยงโรคหัวใจด้วย K-Nearest Neighbor",
         "KNN เปรียบเทียบข้อมูลใหม่กับตัวอย่างที่ใกล้เคียงที่สุด โดยปรับสเกลข้อมูลด้วย StandardScaler")
    t1,t2,t3=st.tabs(["🧑🏻‍💻 ทดลองทำนาย","📊 ประสิทธิภาพ","📋 ข้อมูล"])
    with t1:
        with st.form("knn_form"):
            left,right=st.columns(2)
            with left:
                age=st.number_input("อายุ (Age)",18,100,54)
                sex=st.selectbox("เพศ (Sex)",["หญิง (0)","ชาย (1)"])
                chest=st.selectbox("อาการเจ็บหน้าอก (ChestPainType)",[0,1,2,3],index=1)
                bp=st.number_input("ความดันขณะพัก (RestingBP)",70,220,130)
                chol=st.number_input("คอเลสเตอรอล (Cholesterol)",100,600,223)
            with right:
                maxhr=st.number_input("อัตราการเต้นหัวใจสูงสุด (MaxHR)",60,220,138)
                exercise=st.selectbox("เจ็บหน้าอกจากออกกำลังกาย",["ไม่ใช่ (0)","ใช่ (1)"])
                oldpeak=st.number_input("ค่า Oldpeak",0.0,8.0,0.6,.1)
                k=st.slider("จำนวนเพื่อนบ้าน (K)",1,21,5)
            submit=st.form_submit_button("ทำนายความเสี่ยง",type="primary")
        if submit:
            X=heart.drop(columns="HeartRisk"); y=heart.HeartRisk
            model,_,_,_=split_metrics(make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=k)),X,y)
            sample=pd.DataFrame([[age,int("ชาย" in sex),chest,bp,chol,maxhr,int("ใช่" in exercise),oldpeak]],columns=X.columns)
            result=int(model.predict(sample)[0])
            if result: st.error("ผลการทำนาย: กลุ่มตัวอย่างมีแนวโน้มความเสี่ยงสูง")
            else: st.success("ผลการทำนาย: กลุ่มตัวอย่างมีแนวโน้มความเสี่ยงต่ำ")
            st.caption("ผลนี้เป็นเพียงการสาธิตจากข้อมูลสังเคราะห์ ไม่ใช่การวินิจฉัยทางการแพทย์")
    with t2:
        model,Xte,yte,pred=split_metrics(make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5)),heart.drop(columns="HeartRisk"),heart.HeartRisk)
        classification_result(yte,pred)
        st.bar_chart(pd.Series(model.named_steps["kneighborsclassifier"].n_neighbors,index=["K"],name="จำนวนเพื่อนบ้าน"))
    with t3: show_data_tab(heart,"ชุดข้อมูลจำลองสำหรับสาธิต KNN")

# ---------- Decision Tree ----------
elif page=="Decision Tree":
    hero("MODEL 02 · MULTICLASS CLASSIFICATION","จำแนกประเภทไวน์ด้วย Decision Tree",
         "ใช้โครงสร้างต้นไม้ตัดสินใจจำแนกไวน์ 3 กลุ่มจากคุณสมบัติทางเคมี 13 ตัวแปร")
    t1,t2,t3=st.tabs(["🧑🏻‍💻 ทดลองจำแนก","📊 ประสิทธิภาพ","📋 ข้อมูล"])
    X=wine_df.drop(columns="target"); y=wine_df["target"]
    with t1:
        with st.form("tree_form"):
            depth=st.slider("ความลึกสูงสุดของต้นไม้ (Max Depth)",1,12,4)
            idx=st.selectbox("เลือกตัวอย่างไวน์",range(len(wine_df)),format_func=lambda i:f"ตัวอย่างที่ {i+1}")
            submitted=st.form_submit_button("จำแนกประเภทไวน์",type="primary")
        if submitted:
            model,_,_,_=split_metrics(DecisionTreeClassifier(max_depth=depth,random_state=42),X,y)
            pred=int(model.predict(X.iloc[[idx]])[0])
            st.success(f"ผลการจำแนก: {wine_names[pred]}")
            st.caption("แสดงผลจากตัวอย่างในชุดข้อมูล Wine ที่มาพร้อม scikit-learn")
    with t2:
        model,Xte,yte,pred=split_metrics(DecisionTreeClassifier(max_depth=4,random_state=42),X,y)
        classification_result(yte,pred)
        fig,ax=plt.subplots(figsize=(13,5))
        plot_tree(model,feature_names=list(X.columns),class_names=list(wine_names),filled=True,rounded=True,max_depth=2,fontsize=7,ax=ax)
        st.pyplot(fig); plt.close(fig)
    with t3: show_data_tab(wine_df,"Wine Dataset จาก scikit-learn มีตัวแปรทางเคมี 13 ตัวและกลุ่มไวน์ 3 ประเภท")

# ---------- SVM ----------
elif page=="Support Vector Machine":
    hero("MODEL 03 · SUPPORT VECTOR MACHINE","ทำนายการยกเลิกบริการด้วย SVM",
         "วิเคราะห์พฤติกรรมการใช้บริการด้วย Support Vector Machine และประเมินแนวโน้มการยกเลิก")
    t1,t2,t3=st.tabs(["🧑🏻‍💻 ทดลองทำนาย","📊 ประสิทธิภาพ","📋 ข้อมูล"])
    X=churn.drop(columns="Churn"); y=churn.Churn
    with t1:
        with st.form("svm_form"):
            a,b=st.columns(2)
            with a:
                tenure=st.number_input("อายุสมาชิก (เดือน)",0,100,18)
                calls=st.number_input("จำนวนครั้งติดต่อ Call Center",0,30,4)
                fee=st.number_input("ค่าบริการรายเดือน (บาท)",100,3000,650)
                contract=st.selectbox("ประเภทสัญญา",[1,2,3],format_func=lambda x:{1:"รายเดือน",2:"1 ปี",3:"2 ปี"}[x])
            with b:
                late=st.number_input("จำนวนครั้งชำระล่าช้า",0,20,1)
                usage=st.number_input("ปริมาณใช้งาน (GB)",0.0,200.0,35.0)
                age=st.number_input("อายุลูกค้า",15,100,30)
            submitted=st.form_submit_button("ทำนายความเสี่ยงการยกเลิก",type="primary")
        if submitted:
            model,_,_,_=split_metrics(make_pipeline(StandardScaler(),SVC(kernel="rbf",C=1)),X,y)
            sample=pd.DataFrame([[tenure,calls,fee,contract,late,usage,age]],columns=X.columns)
            result=int(model.predict(sample)[0])
            (st.warning if result else st.success)("ผลการทำนาย: มีแนวโน้มยกเลิกบริการ" if result else "ผลการทำนาย: มีแนวโน้มใช้บริการต่อ")
    with t2:
        model,Xte,yte,pred=split_metrics(make_pipeline(StandardScaler(),SVC(kernel="rbf",C=1)),X,y)
        classification_result(yte,pred)
    with t3: show_data_tab(churn,"ข้อมูลลูกค้าจำลอง ไม่มีข้อมูลส่วนบุคคลจริง")

# ---------- KMeans ----------
elif page=="K-Means":
    hero("MODEL 04 · UNSUPERVISED LEARNING","จัดกลุ่มดอก Iris ด้วย K-Means",
         "K-Means แบ่งข้อมูลเป็นกลุ่มตามความคล้ายคลึง โดยไม่ใช้ป้ายกำกับในการฝึก")
    t1,t2,t3=st.tabs(["🧑🏻‍💻 ทดลองจัดกลุ่ม","📈 ภาพคลัสเตอร์","📋 รายละเอียด"])
    X=iris.data
    with t1:
        with st.form("kmeans_form"):
            a,b=st.columns(2)
            with a:
                sl=st.slider("Sepal length (cm)",4.0,8.0,5.8,.1)
                sw=st.slider("Sepal width (cm)",2.0,4.5,3.0,.1)
            with b:
                pl=st.slider("Petal length (cm)",1.0,7.0,4.3,.1)
                pw=st.slider("Petal width (cm)",.1,2.5,1.3,.1)
            k=st.slider("จำนวนกลุ่ม (K)",2,6,3)
            submitted=st.form_submit_button("จัดกลุ่มข้อมูล",type="primary")
        if submitted:
            scaler=StandardScaler(); Xs=scaler.fit_transform(X)
            model=KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs)
            label=int(model.predict(scaler.transform([[sl,sw,pl,pw]]))[0])
            st.success(f"ข้อมูลที่กรอกอยู่ใน Cluster {label+1}")
            st.caption("คลัสเตอร์เป็นกลุ่มเชิงคณิตศาสตร์ ไม่ได้แปลว่าเป็นชนิดพันธุ์ที่ยืนยันแล้ว")
    with t2:
        scaler=StandardScaler(); Xs=scaler.fit_transform(X)
        model=KMeans(n_clusters=3,random_state=42,n_init=10).fit(Xs)
        labels=model.labels_
        st.metric("Silhouette Score",f"{silhouette_score(Xs,labels):.3f}")
        fig,ax=plt.subplots(figsize=(9,5))
        scatter=ax.scatter(X.iloc[:,0],X.iloc[:,2],c=labels,alpha=.75)
        ax.set_xlabel("Sepal length (cm)"); ax.set_ylabel("Petal length (cm)")
        ax.set_title("K-Means clusters · Iris")
        st.pyplot(fig); plt.close(fig)
    with t3: show_data_tab(iris.frame,"Iris Dataset มีข้อมูลดอกไม้ 150 ตัวอย่าง 4 คุณลักษณะ")

# ---------- Regression ----------
elif page=="Regression":
    hero("MODEL 05 · REGRESSION","ประมาณราคาบ้านจากคุณลักษณะของบ้าน",
         "Random Forest Regressor เรียนรู้ความสัมพันธ์ระหว่างรายได้ อายุบ้าน จำนวนห้อง และทำเล")
    st.info("ชุดข้อมูลราคาบ้านนี้เป็นข้อมูลสังเคราะห์เพื่อการศึกษา ไม่ใช่ราคาซื้อขายจริง")
    t1,t2=st.tabs(["🧑🏻‍💻 ทดลองประมาณราคา","📋 ข้อมูลและประสิทธิภาพ"])
    X=housing.drop(columns="HouseValue"); y=housing.HouseValue
    with t1:
        with st.form("reg_form"):
            a,b=st.columns(2)
            with a:
                income=st.slider("รายได้เฉลี่ย (MedInc × $10,000)",.5,12.0,5.0,.1)
                age=st.slider("อายุบ้าน (ปี)",1,52,20)
                rooms=st.slider("จำนวนห้องเฉลี่ย",2.0,12.0,5.0,.1)
                beds=st.slider("จำนวนห้องนอนเฉลี่ย",1.0,5.0,1.5,.1)
            with b:
                pop=st.slider("จำนวนประชากร",100,6000,1500,50)
                occ=st.slider("จำนวนคนต่อครัวเรือน",1.0,8.0,3.0,.1)
                lat=st.slider("Latitude",32.0,42.0,35.0,.1)
                lon=st.slider("Longitude",-124.0,-114.0,-119.0,.1)
            submitted=st.form_submit_button("ทำนายราคาบ้าน",type="primary")
        if submitted:
            model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1).fit(X,y)
            sample=pd.DataFrame([[income,age,rooms,beds,pop,occ,lat,lon]],columns=X.columns)
            value=float(model.predict(sample)[0])
            st.metric("ราคาบ้านที่โมเดลประมาณ",f"${value:,.0f}")
    with t2:
        model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        model.fit(Xtr,ytr); pred=model.predict(Xte)
        a,b,c=st.columns(3)
        a.metric("MAE",f"${mean_absolute_error(yte,pred):,.0f}")
        b.metric("RMSE",f"${np.sqrt(mean_squared_error(yte,pred)):,.0f}")
        c.metric("R² Score",f"{r2_score(yte,pred):.3f}")
        st.dataframe(housing.head(20),use_container_width=True)

# ---------- Random Forest ----------
elif page=="Random Forest":
    hero("MODEL 06 · ENSEMBLE CLASSIFICATION","ทำนายแนวโน้มสอบผ่านด้วย Random Forest",
         "Ensemble Learning รวมผลจากต้นไม้ตัดสินใจหลายต้น เพื่อประเมินแนวโน้มผลการเรียน")
    st.warning("ข้อมูลนักศึกษาเป็นข้อมูลสังเคราะห์ ไม่ใช่ผลการเรียนจริง")
    t1,t2,t3=st.tabs(["🧑🏻‍💻 ทดลองทำนาย","📊 ประสิทธิภาพ","📋 ข้อมูล"])
    X=student.drop(columns="Pass"); y=student.Pass
    with t1:
        with st.form("rf_form"):
            a,b=st.columns(2)
            with a:
                study=st.number_input("ชั่วโมงอ่านหนังสือต่อวัน",0.0,16.0,4.0,.5)
                attendance=st.slider("เปอร์เซ็นต์การเข้าเรียน",0,100,85)
                assignment=st.slider("คะแนนงาน/การบ้าน",0,100,75)
                quiz=st.slider("คะแนนแบบทดสอบ",0,100,72)
            with b:
                sleep=st.slider("ชั่วโมงนอนเฉลี่ย",0.0,12.0,7.0,.5)
                participation=st.selectbox("การมีส่วนร่วมในชั้นเรียน",[1,2,3,4,5])
                online=st.selectbox("ใช้แหล่งเรียนรู้ออนไลน์",["มี","ไม่มี"])
                extra=st.selectbox("เข้าร่วมการสอนเสริม",["เข้าร่วม","ไม่เข้าร่วม"])
            submitted=st.form_submit_button("ทำนายผลการเรียน",type="primary")
        if submitted:
            model=RandomForestClassifier(n_estimators=160,max_depth=8,random_state=42,n_jobs=-1).fit(X,y)
            sample=pd.DataFrame([[study,attendance,assignment,quiz,sleep,participation,
                                  int(online=="มี"),int(extra=="เข้าร่วม")]],columns=X.columns)
            result=int(model.predict(sample)[0])
            if result: st.success("ผลการทำนาย: มีแนวโน้มสอบผ่าน")
            else: st.error("ผลการทำนาย: ควรทบทวนบทเรียนเพิ่มเติม")
    with t2:
        model,Xte,yte,pred=split_metrics(RandomForestClassifier(n_estimators=160,max_depth=8,random_state=42,n_jobs=-1),X,y)
        classification_result(yte,pred)
        st.markdown("**ความสำคัญของคุณลักษณะ (Feature Importance)**")
        st.bar_chart(pd.Series(model.feature_importances_,index=X.columns).sort_values())
    with t3: show_data_tab(student,"ชุดข้อมูลจำลองพฤติกรรมการเรียนและผลการสอบ")

# ---------- Summary ----------
elif page=="สรุปโครงงาน":
    hero("PROJECT SUMMARY","สรุปผลและแหล่งที่มา","ภาพรวมเทคนิค Machine Learning ทั้ง 6 เทคนิค ข้อจำกัด และข้อมูลอ้างอิง")
    summary=pd.DataFrame([
      ["KNN","จำแนกความเสี่ยงหัวใจ","Classification","Accuracy / F1"],
      ["Decision Tree","จำแนกประเภทไวน์","Multiclass","Accuracy / F1"],
      ["SVM","แนวโน้มยกเลิกบริการ","Classification","Accuracy / F1"],
      ["K-Means","จัดกลุ่ม Iris","Clustering","Silhouette Score"],
      ["Regression","ประมาณราคาบ้าน","Regression","MAE / RMSE / R²"],
      ["Random Forest","แนวโน้มสอบผ่าน","Ensemble","Accuracy / F1"],
    ],columns=["เทคนิค","หัวข้อ","ประเภท","ตัวชี้วัด"])
    st.dataframe(summary,use_container_width=True,hide_index=True)
    st.subheader("ข้อจำกัดของโครงงาน")
    st.markdown("- ผลลัพธ์ขึ้นอยู่กับคุณภาพและลักษณะของชุดข้อมูลที่ใช้ฝึก\n- ข้อมูลสังเคราะห์ไม่สามารถใช้แทนข้อมูลจริงหรือการตัดสินใจในชีวิตจริง\n- ค่าประสิทธิภาพเป็นผลจากการแบ่งข้อมูลฝึกและทดสอบตาม random_state ที่กำหนด")
    st.subheader("แหล่งข้อมูลและเครื่องมือ")
    st.markdown("- [Scikit-learn documentation](https://scikit-learn.org/stable/)\n- [Streamlit documentation](https://docs.streamlit.io/)\n- [Pandas documentation](https://pandas.pydata.org/docs/)\n- [Matplotlib documentation](https://matplotlib.org/stable/)")
    st.caption("จัดทำโดย นายชุติพงศ์ ทองคำ • รหัสนักศึกษา 664245053")
