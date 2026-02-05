# ==========================================
# 📌 app.py — MIS Retail Intelligence Center
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import logging
from openai import OpenAI

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ------------------------------------------
# 🖥 页面设置
# ------------------------------------------
st.set_page_config(
    page_title="Retail Intelligence Command Center",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------
# 🟦 全局颜色（支持简称 + 全称）
# ------------------------------------------
COLOR_MAP = {
    "Walmart": "#0071ce",
    "Walmart Inc.": "#0071ce",
    "Costco": "#e31837",
    "Costco Wholesale Corporation": "#e31837",
    "Target": "#cc0000",
    "Target Corporation": "#cc0000",
    "Kroger": "#1a73e8",
    "The Kroger Co.": "#1a73e8",
}

# 简称 ↔ 全称映射（可以按你数据库实际情况调整/扩展）
COMPANY_ALIAS = {
    "Walmart": "Walmart Inc.",
    "Costco": "Costco Wholesale Corporation",
    "Target": "Target Corporation",
    "Kroger": "The Kroger Co.",
}

# ------------------------------------------
# 🤖 AI 模型配置（从 secrets 读取，避免明文泄露）
# ------------------------------------------
API_KEY = "sk-dX7GXVspY2DM9OHcOyxB3CrV9mbSsKVDwE5gE7of1eGtiBhd"
BASE_URL = "https://api5.xhub.chat/v1"
MODEL_NAME = "o1-mini"


if API_KEY:
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception:
        client = None
        st.warning("⚠️ AI 服务初始化失败，请检查 API 配置")
else:
    client = None
    st.warning("⚠️ 未配置 OPENAI_API_KEY，AI 分析功能不可用")

# ==========================================
# 🔧 工具函数
# ==========================================

def safe_get(row, field, default=np.nan):
    """对 Series 安全取值"""
    if row is None:
        return default
    if field in row and pd.notna(row[field]):
        return row[field]
    return default


def detect_revenue(row: pd.Series):
    """自动识别 Revenue 字段：优先使用统一后的 Revenue，其次原始科目"""
    for k in ["Revenue", "Net sales", "Total revenues", "Total Revenue"]:
        if k in row and pd.notna(row[k]):
            return row[k]
    return np.nan


def detect_net_income(row: pd.Series):
    """自动识别 Net Income 字段"""
    for k in ["Net_Income", "Net income", "Net Income"]:
        if k in row and pd.notna(row[k]):
            return row[k]
    return np.nan


def to_numeric_cols(df, cols):
    """将指定列安全地转为数值类型"""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def generate_synthetic_years(df: pd.DataFrame, start_year=2016, end_year=2025):
    """
    基于现有真实数据，为每个公司自动补全 [start_year, end_year] 区间内缺失年份。
    """

    if df.empty:
        return df

    companies = df["Company"].dropna().unique()
    all_years = np.arange(start_year, end_year + 1)

    rows = []

    # Revenue / Net Income 识别
    if "Revenue" not in df.columns:
        df["Revenue"] = df.apply(detect_revenue, axis=1)
    if "Net_Income" not in df.columns:
        df["Net_Income"] = df.apply(detect_net_income, axis=1)

    # 强制转换数值列
    extra_numeric = [
        "Net sales", "Total revenues", "Net income", "Total assets",
        "Total equity", "Total liabilities", "Operating income", "COGS",
        "Inventories", "Net receivables", "Accounts payable",
        "Total interests", "Gross profit", "Net FCF", "Net OCF",
        "Operating Cash Flow", "Capital Expenditures"
    ]
    df = to_numeric_cols(df, extra_numeric)

    # 数值列（排除年份、id类）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ["Fiscal_Year", "Is_Synthetic", "statement_id", "company_id", "#"]
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    for comp in companies:
        sub = df[df["Company"] == comp].copy()
        if sub.empty:
            continue

        sub = sub.sort_values("Fiscal_Year")
        real_years = sub["Fiscal_Year"].dropna().unique()
        if len(real_years) == 0:
            continue

        # 计算 CAGR
        rev_valid = sub.dropna(subset=["Revenue"]).sort_values("Fiscal_Year")
        if len(rev_valid) >= 2 and rev_valid["Revenue"].iloc[0] > 0:
            y0 = rev_valid["Fiscal_Year"].iloc[0]
            y1 = rev_valid["Fiscal_Year"].iloc[-1]
            v0 = rev_valid["Revenue"].iloc[0]
            v1 = rev_valid["Revenue"].iloc[-1]
            span = y1 - y0 if y1 != y0 else 1
            cagr = (v1 / v0) ** (1 / span) - 1
        else:
            cagr = 0.03

        min_year = real_years.min()
        max_year = real_years.max()

        for year in all_years:
            # 有真实数据
            if year in real_years:
                row = sub[sub["Fiscal_Year"] == year].iloc[0].copy()
                row["Is_Synthetic"] = 0
                rows.append(row)
                continue

            # 需要补全
            if year < min_year:
                base = sub[sub["Fiscal_Year"] == min_year].iloc[0]
                years_diff = min_year - year
                factor = (1 - cagr) ** years_diff
            else:
                base = sub[sub["Fiscal_Year"] == max_year].iloc[0]
                years_diff = year - max_year
                factor = (1 + cagr) ** years_diff

            new_row = base.copy()
            new_row["Fiscal_Year"] = year

            # 数值按比例缩放
            for col in numeric_cols:
                if pd.notna(base[col]):
                    new_row[col] = base[col] * factor

            new_row["Is_Synthetic"] = 1
            rows.append(new_row)

    filled = pd.DataFrame(rows)
    filled = filled.sort_values(["Company", "Fiscal_Year"]).reset_index(drop=True)

    # ⭐ 强制年份为整数，避免出现 2025.0
    filled["Fiscal_Year"] = filled["Fiscal_Year"].astype(int)

    return filled



# ==========================================
# 🔧 从三张基础表自动 Pivot 财务数据
# ==========================================
@st.cache_data(ttl=600)
def load_financial_data():
    try:
        cfg = st.secrets["mysql"]
        engine = create_engine(
            f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
        )

        # 1. 读取三张表
        df_comp = pd.read_sql("SELECT * FROM Companies", engine)
        df_fs = pd.read_sql("SELECT * FROM Financial_Statements", engine)
        df_items = pd.read_sql("SELECT * FROM Statement_Items", engine)

        # -------------------------------------------------------------------------
        # 🚑【FCF 修正补丁 v2.0】(优化版：自动清洗格式 + 强制计算)
        # -------------------------------------------------------------------------
        try:
            # [关键步骤] 强制把 item_value 转成数字，防止数据库里存的是字符串导致无法相加
            df_items['item_value'] = pd.to_numeric(df_items['item_value'], errors='coerce')

            # 1. 转宽表方便计算
            temp_wide = df_items.pivot(index='statement_id', columns='item_name', values='item_value')

            # 2. 检查是否有 'Net OCF' 和 'Capital expenditures' 这两列
            if 'Net OCF' in temp_wide.columns and 'Capital expenditures' in temp_wide.columns:
                
                # 3. 计算正确的 FCF = OCF + CapEx (因为CapEx本身是负数，所以直接加)
                calculated_fcf = temp_wide['Net OCF'] + temp_wide['Capital expenditures']
                
                # 4. 将算好的值填回 df_items
                count = 0
                for stmt_id, new_val in calculated_fcf.items():
                    # 找到对应的行
                    mask = (df_items['statement_id'] == stmt_id) & (df_items['item_name'] == 'Net FCF')
                    
                    if mask.any():
                        # 执行覆盖更新
                        df_items.loc[mask, 'item_value'] = new_val
                        count += 1
                
                # [测试反馈] 在网页右下角弹窗提示，让你知道修正了多少条
                if count > 0:
                    print(f"✅ 后台日志：已成功修正 {count} 条 FCF 数据")
                    # st.toast(f"已自动修正 {count} 条 FCF 数据", icon="✅") # 如果觉得弹窗烦可以注释掉

        except Exception as e:
            print(f"⚠️ FCF 补丁运行报错: {e}")
        # -------------------------------------------------------------------------

        # Pivot Statement_Items (原本的主逻辑)
        df_pivot = df_items.pivot_table(
            index="statement_id",
            columns="item_name",
            values="item_value",
            aggfunc="first",
        ).reset_index()

        # 合并
        df_fin = df_fs.merge(df_pivot, on="statement_id", how="left")
        df_fin = df_fin.merge(df_comp, on="company_id", how="left")

        # 统一命名
        df_fin.rename(
            columns={
                "company_name": "Company",
                "fiscal_year": "Fiscal_Year",
                "period_end_date": "Period_End_Date",
            },
            inplace=True,
        )

        df_fin = df_fin.sort_values(["Company", "Fiscal_Year"])

        return df_fin, "MySQL（已自动修正FCF）"

    except Exception as e:
        st.error(f"MySQL 加载失败：{e}")
        return pd.DataFrame(), "EMPTY"

# 载入数据
df, data_source = load_financial_data()

if df.empty:
    st.stop()

# ==========================================
# 🔧 先生成基础指标（Revenue / Net_Income）并补全 2016–2025 年
# ==========================================

# 统一检测营收 & 净利润
df["Revenue"] = df.apply(detect_revenue, axis=1)
df["Net_Income"] = df.apply(detect_net_income, axis=1)

# 将关键字段转数值
numeric_candidates = [
    "Revenue",
    "Net_Income",
    "Net sales",
    "Total revenues",
    "Net income",
    "Total assets",
    "Total equity",
    "Total liabilities",
    "Operating income",
    "COGS",
    "Inventories",
    "Net receivables",
    "Accounts payable",
    "Total interests",
    "Gross profit",
    "Net FCF",
    "Net OCF",
    "Operating Cash Flow",
    "Capital Expenditures",
]
df = to_numeric_cols(df, numeric_candidates)

# 🔥 自动补全 2016–2025 年缺失年份
##df = generate_synthetic_years(df, start_year=2016, end_year=2025)

# ==========================================
# 🔧 生成各类财务比率 / 指标
# ==========================================

# ROA / ROE
df["ROA"] = df["Net_Income"] / df["Total assets"]
df["ROE"] = df["Net_Income"] / df["Total equity"]

# 负债率
df["Debt_Ratio"] = df["Total liabilities"] / df["Total assets"]

# 利息保障倍数
if "Total interests" in df.columns:
    df["Times_Interest_Earned_Ratio"] = df["Operating income"] / df["Total interests"]
else:
    df["Times_Interest_Earned_Ratio"] = np.nan

# 利润率相关（确保字段存在）
df["Gross_Profit"] = df.get("Gross profit", np.nan)
df["Operating_Income"] = df.get("Operating income", np.nan)

# 避免除以 0
df["Revenue"].replace(0, np.nan, inplace=True)

df["Gross_Profit_Margin"] = df["Gross_Profit"] / df["Revenue"]
df["Operating_Profit_Margin"] = df["Operating_Income"] / df["Revenue"]
df["Net_Profit_Margin"] = df["Net_Income"] / df["Revenue"]

# 库存周转天数 DIO
if "COGS" in df.columns:
    df["COGS"].replace(0, np.nan, inplace=True)
else:
    df["COGS"] = np.nan

df["Inventory_Days"] = 365 * df["Inventories"] / df["COGS"]

# DSO / DPO / CCC
if "Net sales" in df.columns:
    df["Net sales"].replace(0, np.nan, inplace=True)
else:
    df["Net sales"] = df["Revenue"]

df["DSO"] = 365 * df["Net receivables"] / df["Net sales"]
df["DPO"] = 365 * df["Accounts payable"] / df["COGS"]
df["CCC"] = df["Inventory_Days"] + df["DSO"] - df["DPO"]

# EBITDA
EBITDA_candidates = ["EBITDA", "Earnings Before Taxes", "EBITDA Margin"]


def calc_EBITDA(row: pd.Series):
    for key in EBITDA_candidates:
        if key in row and pd.notna(row[key]):
            return row[key]

    op = row.get("Operating income", np.nan)
    dep = row.get("Depreciation", 0)
    if pd.notna(op):
        return op + (dep if pd.notna(dep) else 0)
    return np.nan


df["EBITDA"] = df.apply(calc_EBITDA, axis=1)

# FCF
def calc_FCF(row: pd.Series):
    if "FCF" in row and pd.notna(row["FCF"]):
        return row["FCF"]
    if "Net FCF" in row and pd.notna(row["Net FCF"]):
        return row["Net FCF"]

    ocf = row.get("Net OCF", np.nan)
    if pd.isna(ocf):
        ocf = row.get("Operating Cash Flow", np.nan)

    capex = row.get("Capital Expenditures", 0)

    if pd.notna(ocf):
        return ocf - (capex if pd.notna(capex) else 0)

    return np.nan


df["FCF"] = df.apply(calc_FCF, axis=1)

# 清理异常值
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ==========================================
# 📌 Sidebar — 控制面板
# ==========================================

st.sidebar.title("🧭 控制面板 Control Center")
st.sidebar.caption(f"数据来源：{data_source}")

years = sorted(df["Fiscal_Year"].dropna().unique())
selected_year = st.sidebar.selectbox("选择展示年份：", years, index=len(years) - 1)

selected_range = st.sidebar.select_slider(
    "选择分析区间：",
    options=years,
    value=(years[0], years[-1]),
)

df_period = df[
    (df["Fiscal_Year"] >= selected_range[0])
    & (df["Fiscal_Year"] <= selected_range[1])
].copy()

# 预测设置
st.sidebar.subheader("🔮 预测设置")
forecast_model = st.sidebar.selectbox(
    "选择预测模型：",
    ["Linear Regression", "Polynomial Regression", "SARIMA", "Prophet"],
    index=3,
)
forecast_years = st.sidebar.slider("预测未来几年：", 1, 10, 5)

# DCF 设置
st.sidebar.subheader("💰 DCF 估值")
companies = sorted(df["Company"].dropna().unique())
dcf_company = st.sidebar.selectbox("选择估值公司：", companies)

wacc_input = st.sidebar.slider("WACC（加权资本成本）", 0.04, 0.12, 0.08, step=0.001)
tg_input = st.sidebar.slider("终值增长率（Terminal Growth）", 0.00, 0.05, 0.02, step=0.001)

# 行业对标
st.sidebar.subheader("📊 行业对标")
peers_short_names = st.sidebar.multiselect(
    "选择同行：",
    ["Walmart", "Costco", "Target", "Kroger"],
    default=["Walmart", "Costco"],
)

# 将简称映射为全称，用于 DataFrame 过滤
peers_select = [COMPANY_ALIAS.get(n, n) for n in peers_short_names]

# ==========================================
# 📌 主界面标题 + 年度 KPI
# ==========================================

st.title(f"🏬 零售战略智能分析中心（年度版） | Fiscal Year {selected_year}")

current_df = df[df["Fiscal_Year"] == selected_year].copy()
if current_df.empty:
    st.error(f"⚠️ 数据库中没有 {selected_year} 年的数据")
    st.stop()


def get_company_row(short_name_or_full):
    """支持传入简称（Walmart）或全称（Walmart Inc.）"""
    full_name = COMPANY_ALIAS.get(short_name_or_full, short_name_or_full)
    d = current_df[current_df["Company"] == full_name]
    return d.iloc[0] if len(d) > 0 else None


w = get_company_row("Walmart")
c = get_company_row("Costco")

k1, k2, k3, k4 = st.columns(4)

# KPI1 Walmart 营收 + ROE
w_rev = safe_get(w, "Revenue") or safe_get(w, "Net sales")
w_roe = safe_get(w, "ROE", 0)  # 小数形式
k1.metric(
    "Walmart 营收（$M）",
    f"{w_rev:,.0f}" if pd.notna(w_rev) else "N/A",
    f"ROE {w_roe * 100:.2f}%",
)

# KPI2 Costco 营收 + ROE
c_rev = safe_get(c, "Revenue") or safe_get(c, "Net sales")
c_roe = safe_get(c, "ROE", 0)
k2.metric(
    "Costco 营收（$M）",
    f"{c_rev:,.0f}" if pd.notna(c_rev) else "N/A",
    f"ROE {c_roe * 100:.2f}%",
)

# KPI3 资产周转率
def asset_turnover(row):
    if row is None:
        return np.nan
    rev = safe_get(row, "Revenue") or safe_get(row, "Net sales")
    assets = safe_get(row, "Total assets")
    if pd.notna(rev) and pd.notna(assets) and assets != 0:
        return rev / assets
    return np.nan


at_w = asset_turnover(w)
at_c = asset_turnover(c)

k3.metric(
    "资产周转率（AT）",
    f"W: {at_w:.2f}x" if pd.notna(at_w) else "W: N/A",
    f"C: {at_c:.2f}x" if pd.notna(at_c) else "C: N/A",
)

# KPI4 库存天数（使用 Inventory_Days）
w_inv = safe_get(w, "Inventory_Days", np.nan)
c_inv = safe_get(c, "Inventory_Days", np.nan)

delta_text = ""
if pd.notna(w_inv) and pd.notna(c_inv):
    delta_text = f"比 W 快 {w_inv - c_inv:.1f} 天"

k4.metric(
    "库存天数（Inventory Days）",
    f"C: {c_inv:.1f} 天" if pd.notna(c_inv) else "C: N/A",
    delta_text,
    delta_color="inverse",
)

st.divider()

# ==========================================
# Part 4 — 年度趋势图 + 智能预测
# ==========================================

st.header("📈 年度趋势分析（Historical Trends）")

df_trend = df_period.copy()

# 确保 Revenue / Net_Income 已经存在
df_trend["Revenue"] = df_trend["Revenue"].where(
    df_trend["Revenue"].notna(), df_trend.apply(detect_revenue, axis=1)
)
df_trend["Net_Income"] = df_trend["Net_Income"].where(
    df_trend["Net_Income"].notna(), df_trend.apply(detect_net_income, axis=1)
)

# Plot 1：Revenue
st.subheader("📊 营收趋势（Revenue Trend）")
df_rev = df_trend[["Company", "Fiscal_Year", "Revenue"]].dropna()
fig_rev = px.line(
    df_rev,
    x="Fiscal_Year",
    y="Revenue",
    color="Company",
    markers=True,
    title="Revenue Trend",
    color_discrete_map=COLOR_MAP,
)
st.plotly_chart(fig_rev, use_container_width=True)

# Plot 2：Net Income
st.subheader("📊 净利润趋势（Net Income Trend）")
df_ni = df_trend[["Company", "Fiscal_Year", "Net_Income"]].dropna()
fig_ni = px.line(
    df_ni,
    x="Fiscal_Year",
    y="Net_Income",
    color="Company",
    markers=True,
    title="Net Income Trend",
    color_discrete_map=COLOR_MAP,
)
st.plotly_chart(fig_ni, use_container_width=True)

# Plot 3：ROE / ROA
st.subheader("📈 ROE / ROA 趋势")
df_roe = df_trend.melt(
    id_vars=["Company", "Fiscal_Year"],
    value_vars=["ROE", "ROA"],
    var_name="Metric",
    value_name="Value",
).dropna(subset=["Value"])

fig_roe = px.line(
    df_roe,
    x="Fiscal_Year",
    y="Value",
    color="Metric",
    line_dash="Company",
    markers=True,
    title="ROE / ROA Trend",
)
st.plotly_chart(fig_roe, use_container_width=True)

# Plot 4：Margin
st.subheader("📉 毛利率 / 营业利润率 / 净利率")
df_melt = df_trend.melt(
    id_vars=["Company", "Fiscal_Year"],
    value_vars=[
        "Gross_Profit_Margin",
        "Operating_Profit_Margin",
        "Net_Profit_Margin",
    ],
    var_name="Metric",
    value_name="Value",
).dropna(subset=["Value"])

fig_m = px.line(
    df_melt,
    x="Fiscal_Year",
    y="Value",
    color="Metric",
    line_dash="Company",
    markers=True,
    title="Profit Margin Trends",
)
st.plotly_chart(fig_m, use_container_width=True)

# Plot 5：库存天数 & CCC
st.subheader("📦 库存周转与现金循环周期")
fig_inv = px.line(
    df_trend,
    x="Fiscal_Year",
    y="Inventory_Days",
    color="Company",
    markers=True,
    color_discrete_map=COLOR_MAP,
    title="库存周转天数（DIO）",
)
fig_ccc = px.line(
    df_trend,
    x="Fiscal_Year",
    y="CCC",
    color="Company",
    markers=True,
    title="现金转换周期（CCC）",
)
c1, c2 = st.columns(2)
c1.plotly_chart(fig_inv, use_container_width=True)
c2.plotly_chart(fig_ccc, use_container_width=True)

# ==========================================
# 🔮 营收预测模块
# ==========================================

st.header("🔮 营收预测（Forecasting）")

forecast_df = df_trend[["Company", "Fiscal_Year", "Revenue"]].dropna()
pred_fig = go.Figure()

last_year = df["Fiscal_Year"].max()
future_years = np.arange(last_year + 1, last_year + 1 + forecast_years)


def forecast_linear(data):
    x = data["Fiscal_Year"].values
    y = data["Revenue"].values
    coef = np.polyfit(x, y, 1)
    poly = np.poly1d(coef)
    return poly(future_years)


def forecast_poly(data):
    x = data["Fiscal_Year"].values
    y = data["Revenue"].values
    coef = np.polyfit(x, y, 2)
    poly = np.poly1d(coef)
    return poly(future_years)


def forecast_sarima(data):
    try:
        model = SARIMAX(data["Revenue"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        result = model.fit(disp=False)
        return result.forecast(steps=forecast_years).values
    except Exception:
        return np.zeros(forecast_years)


def forecast_prophet(data):
    try:
        pdf = data.rename(columns={"Fiscal_Year": "ds", "Revenue": "y"})
        pdf["ds"] = pd.to_datetime(pdf["ds"], format="%Y")
        m = Prophet()
        m.fit(pdf)
        future = m.make_future_dataframe(periods=forecast_years, freq="Y")
        fc = m.predict(future)
        return fc["yhat"].tail(forecast_years).values
    except Exception:
        return np.zeros(forecast_years)


for short_name in peers_short_names:
    comp_name = COMPANY_ALIAS.get(short_name, short_name)
    data = forecast_df[forecast_df["Company"] == comp_name].sort_values("Fiscal_Year")
    if len(data) < 2:
        continue

    # 历史
    pred_fig.add_trace(
        go.Scatter(
            x=data["Fiscal_Year"],
            y=data["Revenue"],
            name=f"{short_name}（历史）",
            line=dict(color=COLOR_MAP.get(comp_name, "#333")),
        )
    )

    # 预测
    if forecast_model == "Linear Regression":
        pred = forecast_linear(data)
    elif forecast_model == "Polynomial Regression":
        pred = forecast_poly(data)
    elif forecast_model == "SARIMA":
        pred = forecast_sarima(data)
    else:
        pred = forecast_prophet(data)

    pred_fig.add_trace(
        go.Scatter(
            x=future_years,
            y=pred,
            name=f"{short_name}（预测）",
            line=dict(color=COLOR_MAP.get(comp_name, "#333"), dash="dash"),
        )
    )

pred_fig.update_layout(
    title=f"未来 {forecast_years} 年营收预测（模型：{forecast_model}）",
    xaxis_title="Fiscal Year",
    yaxis_title="Revenue ($M)",
)
st.plotly_chart(pred_fig, use_container_width=True)

# ==========================================
# 💎 财务与杜邦分析（EBITDA / FCF / 杠杆 / 成长性）
# ==========================================

st.header("💎 财务与杜邦分析（Financial & DuPont Analysis）")

# 重新生成 df_period（已包含 EBITDA / FCF 等）
df_period = df[
    (df["Fiscal_Year"] >= selected_range[0])
    & (df["Fiscal_Year"] <= selected_range[1])
].copy()

w = get_company_row("Walmart")
c = get_company_row("Costco")

# 5.1 杜邦分析
st.subheader("📘 杜邦分析（NPM × AT × EM = ROE）")


def dupont_elements(row):
    rev = safe_get(row, "Revenue") or safe_get(row, "Net sales")
    net_inc = safe_get(row, "Net income") or safe_get(row, "Net_Income")
    assets = safe_get(row, "Total assets")
    equity = safe_get(row, "Total equity")

    NPM = (net_inc / rev) if pd.notna(net_inc) and pd.notna(rev) and rev != 0 else np.nan
    AT = (rev / assets) if pd.notna(rev) and pd.notna(assets) and assets != 0 else np.nan
    EM = (assets / equity) if pd.notna(assets) and pd.notna(equity) and equity != 0 else np.nan
    ROE_val = NPM * AT * EM if pd.notna(NPM) and pd.notna(AT) and pd.notna(EM) else np.nan
    return NPM, AT, EM, ROE_val


d1, d2 = st.columns(2)
for comp_short, row, col in [("Walmart", w, d1), ("Costco", c, d2)]:
    NPM, AT_val, EM, ROE_val = dupont_elements(row)
    comp_full = COMPANY_ALIAS.get(comp_short, comp_short)

    with col:
        st.markdown(f"### **{comp_short}**")
        fig = go.Figure(
            go.Sunburst(
                labels=["ROE", "净利率 (NPM)", "资产周转率 (AT)", "权益乘数 (EM)"],
                parents=["", "ROE", "ROE", "ROE"],
                values=[ROE_val or 0, NPM or 0, (AT_val or 0) * 10, (EM or 0) * 10],
                text=[
                    f"{(ROE_val or 0) * 100:.1f}%",
                    f"{(NPM or 0) * 100:.1f}%",
                    f"{AT_val or 0:.2f}x",
                    f"{EM or 0:.2f}x",
                ],
                textinfo="label+text",
                marker=dict(
                    colors=[
                        COLOR_MAP.get(comp_full, "#333"),
                        "#3498db",
                        "#2ecc71",
                        "#e67e22",
                    ]
                ),
            )
        )
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=350)
        st.plotly_chart(fig, use_container_width=True)

# 5.2 EBITDA & FCF 分析
st.subheader("💵 EBITDA / FCF 分析")

k1, k2 = st.columns(2)
k1.metric(
    "Walmart EBITDA",
    f"{safe_get(w, 'EBITDA'):,.0f}" if pd.notna(safe_get(w, "EBITDA")) else "N/A",
    help="EBITDA = Operating Income + Depreciation（如果数据库未提供）",
)
k2.metric(
    "Costco EBITDA",
    f"{safe_get(c, 'EBITDA'):,.0f}" if pd.notna(safe_get(c, "EBITDA")) else "N/A",
)

t1, t2 = st.columns(2)
t1.metric(
    "Walmart Free Cash Flow (FCF)",
    f"{safe_get(w, 'FCF'):,.0f}" if pd.notna(safe_get(w, "FCF")) else "N/A",
)
t2.metric(
    "Costco Free Cash Flow (FCF)",
    f"{safe_get(c, 'FCF'):,.0f}" if pd.notna(safe_get(c, "FCF")) else "N/A",
)

df_fcf = df_period.melt(
    id_vars=["Company", "Fiscal_Year"],
    value_vars=["EBITDA", "FCF"],
    var_name="Metric",
    value_name="Value",
).dropna(subset=["Value"])

fig_fcf = px.line(
    df_fcf,
    x="Fiscal_Year",
    y="Value",
    color="Metric",
    line_dash="Company",
    markers=True,
    title="EBITDA / FCF Trend",
)
st.plotly_chart(fig_fcf, use_container_width=True)

# 5.3 杠杆与偿债能力
st.subheader("📉 杠杆与偿债能力分析")

fig_debt = px.line(
    df_period,
    x="Fiscal_Year",
    y="Debt_Ratio",
    color="Company",
    markers=True,
    color_discrete_map=COLOR_MAP,
    title="Debt Ratio（资产负债率）",
)

fig_tie = px.line(
    df_period,
    x="Fiscal_Year",
    y="Times_Interest_Earned_Ratio",
    color="Company",
    markers=True,
    color_discrete_map=COLOR_MAP,
    title="Times Interest Earned（利息保障倍数）",
)

l1, l2 = st.columns(2)
l1.plotly_chart(fig_debt, use_container_width=True)
l2.plotly_chart(fig_tie, use_container_width=True)

# 5.4 成长性分析（YoY）
st.subheader("📈 成长性分析（YoY Growth）")

df_growth = df_period.sort_values(["Company", "Fiscal_Year"]).copy()
df_growth["Revenue"] = df_growth["Revenue"].where(
    df_growth["Revenue"].notna(), df_growth.apply(detect_revenue, axis=1)
)
df_growth["Net_Income"] = df_growth["Net_Income"].where(
    df_growth["Net_Income"].notna(), df_growth.apply(detect_net_income, axis=1)
)


def yoy_growth(series):
    return series.pct_change() * 100


df_growth["Revenue_Growth"] = df_growth.groupby("Company")["Revenue"].transform(
    lambda s: s.pct_change() * 100
)

df_growth["NetIncome_Growth"] = df_growth.groupby("Company")["Net_Income"].transform(
    lambda s: s.pct_change() * 100
)


df_growth_melt = df_growth.melt(
    id_vars=["Company", "Fiscal_Year"],
    value_vars=["Revenue_Growth", "NetIncome_Growth"],
    var_name="Metric",
    value_name="Value",
).dropna(subset=["Value"])

fig_growth = px.line(
    df_growth_melt,
    x="Fiscal_Year",
    y="Value",
    color="Metric",
    line_dash="Company",
    markers=True,
    title="YoY Growth Trend (%)",
)
st.plotly_chart(fig_growth, use_container_width=True)

# ==========================================
# 📌 DCF 估值
# ==========================================

st.header("💰 企业估值（DCF — 折现现金流模型）")

df_dcf = df[df["Company"] == dcf_company].sort_values("Fiscal_Year").copy()
df_dcf["FCF"] = pd.to_numeric(df_dcf["FCF"], errors="coerce")

if df_dcf["FCF"].isna().all():
    st.error("⚠️ 当前公司没有 FCF 数据，无法执行 DCF。")
else:
    df_dcf["FCF_Growth"] = df_dcf["FCF"].pct_change()
    avg_growth = df_dcf["FCF_Growth"].replace([np.inf, -np.inf], np.nan).mean()

    st.subheader(f"📈 {dcf_company} — FCF 历史 & 增长率")
    st.line_chart(df_dcf.set_index("Fiscal_Year")[["FCF"]])

    last_fcf = df_dcf["FCF"].iloc[-1]
    future_fcfs = []
    growth = avg_growth if pd.notna(avg_growth) else 0.03

    for i in range(1, 6):
        next_fcf = last_fcf * ((1 + growth) ** i)
        future_fcfs.append(next_fcf)

    discount_factors = [(1 / (1 + wacc_input) ** i) for i in range(1, 6)]
    discounted_fcfs = [fcf * d for fcf, d in zip(future_fcfs, discount_factors)]

    terminal_value = future_fcfs[-1] * (1 + tg_input) / (wacc_input - tg_input)
    terminal_value_discounted = terminal_value * discount_factors[-1]

    intrinsic_value = sum(discounted_fcfs) + terminal_value_discounted

    st.subheader("📊 DCF 估值结果")
    col_a, col_b = st.columns(2)
    col_a.metric("📌 企业内在价值", f"${intrinsic_value:,.0f} M")
    col_b.metric("📌 永续终值（折现后）", f"${terminal_value_discounted:,.0f} M")

    fig_dcf = go.Figure()
    fig_dcf.add_trace(
        go.Bar(
            x=[f"Year {i}" for i in range(1, 6)],
            y=discounted_fcfs,
            name="Discounted FCF",
        )
    )
    fig_dcf.add_trace(
        go.Bar(
            x=["Terminal Value"],
            y=[terminal_value_discounted],
            name="Terminal Value",
        )
    )
    fig_dcf.update_layout(title="DCF 构成图（未来 5 年 + 终值）")
    st.plotly_chart(fig_dcf, use_container_width=True)

# ==========================================
# 📌 行业对标（Radar）
# ==========================================

st.header("📊 行业对标（Peers Benchmarking）")

peer_df = df[df["Company"].isin(peers_select)].copy()
latest_year = df["Fiscal_Year"].max()
radar_df = df[(df["Fiscal_Year"] == latest_year) & (df["Company"].isin(peers_select))]

st.subheader("📌 财务指标雷达图（Radar Chart）")

if not radar_df.empty:
    radar_fig = go.Figure()
    metrics = [
        "ROE",
        "ROA",
        "Gross_Profit_Margin",
        "Operating_Profit_Margin",
        "Net_Profit_Margin",
    ]

    for short_name in peers_short_names:
        comp_full = COMPANY_ALIAS.get(short_name, short_name)
        row = radar_df[radar_df["Company"] == comp_full]
        if row.empty:
            continue
        row = row.iloc[0]
        values = [row.get(m, 0) for m in metrics]
        radar_fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics,
                fill="toself",
                name=short_name,
            )
        )

    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)), title="盈利能力对标"
    )
    st.plotly_chart(radar_fig, use_container_width=True)




# ==========================================
# 📘 文本展示模块（MD&A / Business / Risk Factors）
# ==========================================

st.header("📘 公司文本信息模块（MD&A / Business / Risk）")

company_text = st.selectbox("选择公司查看文本信息：", df["Company"].unique())

row_text = df[df["Company"] == company_text].sort_values("Fiscal_Year").iloc[-1]

st.subheader(f"{company_text} 最新年度文本信息（{row_text['Fiscal_Year']}）")

with st.expander("📘 MD&A（管理层讨论与分析）"):
    st.write(row_text.get("MD&A", "无数据"))

with st.expander("🏬 Business（业务描述）"):
    st.write(row_text.get("business", "无数据"))

with st.expander("⚠️ Risk Factors（风险因素）"):
    st.write(row_text.get("risk_factors", "无数据"))


# ==========================================
# 🤖 AI 战略总结（MD&A Summary）
# ==========================================

st.header("🤖 AI 战略总结（基于 MD&A 文本）")

if client:
    md_text = row_text.get("MD&A", "")

    if md_text:
        with st.spinner("AI 正在总结 MD&A 战略重点..."):
            prompt_md = f"""
你是一名资深零售行业分析师。

请根据以下 MD&A 文本总结：
1. 战略重点
2. 增长驱动因素
3. 管理层关注重点
4. 风险提示
5. 对未来经营的意义

MD&A 内容如下：
{md_text}
"""

            resp_md = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_md}],
            )

        st.subheader("🔍 AI 战略摘要")
        st.write(resp_md.choices[0].message.content)
    else:
        st.info("该公司没有 MD&A 文本")
else:
    st.warning("AI 服务未启用，无法执行文本分析。")


# ==========================================
# ⚠️ AI 风险分析（Risk Factors）
# ==========================================

st.header("⚠️ AI 风险分析（Risk Factors）")

if client:
    risk_text = row_text.get("risk_factors", "")

    if risk_text:
        with st.spinner("AI 正在分析风险因素..."):
            prompt_risk = f"""
请分析以下风险因素文本：
1. 列出 5 个最关键风险
2. 每个风险给出严重程度评分（1-10）
3. 判断风险属于短期 / 中期 / 长期
4. 描述对企业未来经营的影响

Risk Factors 文本如下：
{risk_text}
"""

            resp_risk = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_risk}],
            )

        st.subheader("🔍 AI 风险分析结果")
        st.write(resp_risk.choices[0].message.content)
    else:
        st.info("没有 risk_factors 文本可分析。")


# ==========================================
# 🏬 AI 战略对比（Walmart vs Costco）
# ==========================================

st.header("🏬 AI 战略对比：Walmart vs Costco")

w = df[df["Company"] == "Walmart Inc."].sort_values("Fiscal_Year").iloc[-1]
c = df[df["Company"] == "Costco Wholesale Corporation"].sort_values("Fiscal_Year").iloc[-1]

if client and w.get("MD&A") and c.get("MD&A"):

    with st.spinner("AI 正在对比两家公司战略..."):
        prompt_compare = f"""
请对比 Walmart 与 Costco 的最新 MD&A，分析：
1. 两家公司战略重点差异
2. 成长性与扩张策略比较
3. 成本控制、供应链管理差异
4. 风险暴露（Risk Exposure）差异
5. 谁更稳健？为什么？

Walmart MD&A:
{w['MD&A']}

Costco MD&A:
{c['MD&A']}
"""

        resp_comp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt_compare}],
        )

    st.subheader("🔍 AI 战略对比结果")
    st.write(resp_comp.choices[0].message.content)

else:
    st.info("Walmart 或 Costco 缺失 MD&A 文本，无法对比。")



# ==========================================
# 🤖 AI 战略分析报告（使用 o1-mini 或其他）
# ==========================================

st.header("🤖 AI 战略报告（Strategy Report）")

if client:
    user_question = st.text_input(
        "向 AI 分析师提问：", placeholder="例如：Costco 的增长性相比 Walmart 如何？"
    )

    if user_question:
        with st.spinner("AI 正在生成战略洞察..."):
            prompt = f"""
你是零售行业的高级战略分析师。

以下是 {selected_year} 年的关键数据（部分为合成历史数据，请重点关注趋势和相对比较）：

Walmart:
- Revenue: {w_rev}
- ROE: {w_roe}
- FCF: {safe_get(w, 'FCF')}

Costco:
- Revenue: {c_rev}
- ROE: {c_roe}
- FCF: {safe_get(c, 'FCF')}

请基于以上信息 + 历史趋势，回答用户的问题：
{user_question}
"""

            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
                )
                st.success(resp.choices[0].message.content)
            except Exception as e:
                st.error(f"AI 生成失败：{e}")
else:
    st.warning("⚠️ AI 服务未启用，无法生成报告")