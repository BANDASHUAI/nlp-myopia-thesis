# -*- coding: utf-8 -*-
"""
管理者短视主义文本测度：NLP 测度构建五步流水线（预演论文配套代码）
========================================================================
步骤总览（对应《管理研究中的 NLP 测度构建方法》课程框架）：
  Step1 语料获取 Corpus Acquisition   —— 模拟生成上市公司年报 MD&A 文本语料
  Step2 文本预处理 Preprocessing       —— 清洗、jieba 精确分词、三维停用词过滤
  Step3 特征提取 Feature Extraction    —— 词典法（短期视域词典，借鉴胡楠等2021）
  Step4 指标构建 Measure Construction  —— ShortTerm = 短期视域词频/总词数×100，
                                          聚合为企业-年份面板
  Step5 信效度检验 Reliability & Validity
        - 聚合效度：与备选测度（业绩说明会短视词占比）Pearson 相关
        - 区分效度：与公司规模、行业等"不该相关"变量的相关性
        - 条件均值检验：按测度五分位分组看研发投入均值单调性
        - 方差分解：企业间差异占比（ICC）
        - 预测效度：OLS 回归 RD ~ ShortTerm + 控制变量（含年份/行业固定效应）

数据说明：本脚本为课程预演论文配套代码。由于上市公司年报全文涉及版权，
语料由内置模板与词库按给定参数程序化模拟生成，财务变量亦按已知数据
生成过程（DGP）模拟，其中真实因果效应已埋入（theta -> RD 为负）。
所有随机过程设定种子，结果可完整复现。
"""

import os
import numpy as np
import pandas as pd
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

# ---------------------------------------------------------------- 全局设置
SEED = 20261008
rng = np.random.default_rng(SEED)
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

N_FIRM = 300
YEARS = list(range(2018, 2024))          # 2018-2023，6 年
INDUSTRIES = ["制造业", "信息技术", "批发零售", "建筑业", "交通运输", "房地产"]

# ------------------------------------------------- 短期视域词典（示例词表）
# 借鉴胡楠等(2021)《管理世界》"管理者短视主义"测度思路，
# 基于行为经济学"短期视域"理论人工筛选并多轮校验，共 43 词。
SHORTTERM_DICT = [
    "短期", "尽快", "立刻", "立即", "眼下", "当前", "目前", "短期利益",
    "短期目标", "短期业绩", "应急", "权宜", "临时", "即期", "当季", "本月",
    "本季度", "年内", "当年", "见效", "快速回报", "立竿见影", "短平快",
    "快进快出", "周转", "回笼资金", "现金流压力", "业绩压力", "考核指标",
    "季度目标", "月度目标", "冲刺", "保增长", "稳指标", "赶进度",
    "抢工期", "压缩周期", "缩短周期", "快速变现", "及时止损", "短线操作",
    "眼前利益", "当务之急",
]
assert len(SHORTTERM_DICT) == 43, "短期视域词典须为 43 词"

# 长期导向词（用于对照/稳健性）
LONGTERM_DICT = [
    "长期", "战略", "可持续", "研发投入", "核心竞争力", "长远", "百年",
    "深耕", "积淀", "布局未来", "长期主义", "基础研究",
]

# 通用 + 财经领域 + 自定义 三维停用词（节选，论文中报告数量）
STOPWORDS = set("""的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
公司 本公司 报告期 报告期内 年度 同比 增长 下降 万元 亿元 股东 董事 董事会 监事会 监事 情况 相关 上述 以下
其中 以及 进行 通过 根据 按照 关于 对于 由于 因此 同时 此外 另外 我们 你们 他们 其 之 与 及 或 等 对 为 以
于 从 至 由 被 将 已 曾 正 正在 各 该 本 此 其 中 内 外 前 后 年 月 日 第 页 证券 代码 简称 公告""".split())

# MD&A 文本模板与填充词库（模拟语料用）
NEUTRAL_SENT = [
    "报告期内公司主营业务收入保持平稳态势", "公司持续推进内部管理优化工作",
    "市场环境变化对公司经营带来一定影响", "公司严格执行内部控制制度",
    "公司加强应收账款管理与存货周转管理", "行业竞争格局总体保持稳定",
    "公司积极履行社会责任参与公益活动", "公司不断完善公司治理结构",
    "上下游供应链协同效率有所提升", "公司重视安全生产与环保合规",
]
SHORT_SENT = [
    "为缓解现金流压力公司要求各业务线尽快回笼资金",
    "面对业绩压力管理层将季度目标分解到月并全力冲刺",
    "公司采取应急措施压缩项目周期确保当年见效",
    "在当前市场环境下公司优先保证短期目标的达成",
    "管理层强调立竿见影的改进措施以应对考核指标",
    "公司鼓励短平快项目以快速变现改善即期业绩",
]
LONG_SENT = [
    "公司坚持长期主义持续加大研发投入力度",
    "围绕核心竞争力公司深化基础研究布局未来",
    "公司推进可持续发展战略深耕主业积淀能力",
]


def simulate_firms():
    """生成公司层面固定属性与'真实'短视倾向 theta（不可观测，供效度验证）。"""
    firms = pd.DataFrame({
        "firm_id": [f"{i:06d}" for i in range(1, N_FIRM + 1)],
        "industry": rng.choice(INDUSTRIES, N_FIRM, p=[0.45, 0.15, 0.12, 0.10, 0.10, 0.08]),
        "soe": rng.binomial(1, 0.3, N_FIRM),
        "theta": np.clip(rng.normal(0.5, 0.18, N_FIRM), 0.05, 0.95),  # 真实短视倾向
    })
    return firms


# ------------------------------------------------------------ Step1 语料获取
def simulate_mdna(theta, year, rng_local):
    """按短视倾向 theta 模拟一篇 MD&A 文本（词数 800~1500）。"""
    n_short = int(rng_local.poisson(6 + 26 * theta))          # 短视句数量随 theta 上升
    n_long = int(rng_local.poisson(5 + 8 * (1 - theta)))
    n_neutral = int(rng_local.integers(60, 110))
    sents = (list(rng_local.choice(SHORT_SENT, n_short))
             + list(rng_local.choice(LONG_SENT, n_long))
             + list(rng_local.choice(NEUTRAL_SENT, n_neutral)))
    rng_local.shuffle(sents)
    return "。".join(sents) + "。"


def step1_corpus(firms):
    rows = []
    for _, f in firms.iterrows():
        for y in YEARS:
            theta_t = np.clip(f["theta"] + rng.normal(0, 0.05), 0.02, 0.98)
            rows.append({
                "firm_id": f["firm_id"], "year": y,
                "theta_true": round(theta_t, 4),   # 仅供效度验证，实际研究中不可观测
                "mdna_text": simulate_mdna(theta_t, y, rng),
            })
    corpus = pd.DataFrame(rows)
    corpus.to_csv(os.path.join(DATA, "corpus_mdna_simulated.csv"),
                  index=False, encoding="utf-8-sig")
    print(f"[Step1] 语料获取完成：{len(corpus)} 篇 MD&A 模拟文本 -> corpus_mdna_simulated.csv")
    return corpus


# ------------------------------------------------------------ Step2 文本预处理
def preprocess(text):
    """清洗 -> jieba 精确分词 -> 三维停用词过滤，返回词列表。"""
    text = "".join(ch for ch in text if ch.isalnum() or ord(ch) > 127 or ch in "。%")
    words = jieba.lcut(text, cut_all=False)          # 精确模式
    return [w for w in words if len(w) > 1 and w not in STOPWORDS and not w.isdigit()]


def step2_preprocess(corpus):
    corpus["tokens"] = corpus["mdna_text"].map(preprocess)
    corpus["n_words"] = corpus["tokens"].map(len)
    print(f"[Step2] 预处理完成：平均篇长 {corpus['n_words'].mean():.0f} 词（jieba 精确模式）")
    return corpus


# ------------------------------------------------------------ Step3 特征提取（词典法）
def step3_features(corpus):
    def count_hits(tokens, dic):
        s = set(dic)
        return sum(1 for t in tokens if t in s)

    corpus["short_hits"] = corpus["tokens"].map(lambda t: count_hits(t, SHORTTERM_DICT))
    corpus["long_hits"] = corpus["tokens"].map(lambda t: count_hits(t, LONGTERM_DICT))
    print(f"[Step3] 特征提取完成：短期视域词命中均值 {corpus['short_hits'].mean():.2f} 次/篇")
    return corpus


# ------------------------------------------------------------ Step4 指标构建
def step4_measure(corpus, firms):
    # 主测度：ShortTerm = 短期视域词命中数 / 总词数 × 100
    corpus["ShortTerm"] = corpus["short_hits"] / corpus["n_words"] * 100
    corpus["LongTerm"] = corpus["long_hits"] / corpus["n_words"] * 100

    # 备选测度（聚合效度用）：模拟业绩说明会短视词占比，与真实 theta 相关但独立测量
    corpus["AltMeasure"] = (corpus["theta_true"] * 8 + rng.normal(0, 1.2, len(corpus))).round(4)

    # 财务变量 DGP：RD = 6 - 3.2*theta_z + 控制变量效应 + 噪声（真实效应为负）
    df = corpus.merge(firms, on="firm_id")
    theta_z = (df["theta_true"] - df["theta_true"].mean()) / df["theta_true"].std()
    df["Size"] = np.clip(22 + rng.normal(0, 1, len(df)) - 0.1 * theta_z, 18, 27)
    df["Lev"] = np.clip(0.42 + 0.10 * theta_z + rng.normal(0, 0.08, len(df)), 0.05, 0.9)
    df["ROA"] = np.clip(0.05 - 0.02 * theta_z + rng.normal(0, 0.03, len(df)), -0.2, 0.3)
    df["Growth"] = np.clip(0.12 - 0.05 * theta_z + rng.normal(0, 0.10, len(df)), -0.5, 1.0)
    df["Age"] = rng.integers(3, 26, len(df))
    df["Top1"] = np.clip(0.32 + rng.normal(0, 0.12, len(df)), 0.05, 0.75)
    year_fe = {y: (y - 2018) * 0.15 for y in YEARS}
    df["RD"] = np.clip(
        6.0 - 3.2 * theta_z + 0.25 * (df["Size"] - 22) - 1.5 * df["Lev"]
        + 2.0 * df["ROA"] + df["year"].map(year_fe) + rng.normal(0, 1.0, len(df)),
        0.1, 20)

    # 样本筛选（模拟论文惯例）：剔除上市不足一年(Age<2已隐含)、关键变量缺失、1%/99% 缩尾
    panel = df[["firm_id", "year", "industry", "soe", "ShortTerm", "LongTerm",
                "AltMeasure", "RD", "Size", "Lev", "ROA", "Growth", "Age", "Top1",
                "n_words", "short_hits", "theta_true"]].copy()

    def winsor(s, p=0.01):
        lo, hi = s.quantile(p), s.quantile(1 - p)
        return s.clip(lo, hi)

    for c in ["ShortTerm", "RD", "Size", "Lev", "ROA", "Growth", "Top1"]:
        panel[c] = winsor(panel[c])

    panel = panel.rename(columns={"firm_id": "Stkcd", "year": "Year"})
    panel.to_csv(os.path.join(DATA, "panel_firm_year.csv"), index=False, encoding="utf-8-sig")
    print(f"[Step4] 指标构建完成：{panel['Stkcd'].nunique()} 家公司 × {panel['Year'].nunique()} 年"
          f" = {len(panel)} 个企业-年度观测 -> panel_firm_year.csv")
    return panel


# ------------------------------------------------------------ Step5 信效度检验
def step5_validity(panel):
    print("\n[Step5] 信效度检验" + "=" * 40)
    results = {}

    # (1) 聚合效度：与备选测度相关
    r_conv, p_conv = stats.pearsonr(panel["ShortTerm"], panel["AltMeasure"])
    results["convergent"] = (r_conv, p_conv)
    print(f"(1) 聚合效度：ShortTerm 与备选测度 Pearson r = {r_conv:.3f} (p = {p_conv:.2e})")

    # (2) 区分效度：与 Size、Age 等"不该强相关"变量低相关
    r_size, _ = stats.pearsonr(panel["ShortTerm"], panel["Size"])
    r_age, _ = stats.pearsonr(panel["ShortTerm"], panel["Age"])
    results["discriminant"] = (r_size, r_age)
    print(f"(2) 区分效度：与 Size r = {r_size:.3f}，与 Age r = {r_age:.3f}（|r|<0.3 为佳）")

    # (3) 与"真实"theta 的相关（模拟数据独有的终极检验）
    r_true, _ = stats.pearsonr(panel["ShortTerm"], panel["theta_true"])
    results["true"] = r_true
    print(f"(3) 与真实短视倾向 theta（模拟 DGP 中不可观测）r = {r_true:.3f}")

    # (4) 条件均值检验：五分位分组 RD 均值
    panel["q"] = pd.qcut(panel["ShortTerm"], 5, labels=["Q1(最低)", "Q2", "Q3", "Q4", "Q5(最高)"])
    grp = panel.groupby("q", observed=True)["RD"].agg(["mean", "std", "count"])
    results["group"] = grp
    print("(4) 条件均值检验（按 ShortTerm 五分位分组的 RD 均值）：")
    print(grp.round(3).to_string())

    # (5) 方差分解：企业间差异占比 ICC
    grand = panel["ShortTerm"].mean()
    fm = panel.groupby("Stkcd")["ShortTerm"].mean()
    n_bar = panel.groupby("Stkcd").size().mean()
    msb = ((fm - grand) ** 2).sum() * n_bar / (panel["Stkcd"].nunique() - 1)
    msw = ((panel["ShortTerm"] - panel["Stkcd"].map(fm)) ** 2).sum() / (len(panel) - panel["Stkcd"].nunique())
    icc = (msb - msw) / (msb + (n_bar - 1) * msw)
    results["icc"] = icc
    print(f"(5) 方差分解：企业间差异占比 ICC = {icc:.3f}（>0.5 说明测度捕捉了稳定企业特征）")

    # (6) 预测效度：OLS 回归 RD ~ ShortTerm + 控制变量 + 年份/行业固定效应
    m1 = smf.ols("RD ~ ShortTerm", data=panel).fit(cov_type="cluster",
                                                   cov_kwds={"groups": panel["Stkcd"]})
    m2 = smf.ols("RD ~ ShortTerm + Size + Lev + ROA + Growth + Age + Top1 + soe",
                 data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["Stkcd"]})
    m3 = smf.ols("RD ~ ShortTerm + Size + Lev + ROA + Growth + Age + Top1 + soe"
                 " + C(Year) + C(industry)",
                 data=panel).fit(cov_type="cluster", cov_kwds={"groups": panel["Stkcd"]})
    results["models"] = (m1, m2, m3)
    for i, m in enumerate((m1, m2, m3), 1):
        b, t, p = m.params["ShortTerm"], m.tvalues["ShortTerm"], m.pvalues["ShortTerm"]
        print(f"(6-{i}) 模型{i}：ShortTerm 系数 = {b:.4f}，t = {t:.2f}，p = {p:.2e}，"
              f"R² = {m.rsquared:.3f}，N = {int(m.nobs)}")
    return results


# ------------------------------------------------------------ 图表输出
def make_figures(panel, results):
    # 图1 测度分布直方图
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.hist(panel["ShortTerm"], bins=40, color="#4C78A8", edgecolor="white")
    ax.axvline(panel["ShortTerm"].mean(), color="#E45756", ls="--",
               label=f"均值 = {panel['ShortTerm'].mean():.3f}")
    ax.set_xlabel("ShortTerm（短期视域词频/总词数×100）")
    ax.set_ylabel("频数")
    ax.set_title("图1  管理者短视测度 ShortTerm 分布")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_distribution.png"))
    plt.close(fig)

    # 图2 年度均值趋势
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    yr = panel.groupby("Year")[["ShortTerm", "RD"]].mean()
    ax.plot(yr.index, yr["ShortTerm"], "o-", color="#4C78A8", label="ShortTerm（左轴）")
    ax.set_xlabel("年份")
    ax.set_ylabel("ShortTerm")
    ax2 = ax.twinx()
    ax2.plot(yr.index, yr["RD"], "s--", color="#F58518", label="RD 强度（右轴，%）")
    ax2.set_ylabel("RD 强度（%）")
    ax.set_title("图2  短视测度与研发投入强度的年度趋势")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_trend.png"))
    plt.close(fig)

    # 图3 相关系数热图
    cols = ["ShortTerm", "AltMeasure", "RD", "Size", "Lev", "ROA", "Growth", "Age", "Top1"]
    corr = panel[cols].corr()
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=150)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i, j]) > 0.6 else "black", fontsize=8)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("图3  主要变量 Pearson 相关系数矩阵")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_corr_heatmap.png"))
    plt.close(fig)

    # 图4 五分位条件均值
    grp = results["group"]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    ax.bar(grp.index.astype(str), grp["mean"], yerr=grp["std"] / np.sqrt(grp["count"]),
           color="#54A24B", edgecolor="white", capsize=4)
    ax.set_xlabel("ShortTerm 五分位组")
    ax.set_ylabel("RD 强度均值（%）")
    ax.set_title("图4  条件均值检验：短视程度越高，研发投入越低")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_group_mean.png"))
    plt.close(fig)
    print("\n[Figures] 已输出 4 张图 -> figures/")


# ------------------------------------------------------------ 描述性统计表
def descriptive(panel):
    cols = ["ShortTerm", "LongTerm", "AltMeasure", "RD", "Size", "Lev", "ROA",
            "Growth", "Age", "Top1", "n_words"]
    desc = panel[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
    desc.to_csv(os.path.join(DATA, "table_descriptive.csv"), encoding="utf-8-sig")

    # 回归结果表
    m1, m2, m3 = results_models
    rows = []
    for name, m in zip(["模型(1)", "模型(2)", "模型(3)"], (m1, m2, m3)):
        for v in ["ShortTerm", "Size", "Lev", "ROA", "Growth", "Age", "Top1", "soe"]:
            if v in m.params.index:
                star = "***" if m.pvalues[v] < 0.01 else "**" if m.pvalues[v] < 0.05 else "*" if m.pvalues[v] < 0.1 else ""
                rows.append({"变量": v, "模型": name,
                             "系数": f"{m.params[v]:.4f}{star}", "t值": f"{m.tvalues[v]:.2f}"})
        rows.append({"变量": "R²", "模型": name, "系数": f"{m.rsquared:.3f}", "t值": ""})
        rows.append({"变量": "N", "模型": name, "系数": f"{int(m.nobs)}", "t值": ""})
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "table_regression.csv"),
                              index=False, encoding="utf-8-sig")
    print("[Tables] 描述性统计与回归结果表 -> table_descriptive.csv / table_regression.csv")
    return desc


if __name__ == "__main__":
    firms = simulate_firms()
    corpus = step1_corpus(firms)
    corpus = step2_preprocess(corpus)
    corpus = step3_features(corpus)
    panel = step4_measure(corpus, firms)
    results = step5_validity(panel)
    results_models = results["models"]
    desc = descriptive(panel)
    make_figures(panel, results)
    print("\n流水线全部完成。")
