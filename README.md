# 管理者短视主义的文本测度及其对研发投入的影响——基于 NLP 词典法的预演毕业论文

> 大连财经学院 · 大数据管理与应用 · 2026 届毕业论文方法预备课《管理研究中的 NLP 测度构建方法》课后作业：三周预演毕业论文（≥6000 字）

本仓库包含预演论文全文及全部可复现支撑材料：五步流水线 Python 脚本、企业-年份面板数据、生成图表与结果表。

## 仓库结构

```
nlp-myopia-thesis/
├── README.md                     ← 本文件（数据来源与文件说明）
├── thesis/
│   └── 预演论文.md                ← 论文全文（摘要/绪论/文献综述/方法/实证/结论，≥6000字）
├── code/
│   └── pipeline.py               ← NLP 测度构建五步流水线脚本（语料→预处理→词典法→指标→信效度）
├── data/
│   ├── corpus_mdna_simulated.csv ← 模拟 MD&A 语料（1800 篇）
│   ├── panel_firm_year.csv       ← 企业-年份面板（测度结果表，300 家 × 6 年）
│   ├── table_descriptive.csv     ← 描述性统计表
│   └── table_regression.csv      ← 回归结果表
└── figures/
    ├── fig1_distribution.png     ← 图1 测度分布直方图
    ├── fig2_trend.png            ← 图2 年度趋势图
    ├── fig3_corr_heatmap.png     ← 图3 相关系数热图
    └── fig4_group_mean.png       ← 图4 条件均值检验图
```

## 数据来源说明（每个文件一句话）

- `corpus_mdna_simulated.csv`：由 `pipeline.py` 按内置 MD&A 模板与词库程序化模拟生成的语料（真实年报全文涉及版权，故以模拟文本演示完整流水线，真实效应已埋入数据生成过程）。
- `panel_firm_year.csv`：由流水线 Step3–Step4 对语料做词典法计数并标准化得到的企业-年份面板（测度结果表），财务变量按论文所述 DGP 模拟。
- `table_descriptive.csv` / `table_regression.csv`：由流水线 Step5 自动输出的描述性统计与回归结果。
- `figures/*.png`：由流水线自动绘制的论文用图。
- `thesis/预演论文.md`：依据上述结果手工撰写的论文全文。
- `code/pipeline.py`：固定随机种子（20261008），全流程可一键复现。

## 复现方式

```bash
pip install jieba pandas numpy matplotlib scipy statsmodels
python code/pipeline.py
```

运行后自动重新生成 `data/` 下全部表格与 `figures/` 下全部图片，终端输出信效度检验与回归结果。

## 研究设计速览

| 项目 | 内容 |
|---|---|
| 研究问题 | 管理者短视主义是否抑制企业研发投入？ |
| 测度方法 | 词典法：ShortTerm = 短期视域词命中数 ÷ MD&A 总词数 × 100（43 词，借鉴胡楠等 2021） |
| 样本 | 模拟 300 家公司 × 2018–2023 年 = 1800 个企业-年度观测 |
| 信效度 | 聚合效度、区分效度、条件均值检验、方差分解（ICC）、预测效度（OLS + 年份/行业固定效应，公司层面聚类稳健标准误） |
| 核心结论 | 管理者短视程度与研发投入强度显著负相关 |
