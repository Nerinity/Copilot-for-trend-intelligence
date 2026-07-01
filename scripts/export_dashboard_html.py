#!/usr/bin/env python3
"""Export dashboard to a single self-contained HTML for documentation."""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

ROOT     = Path(__file__).parent.parent
PKL_PATH = ROOT / "data" / "processed" / "dashboard_data_500k.pkl"
OUT_HTML = Path.home() / "Desktop" / "TrendCopilot_dashboard_export.html"

COLORS = {
    "rising":   "#22C55E",
    "stable":   "#94A3B8",
    "declining":"#EF4444",
    "primary":  "#2563EB",
}

DIR_ZH  = {"rising":"上升中","stable":"平稳","declining":"下降中"}
DIR_EN  = {"rising":"Rising","stable":"Stable","declining":"Declining"}
DIR_EMO = {"rising":"🟢","stable":"⚪","declining":"🔴"}

def cat_label(c: str) -> str:
    return c.replace("_"," ").title()

def fig_to_div(fig, h=None):
    if h:
        fig.update_layout(height=h)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displayModeBar": False})

def sentiment_color(s: float) -> str:
    if s >= 0.05:  return COLORS["rising"]
    if s <= -0.05: return COLORS["declining"]
    return COLORS["stable"]

def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    win_label = data["window_labels"][0]
    wd        = data["windows"][win_label]
    stats     = wd["stats"]
    cat_brand = wd["cat_brand_data"]
    win       = wd["window"]

    top25 = stats.head(25).copy()
    top25["cat_label"] = top25["category"].apply(cat_label)

    # ── Chart 1: Trend Score Bar ──────────────────────────────────────────────
    fig1 = go.Figure()
    for d in ["rising", "stable", "declining"]:
        sub = top25[top25["trend_direction"] == d]
        if sub.empty: continue
        fig1.add_trace(go.Bar(
            x=sub["trend_score"], y=sub["cat_label"],
            orientation="h",
            name=f"{DIR_EMO[d]} {DIR_ZH[d]} · {DIR_EN[d]}",
            marker_color=COLORS[d],
            hovertemplate=(
                "<b>%{y}</b><br>"
                "综合趋势分 Score: %{x:.2f}<br>"
                "热度增长 Spike: %{customdata[0]:.1f}x<br>"
                "帖子量 Posts: %{customdata[1]:,}<extra></extra>"
            ),
            customdata=sub[["spike_ratio", "current_mentions"]].values,
        ))
    fig1.update_layout(
        title=dict(text="综合趋势分 — Top 25 品类 / Category Trend Score", font=dict(size=15)),
        barmode="overlay",
        height=580,
        yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
        xaxis=dict(title="综合趋势分 / Trend Score (0–1)", range=[0, 1.1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=40, t=60, b=30),
        font=dict(family="Inter, system-ui, sans-serif"),
    )

    # ── Chart 2: Spike × Reach Bubble ────────────────────────────────────────
    top25["bsize"] = (top25["current_mentions"] / top25["current_mentions"].max() * 55 + 8).round()
    fig2 = go.Figure()
    for x0,y0,x1,y1,col in [(.5,.5,1.05,1.05,"#F0FDF4"),(0,.5,.5,1.05,"#F8FAFC"),
                              (.5,0,1.05,.5,"#EFF6FF"),(0,0,.5,.5,"#FAFAFA")]:
        fig2.add_shape(type="rect",x0=x0,y0=y0,x1=x1,y1=y1,fillcolor=col,opacity=.5,line_width=0)
    for d in ["rising","stable","declining"]:
        sub = top25[top25["trend_direction"]==d]
        if sub.empty: continue
        fig2.add_trace(go.Scatter(
            x=sub["normalized_spike"], y=sub["cross_community"],
            mode="markers+text",
            name=f"{DIR_EMO[d]} {DIR_ZH[d]}",
            marker=dict(size=sub["bsize"],color=COLORS[d],opacity=.75,
                        line=dict(width=1,color="white")),
            text=sub["cat_label"].str[:12],
            textposition="top center",textfont=dict(size=8),
            hovertemplate=(
                "<b>%{text}</b><br>热度 Spike: %{x:.2f}<br>扩散 Reach: %{y:.2f}<extra></extra>"
            ),
        ))
    for xp,yp,lab in [(.75,.97,"🔥 高热·广扩"),(.18,.97,"广扩·低热"),
                       (.75,.03,"局部爆点"),(.18,.03,"低优先")]:
        fig2.add_annotation(x=xp,y=yp,text=lab,showarrow=False,font=dict(size=9,color="#94A3B8"))
    fig2.add_hline(y=.5,line_dash="dot",line_color="#CBD5E1",line_width=1)
    fig2.add_vline(x=.5,line_dash="dot",line_color="#CBD5E1",line_width=1)
    fig2.update_layout(
        title=dict(text="热度 × 扩散象限图 / Spike × Cross-community Reach", font=dict(size=15)),
        xaxis=dict(title="热度增长强度 / Spike Intensity",range=[-.05,1.1]),
        yaxis=dict(title="跨社区扩散度 / Cross-community Reach",range=[-.05,1.1]),
        height=500,plot_bgcolor="white",paper_bgcolor="white",
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
        margin=dict(l=10,r=10,t=60,b=30),
        font=dict(family="Inter, system-ui, sans-serif"),
    )

    # ── Chart 3: Spike Board Top 6 ───────────────────────────────────────────
    top6 = stats.nlargest(6, "normalized_spike")
    spike_figs_html = []
    for _, row in top6.iterrows():
        cat  = row["category"]
        bdf  = cat_brand.get(cat, pd.DataFrame())
        if bdf.empty: continue
        bdf_s = bdf.head(10).copy()
        bdf_s["brand_short"] = bdf_s["brand"].str[:20]
        fg = go.Figure(go.Bar(
            x=bdf_s["cur_mentions"], y=bdf_s["brand_short"],
            orientation="h",
            marker_color=COLORS["primary"], opacity=0.8,
            text=bdf_s["cur_mentions"].apply(lambda v: f"{v:,}"),
            textposition="outside", textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>提及量: %{x:,}<extra></extra>",
        ))
        dcolor = COLORS[row["trend_direction"]]
        fg.update_layout(
            title=dict(
                text=f"#{list(top6['category']).index(cat)+1}  {cat_label(cat)}  "
                     f"<span style='color:{dcolor}'>{row['spike_ratio']:.1f}x</span>  "
                     f"| {int(row['current_mentions']):,} posts",
                font=dict(size=13)
            ),
            height=280,
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=9)),
            xaxis=dict(title="提及量 Mentions", showgrid=True, gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=5, r=50, t=50, b=25),
            showlegend=False,
            font=dict(family="Inter, system-ui, sans-serif"),
        )
        spike_figs_html.append(fig_to_div(fg))

    # ── Chart 4 & 5: Sample category detail (top category) ───────────────────
    sample_cat = stats.iloc[0]["category"]
    bdf_full   = cat_brand.get(sample_cat, pd.DataFrame())

    fig4 = go.Figure()  # sentiment-colored brand bar
    fig5 = go.Figure()  # delta bar

    if not bdf_full.empty:
        bdf_s = bdf_full.head(15).copy()
        bdf_s["bar_color"] = bdf_s["avg_sentiment"].apply(sentiment_color)
        bdf_s["brand_short"] = bdf_s["brand"].str[:22]
        bdf_s["b_spike"] = (bdf_s["cur_mentions"] / bdf_s["prev_mentions"].replace(0,1)).round(2)

        fig4 = go.Figure(go.Bar(
            x=bdf_s["cur_mentions"], y=bdf_s["brand_short"],
            orientation="h", marker_color=bdf_s["bar_color"],
            hovertemplate=(
                "<b>%{y}</b><br>当前提及: %{x:,}<br>"
                "好感度 Sentiment: %{customdata:+.2f}<extra></extra>"
            ),
            customdata=bdf_s["avg_sentiment"],
        ))
        fig4.update_layout(
            title=dict(text=f"品牌提及量（按好感度着色）— {cat_label(sample_cat)}<br>"
                            "<sup>🟢正面  ⚪中性  🔴负面</sup>", font=dict(size=13)),
            height=440,
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=10)),
            xaxis=dict(title="近两周提及量 / Current Mentions", showgrid=True, gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=30, t=70, b=30),
            font=dict(family="Inter, system-ui, sans-serif"),
        )

        bdf_d = bdf_full.head(15).copy()
        bdf_d["delta"] = (bdf_d["cur_mentions"] - bdf_d["prev_mentions"]).astype(int)
        bdf_d = bdf_d.sort_values("delta", ascending=True)
        bdf_d["brand_short"] = bdf_d["brand"].str[:20]
        bdf_d["bar_color"] = bdf_d["delta"].apply(lambda d: COLORS["rising"] if d>0 else COLORS["declining"])
        bdf_d["delta_txt"] = bdf_d["delta"].apply(lambda d: f"+{d:,}" if d>=0 else f"{d:,}")

        fig5 = go.Figure(go.Bar(
            x=bdf_d["delta"], y=bdf_d["brand_short"],
            orientation="h", marker_color=bdf_d["bar_color"],
            text=bdf_d["delta_txt"], textposition="outside", textfont=dict(size=9),
            hovertemplate="<b>%{y}</b><br>环比增减: %{x:,}<extra></extra>",
        ))
        fig5.add_vline(x=0, line_color="#CBD5E1", line_width=1)
        fig5.update_layout(
            title=dict(text=f"品牌环比增量（本期 vs 上期）— {cat_label(sample_cat)}", font=dict(size=13)),
            height=440,
            yaxis=dict(tickfont=dict(size=10)),
            xaxis=dict(title="环比增减量 / Delta vs Prior Period", showgrid=True, gridcolor="#F3F4F6"),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=55, t=60, b=30),
            font=dict(family="Inter, system-ui, sans-serif"),
        )

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    charts_row1 = f"""
    <div class="row2">
      <div class="chart-box">{fig_to_div(fig1, 580)}</div>
      <div class="chart-box">{fig_to_div(fig2, 500)}</div>
    </div>"""

    spike_grid_html = ""
    for i in range(0, len(spike_figs_html), 2):
        row_html = '<div class="row2">'
        row_html += f'<div class="chart-box">{spike_figs_html[i]}</div>'
        if i+1 < len(spike_figs_html):
            row_html += f'<div class="chart-box">{spike_figs_html[i+1]}</div>'
        row_html += "</div>"
        spike_grid_html += row_html

    detail_html = f"""
    <div class="row2">
      <div class="chart-box">{fig_to_div(fig4, 440)}</div>
      <div class="chart-box">{fig_to_div(fig5, 440)}</div>
    </div>"""

    # Build metric rows from stats
    def badge(d):
        c = COLORS[d]; t = f"{DIR_EMO[d]} {DIR_ZH[d]} · {DIR_EN[d]}"
        return f'<span class="badge" style="background:{c}18;color:{c}">{t}</span>'

    rows_html = ""
    for _, r in stats.head(20).iterrows():
        sent_class = "pos" if r["mean_sentiment"]>=0.05 else ("neg" if r["mean_sentiment"]<=-0.05 else "neu")
        rows_html += f"""
        <tr>
          <td>{cat_label(r['category'])}</td>
          <td>{badge(r['trend_direction'])}</td>
          <td><b>{r['trend_score']:.2f}</b></td>
          <td>{r['spike_ratio']:.1f}x</td>
          <td>{int(r['current_mentions']):,}</td>
          <td class="{sent_class}">{r['mean_sentiment']:+.2f}</td>
          <td>{'▲' if r['mentions_delta']>=0 else '▼'} {abs(int(r['mentions_delta'])):,}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reddit Trend Copilot — Dashboard Export</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Inter',system-ui,sans-serif;background:#F8FAFC;color:#111827;}}
  .container{{max-width:1400px;margin:0 auto;padding:24px 20px;}}
  .header{{margin-bottom:24px;padding:20px 24px;background:white;border-radius:12px;
    border:1px solid #E5E7EB;box-shadow:0 1px 4px rgba(0,0,0,.05);}}
  .header h1{{font-size:1.5rem;font-weight:700;color:#111;margin-bottom:4px;}}
  .header .sub{{font-size:.85rem;color:#6B7280;}}
  .header .meta{{font-size:.8rem;color:#2563EB;margin-top:6px;font-weight:500;}}
  .section-title{{font-size:1rem;font-weight:700;color:#111;margin:28px 0 8px;
    padding-left:12px;border-left:4px solid #2563EB;}}
  .caption{{font-size:.78rem;color:#6B7280;padding:6px 10px;background:#F8FAFC;
    border-left:3px solid #2563EB;border-radius:0 6px 6px 0;margin-bottom:12px;line-height:1.5;}}
  .row2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;}}
  .chart-box{{background:white;border:1px solid #E5E7EB;border-radius:12px;
    padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.05);overflow:hidden;}}
  .metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:20px;}}
  .metric{{background:white;border:1px solid #E5E7EB;border-radius:10px;padding:14px 16px;}}
  .metric .label{{font-size:.72rem;color:#6B7280;margin-bottom:4px;}}
  .metric .value{{font-size:1.4rem;font-weight:700;color:#111;}}
  .metric .delta{{font-size:.75rem;color:#6B7280;margin-top:2px;}}
  table{{width:100%;border-collapse:collapse;background:white;border-radius:10px;
    overflow:hidden;border:1px solid #E5E7EB;font-size:.83rem;}}
  th{{background:#F8FAFC;padding:10px 14px;text-align:left;font-weight:600;
      font-size:.78rem;color:#374151;border-bottom:1px solid #E5E7EB;}}
  td{{padding:8px 14px;border-bottom:1px solid #F3F4F6;}}
  tr:last-child td{{border-bottom:none;}}
  tr:hover td{{background:#F9FAFB;}}
  .badge{{border-radius:6px;padding:2px 8px;font-size:.72rem;font-weight:600;white-space:nowrap;}}
  .pos{{color:#22C55E;font-weight:600;}}
  .neg{{color:#EF4444;font-weight:600;}}
  .neu{{color:#94A3B8;}}
  @media(max-width:900px){{.row2{{grid-template-columns:1fr;}}.metrics{{grid-template-columns:repeat(2,1fr);}}}}</style>
</head>
<body>
<div class="container">

  <div class="header">
    <h1>📡 Reddit Product Trend Copilot</h1>
    <div class="sub">Reddit 社区讨论产品趋势雷达 · Reddit Community Trend Radar</div>
    <div class="meta">
      分析周期 Period: {win['cur_start'].strftime('%Y/%m/%d')} → {win['cur_end'].strftime('%Y/%m/%d')}
      &nbsp;|&nbsp; 对比周期 Compare: {win['prev_start'].strftime('%Y/%m/%d')} → {win['prev_end'].strftime('%Y/%m/%d')}
      &nbsp;|&nbsp; 数据来源 Source: Reddit (Arctic-Shift API) · 496,692 posts · 66 categories
    </div>
  </div>

  <!-- Metric summary -->
  <div class="metrics">
    <div class="metric">
      <div class="label">活跃品类 / Active Categories</div>
      <div class="value">{len(stats)}</div>
      <div class="delta">across 66 categories</div>
    </div>
    <div class="metric">
      <div class="label">🟢 上升中 / Rising</div>
      <div class="value" style="color:#22C55E">{(stats['trend_direction']=='rising').sum()}</div>
      <div class="delta">{(stats['trend_direction']=='rising').sum()/len(stats):.0%} of all</div>
    </div>
    <div class="metric">
      <div class="label">⚪ 平稳 / Stable</div>
      <div class="value" style="color:#94A3B8">{(stats['trend_direction']=='stable').sum()}</div>
      <div class="delta">{(stats['trend_direction']=='stable').sum()/len(stats):.0%} of all</div>
    </div>
    <div class="metric">
      <div class="label">🔴 下降 / Declining</div>
      <div class="value" style="color:#EF4444">{(stats['trend_direction']=='declining').sum()}</div>
      <div class="delta">{(stats['trend_direction']=='declining').sum()/len(stats):.0%} of all</div>
    </div>
    <div class="metric">
      <div class="label">趋势分算法 / Score Formula</div>
      <div class="value" style="font-size:.95rem;color:#2563EB">各 25%</div>
      <div class="delta">Spike · Reach · Sentiment · Engagement</div>
    </div>
  </div>

  <!-- Tab 1: Charts -->
  <div class="section-title">📊 Tab 1 — 趋势品类目 / Category Trends</div>
  <div class="caption">
    <b>综合趋势分</b> = 热度增长（25%）+ 跨社区扩散（25%）+ 用户好感度（25%）+ 互动参与（25%），四维等权。
    颜色 🟢上升 ⚪平稳 🔴下降。右图气泡大小 = 帖子量，右上角 = 核心机会区。
    <br><b>Trend Score</b> = Spike(25%) + Cross-community Reach(25%) + Sentiment(25%) + Engagement(25%). Right chart: bubble size = post volume; top-right quadrant = high-opportunity zone.
  </div>
  {charts_row1}

  <!-- Data table -->
  <div class="section-title">📋 品类数据总览 / Category Data Summary (Top 20)</div>
  <div style="overflow-x:auto;margin-bottom:20px">
    <table>
      <thead><tr>
        <th>品类 Category</th><th>趋势方向 Direction</th><th>综合分 Score</th>
        <th>热度增长 Spike</th><th>本期帖子 Posts</th><th>好感度 Sentiment</th><th>环比增减 Delta</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>

  <!-- Tab 2: Spike Board -->
  <div class="section-title">🚀 Tab 2 — 飙升榜 / Spike Board (Top 6 × Brand Breakdown)</div>
  <div class="caption">
    仅按近两周热度增长倍数排序，找最近爆发最快的品类。卡片内 Bar = 该品类内各品牌近两周提及量。
    <br>Ranked by 2-week spike ratio only. Bar length = brand mention count in current period.
  </div>
  {spike_grid_html}

  <!-- Tab 3: Category Detail -->
  <div class="section-title">🔍 Tab 3 — 品类详情示例 / Category Detail Sample: {cat_label(sample_cat)}</div>
  <div class="caption">
    左图：条形长度 = 近两周提及帖子数；条形颜色 = 该品牌帖子平均好感度（🟢正面 ⚪中性 🔴负面）。
    右图：条形长度 = 本期 vs 上期提及量增减数（环比增量）。
    <br>Left: bar length = current-period posts; color = avg post sentiment per brand.
    Right: bar = mentions delta vs prior period (positive = growing).
  </div>
  {detail_html}

  <div style="text-align:center;color:#9CA3AF;font-size:.75rem;margin-top:32px;padding-top:16px;border-top:1px solid #E5E7EB">
    Generated by Reddit Trend Copilot · Data: Reddit Arctic-Shift API · NLP: VADER + SentenceTransformers + UMAP + HDBSCAN
  </div>
</div>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ Exported → {OUT_HTML}")
    print(f"   File size: {OUT_HTML.stat().st_size/1024:.0f} KB")

if __name__ == "__main__":
    main()
