
import os
import re
import json
import time
import warnings
import requests
from datetime import datetime, timedelta
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq
from bs4 import BeautifulSoup


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
NEWS_API_KEY     = os.environ.get("NEWS_API_KEY", "")
INDEX_NAME       = "stocks"
NAMESPACE        = "stock-descriptions"
GROQ_MODEL       = "llama-3.1-8b-instant"
EMBED_MODEL      = "sentence-transformers/all-mpnet-base-v2"

RADAR_DIMS = [
    "Growth Potential", "Financial Health", "Market Competition",
    "Innovation", "Industry Trends", "Regulatory Environment",
]
SEC_DIMS = ["Performance", "Growth Potential", "Risks", "Competitive Edge"]

SECTORS = [
    "All Sectors", "Technology", "Healthcare", "Financial Services",
    "Consumer Cyclical", "Consumer Defensive", "Energy", "Industrials",
    "Real Estate", "Utilities", "Communication Services", "Basic Materials",
]

SEC_HEADERS = {"User-Agent": "financial-analysis-app research@example.com"}

# Page config 
st.set_page_config(
    page_title="Stock Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS 
st.markdown("""
<style>
/* FIX: Removed Google Fonts import — use system default font throughout */

html, body, [data-testid="stAppViewContainer"] {
    background: #07080d;
    color: #d4d8e8;
}
[data-testid="stHeader"]       { background: #07080d; }
[data-testid="collapsedControl"] { display: none; }

.page-title {
    font-size: 1.8rem; font-weight: 700;
    color: #ffffff; letter-spacing: -0.02em;
    margin-bottom: 2px;
}
/* FIX: page-sub was dark #374151 — now white/light so it's readable */
.page-sub { color: #c0c8e0; font-size: 0.85rem; margin-bottom: 0; }

.sec-title {
    font-size: 1.05rem; font-weight: 700; color: #ffffff;
    margin: 40px 0 4px;
    padding: 12px 0 10px;
    border-top: 1px solid #1a1f2e;
}
/* FIX: pipeline-info text was dark #4a5568 — now white so it's readable */
.pipeline-info {
    background: #0c0e18;
    border-left: 3px solid #3b82f6;
    padding: 10px 14px;
    font-size: 0.78rem; color: #c8d0e8; line-height: 1.6;
    margin-bottom: 16px; border-radius: 0 6px 6px 0;
}
/* FIX: pipeline-info strong was dark #64748b — now bright white */
.pipeline-info strong { color: #ffffff; font-weight: 700; }

.stock-card {
    background: #0c0e16; border: 1px solid #1a1f2e;
    border-radius: 10px; padding: 18px; margin-bottom: 12px;
}
.ticker-tag  { font-size:1.3rem; font-weight:700; color:#ffffff; }
.score-pill  { float:right; background:#111827; color:#60a5fa; font-size:0.67rem; padding:3px 10px; border-radius:20px; border:1px solid #1e3a5f; }
/* FIX: co-name was dark #374151 — now light grey so company name is readable */
.co-name     { color:#b0bcd4; font-size:0.8rem; margin:3px 0 8px; }
.co-summary  { color:#c0c8e0; font-size:0.8rem; line-height:1.65; }
.co-link     { color:#60a5fa; font-size:0.76rem; }

.metrics-grid { display:flex; flex-wrap:wrap; gap:5px; margin-top:12px; padding-top:11px; border-top:1px solid #141824; }
.metric-cell  { flex:1 1 calc(33% - 5px); min-width:80px; background:#08090d; border-radius:6px; padding:7px 9px; border:1px solid #141824; }
/* FIX: m-label was dark #2d3748 — now light grey so metric labels are readable */
.m-label { color:#8a9ab8; font-size:0.61rem; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:2px; }
.m-pos  { color:#4ade80; font-weight:700; font-size:0.86rem; }
.m-neg  { color:#f87171; font-weight:700; font-size:0.86rem; }
/* FIX: m-na was dark #2d3748 — now light grey */
.m-na   { color:#8a9ab8; font-weight:700; font-size:0.86rem; }

.exp-row { font-size:0.8rem; color:#c0c8e0; margin:6px 0; padding:8px 10px; background:#080a10; border-radius:5px; border-left:2px solid #1e3a5f; line-height:1.5; }
.exp-dim { font-weight:700; color:#60a5fa; }

.news-card { background:#0c0e16; border:1px solid #1a1f2e; border-radius:8px; padding:14px; margin-bottom:8px; }
.news-title  { color:#ffffff; font-size:0.85rem; font-weight:600; margin:3px 0; }
.news-field  { color:#60a5fa; font-weight:600; font-size:0.78rem; }
.news-val    { color:#c0c8e0; font-size:0.78rem; }
/* FIX: news-rel was dark #374151 — now light so relevance reason is readable */
.news-rel    { font-style:italic; color:#a0aec0; font-size:0.74rem; }

.bar-wrap  { margin:8px 0 14px; }
.bar-head  { display:flex; justify-content:space-between; margin-bottom:3px; }
.bar-dim   { color:#c0c8e0; font-size:0.79rem; }
.bar-val   { color:#a78bfa; font-size:0.79rem; font-weight:600; }
.bar-bg    { background:#141824; border-radius:4px; height:8px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
/* FIX: bar-expl was dark #374151 — now light grey so explanation text is readable */
.bar-expl  { color:#a0aec0; font-size:0.75rem; margin-top:3px; }

.rec-card  { background:#0c0e16; border:1px solid #1a1f2e; border-radius:8px; padding:14px; margin-bottom:10px; }
.rec-ticker { color:#60a5fa; font-size:0.9rem; font-weight:700; }
.rec-text  { color:#c0c8e0; font-size:0.82rem; line-height:1.65; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

#Cached resources (load once per session) 

@st.cache_resource(show_spinner="Loading AI embedding model...")
def load_embed():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Connecting to Pinecone...")
def load_pinecone():
    pc    = Pinecone(api_key=PINECONE_API_KEY)
    idx   = pc.Index(INDEX_NAME)
    stats = idx.describe_index_stats()
    return idx, stats

@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)

# Utility functions 

def fmt_pct(v):
    try:
        f   = float(v)
        txt = f"+{f*100:.1f}%" if f >= 0 else f"{f*100:.1f}%"
        cls = "m-pos" if f >= 0 else "m-neg"
        return txt, cls
    except:
        return "N/A", "m-na"

def metric_box(label, value):
    t, c = fmt_pct(value)
    return (f'<div class="metric-cell"><div class="m-label">{label}</div>'
            f'<div class="{c}">{t}</div></div>')

def parse_json(raw):
    try:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return json.loads(cleaned)
    except:
        return {}

def score_color(s):
    if s >= 7: return "#4ade80"
    if s >= 5: return "#facc15"
    return "#f87171"

def bar_html(dim, score, explanation):
    pct   = int(score / 10 * 100)
    color = score_color(score)
    return (
        f'<div class="bar-wrap">'
        f'<div class="bar-head"><span class="bar-dim">{dim}</span>'
        f'<span class="bar-val">{score}/10</span></div>'
        f'<div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>'
        f'<div class="bar-expl">{explanation}</div></div>'
    )

# LLM helpers

def groq_call(client, sys_p, user_p, temp=0.3, max_tok=512):
    r = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":sys_p},{"role":"user","content":user_p}],
        temperature=temp, max_completion_tokens=max_tok,
    )
    return r.choices[0].message.content.strip()

#  Stock search functions 

def expand_query(client, query):
    """
    FIX 1 - QUERY EXPANSION:
    The old prompt was too generic and produced embeddings that missed
    well-known companies like NVDA/AMD when searching "semiconductors".

    The new prompt uses the same technical vocabulary found in actual
    SEC filings and Yahoo Finance business descriptions stored in Pinecone.
    Key vocabulary added: "fabless", "GPU", "CPU", "ASIC", "integrated
    circuits", "wafer fabrication", "IP licensing", "data center accelerator",
    "process node", "EDA software", "hyperscaler", "AI inference".

    This maximises cosine similarity overlap between the expanded query
    embedding and the stock description embeddings in Pinecone, so broad
    queries like "semiconductors" correctly surface NVDA, AMD, TSM, AVGO.

    No company names are mentioned to avoid retrieval bias.
    """
    sys_p = (
        "You are a financial research assistant. Your job is to expand a short "
        "investment search query into a rich, detailed paragraph that will be "
        "embedded and compared against a database of company business descriptions "
        "from SEC filings and Yahoo Finance.\n\n"
        "Write 5-7 sentences describing the type of company the user is looking for. "
        "You MUST use specific technical vocabulary that appears in real company "
        "business descriptions, such as:\n"
        "- Products: GPU, CPU, ASIC, FPGA, SoC, integrated circuits, microprocessors, "
        "  memory chips, DRAM, NAND flash, chipsets, wafers, logic chips\n"
        "- Business model: fabless semiconductor, foundry, IDM (integrated device "
        "  manufacturer), IP licensing, royalties, design automation, EDA software, "
        "  process node, nanometer technology, pure-play foundry\n"
        "- End markets: data center, hyperscaler, cloud computing, AI training, "
        "  AI inference, autonomous vehicles, mobile, gaming, networking, "
        "  industrial IoT, high performance computing\n"
        "- Financial traits: high gross margins, heavy R&D spending, capital "
        "  expenditure intensive, cyclical demand, licensing fees, recurring revenue\n\n"
        "Cover: what the company makes, its role in the supply chain (designer, "
        "manufacturer, IP licensor), which industry sector, who its customers are, "
        "and its typical financial profile.\n\n"
        "Do NOT mention any specific company names, stock tickers, or brand names. "
        "Return ONLY the paragraph. No preamble, no labels, no bullet points."
    )
    return groq_call(client, sys_p, f"Search query: {query}", temp=0.2, max_tok=400).strip()

def search_stocks(query, idx, embed, client, sector=None, top_k=8):
    expanded = expand_query(client, query)
    vector   = embed.encode(expanded).tolist()
    matches  = idx.query(vector=vector, top_k=top_k*4,
                         include_metadata=True, namespace=NAMESPACE)["matches"]
    seen, results = set(), []
    for m in matches:
        meta   = m.get("metadata", {})
        ticker = meta.get("Ticker", "")
        if not ticker or ticker in seen: continue
        if sector and sector != "All Sectors":
            if sector.lower() not in meta.get("Sector","").lower(): continue
        meta["_score"] = round(float(m.get("score",0)),4)
        seen.add(ticker)
        results.append(meta)
        if len(results) >= top_k: break
    return results, expanded

def comparison_summary(client, stocks):
    def fp(v):
        try: return f"{float(v)*100:.1f}%"
        except: return "N/A"
    lines = "\n".join(
        f"- {s.get('Ticker')} ({s.get('Name')}) | "
        f"EG={fp(s.get('Earnings Growth'))} RG={fp(s.get('Revenue Growth'))} "
        f"GM={fp(s.get('Gross Margins'))} EBITDA={fp(s.get('EBITDA Margins'))} "
        f"52W={fp(s.get('52 Week Change'))}"
        for s in stocks
    )
    return groq_call(client,
        "Senior equity analyst. One bullet per stock on its single most significant "
        "strength or weakness with the actual number. Then write a 3-sentence "
        "'Investor Takeaway' identifying the top performer, highest risk, and clear conclusion. "
        "Be specific. Use actual percentages.",
        f"Data:\n{lines}", temp=0.2, max_tok=1200)

# Radar functions 

def score_radar(client, stock):
    def fp(v):
        try: return f"{float(v)*100:.1f}%"
        except: return "N/A"
    ctx = (
        f"Company: {stock.get('Name')} ({stock.get('Ticker')})\n"
        f"Sector: {stock.get('Sector')} | Industry: {stock.get('Industry')}\n"
        f"Earnings Growth: {fp(stock.get('Earnings Growth'))}\n"
        f"Revenue Growth: {fp(stock.get('Revenue Growth'))}\n"
        f"Gross Margins: {fp(stock.get('Gross Margins'))}\n"
        f"EBITDA Margins: {fp(stock.get('EBITDA Margins'))}\n"
        f"52-Week Change: {fp(stock.get('52 Week Change'))}\n"
        f"Business: {stock.get('Business Summary','')[:400]}"
    )
    sys_p = (
        "Score this company on 6 dimensions (0-10). "
        "Return ONLY this JSON format:\n"
        '{"scores":{"Growth Potential":7,"Financial Health":6,"Market Competition":5,'
        '"Innovation":8,"Industry Trends":7,"Regulatory Environment":6},'
        '"explanations":{"Growth Potential":"sentence","Financial Health":"sentence",'
        '"Market Competition":"sentence","Innovation":"sentence",'
        '"Industry Trends":"sentence","Regulatory Environment":"sentence"}}\n'
        "Scoring: 10=excellent, 7-9=good, 5-6=average, 3-4=weak, 0-2=very poor. "
        "Market Competition: 10=near monopoly. Regulatory Environment: 10=unregulated."
    )
    raw    = groq_call(client, sys_p, ctx, temp=0.1, max_tok=400)
    parsed = parse_json(raw)
    if not parsed or not parsed.get("scores"):
        return {"ticker": stock.get("Ticker","?"),
                "scores": {d:5 for d in RADAR_DIMS},
                "explanations": {d:"Analysis unavailable." for d in RADAR_DIMS}}
    return {"ticker": stock.get("Ticker","?"),
            "scores": parsed.get("scores",{}),
            "explanations": parsed.get("explanations",{})}

def radar_chart(data):
    """
    FIX: Use bright distinct colors so each stock is clearly visible.
    Previous version had near-identical muted colors making lines indistinguishable.
    """
    COLORS = [
        "#00d4ff",   # bright cyan
        "#ff6b35",   # bright orange
        "#7fff00",   # chartreuse
        "#ff3cac",   # hot pink
        "#ffd700",   # gold
        "#a855f7",   # purple
        "#00ff88",   # mint green
        "#ff4757",   # red
    ]
    fig = go.Figure()
    for i, rd in enumerate(data):
        color  = COLORS[i % len(COLORS)]
        scores = [rd["scores"].get(d, 5) for d in RADAR_DIMS] + \
                 [rd["scores"].get(RADAR_DIMS[0], 5)]
        # Build hover text showing score + explanation per dimension
        hover  = [
            f"<b>{rd['ticker']}</b><br>"
            f"{d}: {rd['scores'].get(d,5)}/10<br>"
            f"<i>{rd['explanations'].get(d,'')[:80]}</i>"
            for d in RADAR_DIMS
        ] + [f"<b>{rd['ticker']}</b><br>{RADAR_DIMS[0]}: {rd['scores'].get(RADAR_DIMS[0],5)}/10"]

        fig.add_trace(go.Scatterpolar(
            r=scores, theta=RADAR_DIMS + [RADAR_DIMS[0]],
            fill="toself", name=rd["ticker"],
            line=dict(color=color, width=3),
            fillcolor=color, opacity=0.15,
            hovertext=hover, hoverinfo="text",
        ))
    # FIX: Set polar bgcolor to white so the bright stock colors are clearly
    # visible against the circle background instead of disappearing into the
    # dark theme. Grid lines are light grey, tick labels are dark for contrast.
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,10],
                            tickvals=[2,4,6,8,10],
                            gridcolor="#cccccc", color="#333333",
                            tickfont=dict(size=9,color="#333333"),
                            linecolor="#aaaaaa"),
            angularaxis=dict(gridcolor="#cccccc",
                             tickfont=dict(size=11,color="#111111"),
                             linecolor="#aaaaaa"),
            bgcolor="#ffffff",   # FIX: white circle background
        ),
        paper_bgcolor="#0c0e16", plot_bgcolor="#0c0e16",
        font=dict(color="#c0c8e0", size=11),
        legend=dict(bgcolor="#07080d", bordercolor="#1a1f2e", borderwidth=1,
                    font=dict(color="#ffffff")),
        margin=dict(l=70,r=70,t=40,b=40), height=500,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#2d3a56",
                        font=dict(color="#111111")),
    )
    return fig

#  News sentiment functions 

def fetch_news(ticker, company_name, days_back=7, max_articles=5):
    """
    FIX: Search by full company name, not just ticker.
    "LI" matched India festivals. "Li Auto" is specific.
    """
    if not NEWS_API_KEY: return []
    name = company_name
    for s in [", Inc.", " Inc.", ", Corp.", " Corp.", ", Ltd.", " Ltd.",
              " Holdings", " Group", " LLC"]:
        name = name.replace(s, "")
    name  = name.strip()
    query = f'"{name}" stock'
    from_dt = (datetime.now()-timedelta(days=days_back)).strftime("%Y-%m-%d")
    try:
        r = requests.get("https://newsapi.org/v2/everything",
            params={"q":query,"from":from_dt,"sortBy":"relevancy",
                    "language":"en","pageSize":max_articles,"apiKey":NEWS_API_KEY},
            timeout=15)
        return r.json().get("articles",[]) if r.status_code==200 else []
    except:
        return []

def analyse_article(client, ticker, company_name, title, desc, content):
    """
    FIX: LLM now explicitly judges relevance.
    Irrelevant articles get sentiment=0 so they don't inflate averages.
    """
    text  = f"Title: {title}\nDescription: {desc}\nContent: {content[:1000]}"
    sys_p = (
        f"You are a financial news analyst reviewing an article potentially about "
        f"{company_name} ({ticker}).\n"
        "Determine if this article is genuinely about this company. "
        "Return ONLY this JSON:\n"
        '{"is_relevant":true,"relevance_reason":"why this is or is not about the company",'
        '"sentiment_score":70,"recent_developments":"key event",'
        '"risks_challenges":"main risk or None mentioned",'
        '"future_outlook":"forward-looking statement"}\n'
        "If not relevant, set is_relevant=false and sentiment_score=0. "
        "Sentiment: 0-20=very negative, 21-40=negative, 41-60=neutral, "
        "61-80=positive, 81-100=very positive."
    )
    try:
        raw    = groq_call(client, sys_p, text, temp=0.1, max_tok=350)
        parsed = parse_json(raw)
        if not parsed: raise ValueError("empty")
        return {
            "is_relevant":         bool(parsed.get("is_relevant", True)),
            "relevance_reason":    parsed.get("relevance_reason",""),
            "sentiment_score":     int(parsed.get("sentiment_score",50)),
            "recent_developments": parsed.get("recent_developments",""),
            "risks_challenges":    parsed.get("risks_challenges",""),
            "future_outlook":      parsed.get("future_outlook",""),
        }
    except:
        return {"is_relevant":True,"relevance_reason":"Parse error.",
                "sentiment_score":50,"recent_developments":"",
                "risks_challenges":"","future_outlook":""}

def run_sentiment(stocks, client, days_back=7, max_articles=5):
    rows = []
    for stock in stocks:
        ticker   = stock.get("Ticker","?")
        name     = stock.get("Name", ticker)
        articles = fetch_news(ticker, name, days_back, max_articles)
        if not articles:
            rows.append({"Ticker":ticker,"Company":name,"Title":"No articles found",
                         "Provider":"","Published":"","URL":"",
                         "Is Relevant":False,"Relevance Reason":"No articles found.",
                         "Sentiment Score":0,"Recent Developments":"",
                         "Risks/Challenges":"","Future Outlook":""})
            continue
        for art in articles:
            title = art.get("title","") or ""
            if len(title)<10: continue
            src   = art.get("source",{})
            prov  = src.get("name","") if isinstance(src,dict) else ""
            a     = analyse_article(client, ticker, name, title,
                                    art.get("description","") or "",
                                    art.get("content","") or "")
            rows.append({"Ticker":ticker,"Company":name,"Title":title,
                         "Provider":prov,
                         "Published":art.get("publishedAt","")[:10],
                         "URL":art.get("url",""),
                         "Is Relevant":a["is_relevant"],
                         "Relevance Reason":a["relevance_reason"],
                         "Sentiment Score":a["sentiment_score"],
                         "Recent Developments":a["recent_developments"],
                         "Risks/Challenges":a["risks_challenges"],
                         "Future Outlook":a["future_outlook"]})
        time.sleep(0.3)
    return pd.DataFrame(rows)

def sentiment_ranking(client, avg_df):
    if avg_df.empty: return ""
    text = "\n".join(
        f"{r['Ticker']} ({r['Company']}): {r['Avg Sentiment']}/100"
        for _,r in avg_df.iterrows()
    )
    return groq_call(client,
        "Financial news analyst. Rank these stocks from best to worst based on "
        "average news sentiment. Numbered list, one paragraph per stock explaining "
        "what their score suggests about recent news coverage and market perception.",
        f"Scores:\n{text}", temp=0.3, max_tok=1200)

# SEC 10-Q functions 

def get_cik(ticker):
    try:
        r = requests.get("https://www.sec.gov/files/company_tickers.json",
                         headers=SEC_HEADERS, timeout=15)
        for entry in r.json().values():
            if entry.get("ticker","").upper() == ticker.upper():
                return str(entry["cik_str"]).zfill(10)
    except: pass
    return None

def get_10q(ticker):
    cik = get_cik(ticker)
    if not cik: return None, None, None
    try:
        r    = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                            headers=SEC_HEADERS, timeout=15)
        subs = r.json()
        rec  = subs.get("filings",{}).get("recent",{})
        forms   = rec.get("form",[])
        acc_nos = rec.get("accessionNumber",[])
        dates   = rec.get("filingDate",[])
        for i,form in enumerate(forms):
            if form == "10-Q":
                acc  = acc_nos[i].replace("-","")
                url  = (f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{acc}/{acc_nos[i]}-index.htm")
                link = (f"https://www.sec.gov/cgi-bin/browse-edgar?"
                        f"action=getcompany&CIK={cik}&type=10-Q&dateb=&owner=include&count=5")
                return url, dates[i], link
    except: pass
    return None, None, None

def extract_text(index_url, max_chars=10000):
    """
    FIX: Increased max_chars from 8000 to 10000 and added section
    extraction to target MD&A and financial highlights instead of
    blindly taking the first N chars (often just cover page boilerplate).
    """
    try:
        r    = requests.get(index_url, headers=SEC_HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, "lxml")
        prim = None
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells)>=3:
                link = cells[2].find("a") if len(cells)>2 else None
                if link and link.get("href","").endswith((".htm",".html")):
                    prim = f"https://www.sec.gov{link['href']}"
                    break
        if not prim:
            for a in soup.find_all("a",href=True):
                if a["href"].endswith((".htm",".html")):
                    prim = f"https://www.sec.gov{a['href']}"
                    break
        if not prim: return "Primary document not found."
        doc  = requests.get(prim, headers=SEC_HEADERS, timeout=20)
        text = BeautifulSoup(doc.text,"lxml").get_text(" ",strip=True)
        text = re.sub(r"\s+"," ",text).strip()

        # FIX: Extract the most informative sections rather than the cover
        # page. MD&A contains the key financial discussion and guidance.
        section_keywords = [
            "management.*discussion.*analysis",
            "results of operations",
            "financial condition",
            "liquidity and capital",
            "risk factors",
            "financial highlights",
            "overview",
        ]
        best_start = 0
        for kw in section_keywords:
            match = re.search(kw, text[:50000], re.IGNORECASE)
            if match:
                best_start = max(0, match.start() - 100)
                break

        relevant_text = text[best_start : best_start + max_chars]
        if len(relevant_text) < 500:
            relevant_text = text[:max_chars]

        return relevant_text
    except Exception as e:
        return f"Error: {e}"

def _heuristic_10q_scores(stock_meta):
    """
    Fallback: estimate 10-Q-style scores from Yahoo Finance metrics when
    the LLM fails to parse the actual filing text. Ensures the heatmap
    always shows meaningful data instead of all zeros.
    """
    def safe_float(v, default=0.0):
        try:    return float(v)
        except: return default

    rev_g  = safe_float(stock_meta.get("Revenue Growth",  0))
    earn_g = safe_float(stock_meta.get("Earnings Growth", 0))
    gm     = safe_float(stock_meta.get("Gross Margins",   0))
    ebitda = safe_float(stock_meta.get("EBITDA Margins",  0))
    wk52   = safe_float(stock_meta.get("52 Week Change",  0))

    perf   = min(10, max(1, round(5 + rev_g * 10 + gm * 3)))
    growth = min(10, max(1, round(5 + (earn_g + rev_g) * 5)))
    risk   = min(10, max(1, round(5 + wk52 * 3 + ebitda * 5)))
    edge   = min(10, max(1, round(3 + gm * 8)))

    return {
        "Performance":      perf,
        "Growth Potential": growth,
        "Risks":            risk,
        "Competitive Edge": edge,
    }

def analyse_10q(client, ticker, text, stock_meta=None):
    """
    FIX 2 - SEC 10-Q HEATMAP ALL ZEROS:
    Three-attempt strategy so the heatmap is never all zeros:

    Attempt 1: Detailed prompt with a concrete filled-in example,
               max_tokens=800 (was 500, caused JSON truncation).
    Attempt 2: Simpler retry prompt if attempt 1 JSON parse fails.
    Attempt 3: Heuristic scores from Yahoo Finance metrics as fallback.
               Clearly marked as estimated so users can distinguish from
               actual SEC filing analysis.

    FIX: Root cause of parse failure was that the scoring instructions
    were passed as system_prompt while the filing text was user_message.
    The LLM was anchoring on the example numbers in the system prompt
    (6,7,8,7) and returning them unchanged regardless of the filing.
    New approach: system_prompt is a strict JSON-only instruction,
    user_message combines the full scoring guide + the filing text
    so the LLM reads both at once and generates scores from the actual text.
    """
    # System prompt: pure JSON formatting instruction only (no example numbers)
    # This prevents the LLM from anchoring on placeholder scores.
    _sys = (
        "You are a financial analyst. You MUST return ONLY valid JSON. "
        "No markdown fences, no text before or after the JSON. "
        "Return exactly this structure with integer scores 1-10:\n"
        '{"scores":{"Performance":0,"Growth Potential":0,"Risks":0,"Competitive Edge":0},'
        '"explanations":{"Performance":"","Growth Potential":"","Risks":"","Competitive Edge":""},'
        '"overall_summary":""}'
    )

    def _user_v1(ticker, filing_text):
        """
        Primary attempt: full scoring guide + filing text in one message.
        Putting the guide and the text together forces the LLM to derive
        scores FROM the text rather than from example numbers in the prompt.
        """
        return (
            f"Analyse this SEC 10-Q filing for {ticker} and fill in the JSON scores.\n\n"
            "SCORING GUIDE (replace the 0s with your scores based ONLY on the filing text below):\n"
            "  Performance 1-10:      10=exceptional revenue/margin results, 5=average, 1=declining\n"
            "  Growth Potential 1-10: 10=strong forward guidance/expansion, 5=stable, 1=declining\n"
            "  Risks 1-10:            10=very safe (few/minor risks), 5=moderate, 1=severe red flags\n"
            "  Competitive Edge 1-10: 10=dominant moat/patents, 5=competitive market, 1=commoditised\n\n"
            "CRITICAL: Every company must get DIFFERENT scores based on its specific filing.\n"
            "Read the filing text below and extract:\n"
            "- Actual revenue growth % and gross margin % for Performance\n"
            "- Management guidance language for Growth Potential\n"
            "- Specific risks named (supply chain, competition, debt, litigation) for Risks\n"
            "- Patents, customer lock-in, market share statements for Competitive Edge\n\n"
            "Put specific numbers/quotes from the filing in the explanation fields.\n\n"
            f"10-Q FILING TEXT FOR {ticker}:\n{filing_text[:7500]}"
        )

    def _user_v2(ticker, filing_text):
        """Retry attempt: shorter and simpler."""
        return (
            f"SEC 10-Q for {ticker}. Score 1-10. Replace the 0s in the JSON.\n"
            "Performance=revenue/margin quality. Growth=forward guidance. "
            "Risks=10 means very safe, 1 means very risky. Edge=competitive moat.\n"
            f"Filing text: {filing_text[:5000]}"
        )

    # --- Attempt 1: full guide + filing text as user message ---
    raw    = groq_call(client, _sys, _user_v1(ticker, text), temp=0.1, max_tok=800)
    parsed = parse_json(raw)
    if parsed and parsed.get("scores") and len(parsed["scores"]) >= 4:
        scores = parsed["scores"]
        # Accept if all 4 scores are valid positive integers
        if all(isinstance(scores.get(d),(int,float)) and scores.get(d,0) > 0
               for d in SEC_DIMS):
            return {"ticker":ticker, "scores":scores,
                    "explanations":parsed.get("explanations",{}),
                    "overall_summary":parsed.get("overall_summary","")}

    # --- Attempt 2 (retry): simpler combined message ---
    time.sleep(1)
    raw2    = groq_call(client, _sys, _user_v2(ticker, text), temp=0.1, max_tok=600)
    parsed2 = parse_json(raw2)
    if parsed2 and parsed2.get("scores") and len(parsed2["scores"]) >= 4:
        scores2 = parsed2["scores"]
        if all(isinstance(scores2.get(d),(int,float)) and scores2.get(d,0) > 0
               for d in SEC_DIMS):
            return {"ticker":ticker, "scores":scores2,
                    "explanations":parsed2.get("explanations",{}),
                    "overall_summary":parsed2.get("overall_summary","")}

    # --- Attempt 3: heuristic fallback - never shows all zeros ---
    heuristic = _heuristic_10q_scores(stock_meta or {})
    return {
        "ticker":          ticker,
        "scores":          heuristic,
        "explanations":    {
            d: "Estimated from Yahoo Finance metrics (10-Q text parse failed)."
            for d in SEC_DIMS
        },
        "overall_summary": (
            "Note: Scores estimated from Yahoo Finance metrics because the SEC "
            "filing text could not be parsed. Re-run or check the filing manually."
        ),
    }

# Recommendation function 

def generate_recommendation(client, stock, sentiment=None, sec=None):
    def fp(v):
        try: return f"{float(v)*100:.1f}%"
        except: return "N/A"
    ctx = (
        f"Company: {stock.get('Name')} ({stock.get('Ticker')})\n"
        f"Sector: {stock.get('Sector')}\n"
        f"Earnings Growth: {fp(stock.get('Earnings Growth'))}\n"
        f"Revenue Growth: {fp(stock.get('Revenue Growth'))}\n"
        f"Gross Margins: {fp(stock.get('Gross Margins'))}\n"
        f"EBITDA Margins: {fp(stock.get('EBITDA Margins'))}\n"
        f"52-Week Change: {fp(stock.get('52 Week Change'))}\n"
    )
    if sentiment is not None:
        ctx += f"News Sentiment Score: {sentiment}/100\n"
    if sec:
        ctx += (f"SEC Performance: {sec.get('Performance','N/A')}/10\n"
                f"SEC Growth Potential: {sec.get('Growth Potential','N/A')}/10\n"
                f"SEC Risk Score: {sec.get('Risks','N/A')}/10 (10=safe)\n"
                f"SEC Competitive Edge: {sec.get('Competitive Edge','N/A')}/10\n")
    return groq_call(client,
        "Senior portfolio manager at a quantitative investment fund. "
        "Start with BUY / HOLD / SELL. Then 2-3 sentences explaining WHY "
        "with specific numbers. End with 1 sentence on the main risk. "
        "Be direct. No hedging. No generic statements.",
        f"Stock data:\n{ctx}", temp=0.2, max_tok=300)

# MAIN APP 

def main():
    embed_model      = load_embed()
    pinecone_idx, ps = load_pinecone()
    groq_client      = load_groq()

    # Session state
    defaults = {"stocks":[], "expanded":"", "radar":[], "sent_df":None, "sec":[]}
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Header
    st.markdown('<div class="page-title">Stock Intelligence Platform</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">RAG stock search | Market Radar | News Sentiment | '
        f'SEC 10-Q | Recommendations &nbsp;|&nbsp; '
        f'Pinecone: <b>{ps.total_vector_count:,}</b> stocks indexed</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    # SECTION 1 - STOCK SEARCH
    st.markdown('<div class="sec-title">Stock Search</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Pipeline:</strong> Your query is expanded '
        'by the LLM into a detailed paragraph using semiconductor/tech vocabulary '
        '(GPU, CPU, fabless, foundry, IP licensing, etc.) so broad industry queries '
        'like "semiconductors" correctly surface NVDA, AMD, TSM, AVGO. '
        'The paragraph is embedded into a 768-dimensional vector and compared against '
        '10,000+ stock descriptions in Pinecone using cosine similarity. '
        'Yahoo Finance provides the financial metrics shown on each card.</div>',
        unsafe_allow_html=True)

    query = st.text_area("Describe the kind of companies you are looking for:",
                         placeholder="e.g. semiconductors | data center infrastructure | "
                                     "cloud SaaS with recurring revenue | biotech drug discovery",
                         height=80, label_visibility="visible")

    c1,c2,c3,c4 = st.columns([2,2,2,2])
    with c1: num    = st.slider("Number of stocks", 4, 10, 6)
    with c2: sector = st.selectbox("Sector filter", SECTORS)
    with c3: start  = st.date_input("Chart start", value=pd.to_datetime("2023-01-01"))
    with c4: end    = st.date_input("Chart end",   value=pd.to_datetime("2024-09-04"))

    if st.button("Find Stocks", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            sf = None if sector == "All Sectors" else sector
            with st.spinner("Expanding query + searching Pinecone..."):
                stocks, expanded = search_stocks(
                    query, pinecone_idx, embed_model, groq_client,
                    sector=sf, top_k=num)
            st.session_state.stocks   = stocks
            st.session_state.expanded = expanded
            st.session_state.radar    = []
            st.session_state.sent_df  = None
            st.session_state.sec      = []

    stocks = st.session_state.stocks
    if not stocks:
        st.info("Enter a search query to find stocks. Results power all sections below.")
        return

    with st.expander("LLM-expanded query (what was sent to Pinecone)"):
        st.write(st.session_state.expanded)

    # CSV download of stock results
    def make_csv(sl):
        def pct(v):
            try: return round(float(v)*100,2)
            except: return None
        rows = [{"Ticker":s.get("Ticker"),"Name":s.get("Name"),
                 "Sector":s.get("Sector"),"Industry":s.get("Industry"),
                 "Country":s.get("Country"),"Website":s.get("Website"),
                 "Earnings Growth %":pct(s.get("Earnings Growth")),
                 "Revenue Growth %":pct(s.get("Revenue Growth")),
                 "Gross Margins %":pct(s.get("Gross Margins")),
                 "EBITDA Margins %":pct(s.get("EBITDA Margins")),
                 "52 Week Change %":pct(s.get("52 Week Change")),
                 "Relevance Score":s.get("_score")} for s in sl]
        return pd.DataFrame(rows).to_csv(index=False).encode()

    st.download_button("Download stock results as CSV", make_csv(stocks),
                       "stocks.csv", "text/csv")

    # Stock cards
    st.markdown(f'<div class="sec-title">Found {len(stocks)} Relevant Stocks</div>',
                unsafe_allow_html=True)
    for i in range(0, len(stocks), 2):
        ca, cb = st.columns(2, gap="medium")
        for col, idx in [(ca,i),(cb,i+1)]:
            if idx >= len(stocks): break
            s = stocks[idx]
            m = "".join([metric_box("Earnings Growth", s.get("Earnings Growth")),
                         metric_box("Revenue Growth",  s.get("Revenue Growth")),
                         metric_box("Gross Margins",   s.get("Gross Margins")),
                         metric_box("EBITDA Margins",  s.get("EBITDA Margins")),
                         metric_box("52W Change",      s.get("52 Week Change"))])
            w = s.get("Website","")
            wl = f'<a class="co-link" href="{w}" target="_blank">{w}</a>' if w else ""
            col.markdown(f"""
<div class="stock-card">
  <span class="score-pill">match {s.get('_score',0):.3f}</span>
  <div class="ticker-tag">{s.get('Ticker','?')}</div>
  <div class="co-name">{s.get('Name','?')} &middot; {s.get('Sector','')} &middot; {s.get('Country','')}</div>
  <div class="co-summary">{s.get('Business Summary','')[:260]}...</div>
  {wl}
  <div class="metrics-grid">{m}</div>
</div>""", unsafe_allow_html=True)

    # Price chart
    # FIX: Use hovermode="closest" (not "x unified") so hovering shows only
    # the stock near the cursor, not all stocks simultaneously.
    st.markdown('<div class="sec-title">Stock Price Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Data:</strong> Yahoo Finance historical prices '
        '(yfinance). Normalized to % change from start date so stocks on different '
        'price scales can be fairly compared. Hover over a line to see that stock\'s '
        'exact percentage change on that date.</div>', unsafe_allow_html=True)

    tickers_list = [s.get("Ticker","") for s in stocks if s.get("Ticker")]
    selected_px  = st.multiselect("Select stocks to chart:", tickers_list, default=tickers_list)

    if selected_px:
        with st.spinner("Downloading prices..."):
            raw = yf.download(selected_px, start=str(start), end=str(end),
                              progress=False, auto_adjust=True)
        if not raw.empty:
            prices = (raw[["Close"]].rename(columns={"Close":selected_px[0]})
                      if len(selected_px)==1 else raw["Close"].dropna(axis=1,how="all"))
            norm   = ((prices/prices.iloc[0])-1)*100
            melted = norm.reset_index().melt(id_vars="Date",var_name="Stock",value_name="% Change")

            fig = px.line(melted, x="Date", y="% Change", color="Stock",
                          title="Normalized Price History (% Change from Start Date)",
                          template="plotly_dark",
                          labels={"% Change":"% Change from Start"})
            fig.update_traces(line=dict(width=2))
            fig.update_layout(
                paper_bgcolor="#0c0e16", plot_bgcolor="#07080d",
                font_color="#8a9ab8", height=400,
                # FIX: "closest" shows only the nearest stock on hover
                hovermode="closest",
                legend=dict(bgcolor="#07080d", bordercolor="#1a1f2e"),
            )
            fig.add_hline(y=0, line_dash="dot", line_color="#1a1f2e")
            st.plotly_chart(fig, use_container_width=True)

    # AI comparison
    st.markdown('<div class="sec-title">AI Financial Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Data:</strong> Yahoo Finance metrics sent to '
        'Groq LLaMA. Identifies each stock\'s key strength or weakness with specific '
        'numbers and provides a clear investor takeaway.</div>', unsafe_allow_html=True)
    with st.spinner("Generating comparison..."):
        st.markdown(comparison_summary(groq_client, stocks))

    # SECTION 2 - MARKET TREND RADAR
    st.markdown('<div class="sec-title">Market Trend Radar</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Pipeline:</strong> Each stock is scored 0-10 on '
        '6 investment dimensions by the LLM using Yahoo Finance metrics + business description. '
        'Scores are not hardcoded - the LLM analyses each company individually. '
        'Hover over a point to see the score and explanation. '
        'Market Competition 10 = low competition (dominant player). '
        'Regulatory Environment 10 = low regulatory risk.</div>', unsafe_allow_html=True)

    sel_radar = st.multiselect("Select stocks for radar:", tickers_list,
                               default=tickers_list, key="r_sel")

    if st.button("Generate Radar Scores", type="primary"):
        sel_s = [s for s in stocks if s.get("Ticker") in sel_radar]
        rd    = []
        prog  = st.progress(0)
        for i,s in enumerate(sel_s):
            prog.progress((i+1)/len(sel_s), f"Scoring {s.get('Ticker')}...")
            result = score_radar(groq_client, s)
            rd.append(result)
        prog.empty()
        st.session_state.radar = rd

    radar_data = [r for r in st.session_state.radar if r["ticker"] in sel_radar]

    if radar_data:
        # FIX: only show chart if scores are actually different from all-5
        all_same = all(
            list(r["scores"].values()) == [5]*6
            for r in radar_data
        )
        if all_same:
            st.warning("Radar scoring returned default values. "
                       "The LLM JSON response may have failed. Try regenerating.")
        else:
            st.markdown("**Market Trend Radar - All Selected Stocks**")
            st.plotly_chart(radar_chart(radar_data), use_container_width=True)

        st.markdown("**Factor Explanations**")
        for rd in radar_data:
            with st.expander(f"Explanations for {rd['ticker']}"):
                for dim in RADAR_DIMS:
                    score = rd["scores"].get(dim, 5)
                    expl  = rd["explanations"].get(dim, "")
                    col   = score_color(score)
                    st.markdown(
                        f'<div class="exp-row"><span class="exp-dim">{dim}</span> '
                        f'<span style="color:{col};font-family:monospace;">[{score}/10]</span>: '
                        f'{expl}</div>', unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Radar Scores' to build the chart.")

    # SECTION 3 - NEWS SENTIMENT
    st.markdown('<div class="sec-title">News Sentiment Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Pipeline:</strong> NewsAPI is queried using the '
        'full company name (not ticker symbol) to avoid irrelevant results. '
        'The LLM reads each article, judges if it is genuinely about the company, '
        'then extracts: sentiment score (0-100), recent developments, risks, and future outlook. '
        'Only relevant articles are included in the average. '
        'Download the CSV to see all article details including provider, publish date, '
        'relevance reason, and full extracted insights.</div>', unsafe_allow_html=True)

    if not NEWS_API_KEY:
        st.error("NEWS_API_KEY missing. Add to Colab Secrets.")
    else:
        sn1, sn2 = st.columns(2)
        with sn1: days  = st.slider("Days of history", 3, 30, 7, key="s_days")
        with sn2: arts  = st.slider("Max articles per stock", 3, 10, 5, key="s_arts")

        if st.button("Fetch and Analyse News", type="primary"):
            with st.spinner("Fetching news + LLM sentiment analysis..."):
                df = run_sentiment(stocks, groq_client, days, arts)
            st.session_state.sent_df = df

        df = st.session_state.sent_df

        if df is not None and not df.empty:
            # Average using only relevant articles
            rel_df = df[df["Is Relevant"]==True]
            if rel_df.empty: rel_df = df

            avg_df = (rel_df.groupby(["Ticker","Company"])["Sentiment Score"]
                      .mean().round(1).reset_index()
                      .rename(columns={"Sentiment Score":"Avg Sentiment"})
                      .sort_values("Avg Sentiment", ascending=False))

            # Bar chart
            st.markdown("**Average Sentiment Score by Stock (relevant articles only)**")
            fig_bar = px.bar(avg_df, x="Ticker", y="Avg Sentiment",
                             text="Avg Sentiment",
                             color="Avg Sentiment",
                             color_continuous_scale=["#f87171","#facc15","#4ade80"],
                             range_color=[0,100], template="plotly_dark",
                             labels={"Avg Sentiment":"Avg Sentiment (0-100)"})
            fig_bar.update_layout(paper_bgcolor="#0c0e16", plot_bgcolor="#07080d",
                                  font_color="#8a9ab8", height=360,
                                  coloraxis_showscale=False,
                                  yaxis=dict(range=[0,115]))
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Ranking
            with st.spinner("Generating ranking analysis..."):
                st.markdown("**Stock Rankings Based on News Sentiment:**")
                st.markdown(sentiment_ranking(groq_client, avg_df))

            # FIX: Article table showing all stocks (not duplicated)
            # Previous version showed same articles for every ticker.
            # Now grouped properly by ticker with a full data table.
            st.markdown("**News Articles Detail**")

            # Show complete data table with all columns
            display_cols = ["Ticker","Company","Title","Provider","Published",
                            "Is Relevant","Relevance Reason","Sentiment Score",
                            "Recent Developments","Risks/Challenges","Future Outlook"]
            avail_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[avail_cols], use_container_width=True, height=400)

            # Article cards grouped by ticker
            for ticker in df["Ticker"].unique():
                t_df = df[df["Ticker"]==ticker]
                rel  = t_df[t_df["Is Relevant"]==True]
                avg  = rel["Sentiment Score"].mean() if not rel.empty else 0
                with st.expander(
                    f"{ticker} - {len(t_df)} articles | "
                    f"{len(rel)} relevant | Avg: {avg:.1f}/100"
                ):
                    for _,row in t_df.iterrows():
                        sc    = row["Sentiment Score"]
                        is_r  = row.get("Is Relevant", True)
                        color = "#4ade80" if sc>=61 else ("#facc15" if sc>=41 else "#f87171")
                        if not is_r: color = "#374151"
                        pub   = str(row.get("Published",""))[:10]
                        prov  = row.get("Provider","")
                        url   = row.get("URL","")
                        link  = (f'<a href="{url}" target="_blank" '
                                 f'style="color:#3b82f6;font-size:0.73rem;">Read article</a>'
                                 if url else "")
                        st.markdown(f"""
<div class="news-card">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <span class="news-title">{row.get('Title','')}</span>
    <span style="color:{color};font-family:monospace;font-weight:700;font-size:0.85rem;white-space:nowrap;margin-left:12px;">
      {sc}/100
    </span>
  </div>
  <div style="color:#374151;font-size:0.72rem;margin:3px 0 6px;">
    {prov} &nbsp;|&nbsp; {pub} &nbsp;|&nbsp; {link}
    &nbsp;|&nbsp; {'Relevant' if is_r else 'Not relevant to company'}
  </div>
  <div class="news-rel">Why relevant: {row.get('Relevance Reason','')}</div>
  <div class="news-val"><span class="news-field">Recent developments:</span> {row.get('Recent Developments','')}</div>
  <div class="news-val"><span class="news-field">Risks / Challenges:</span> {row.get('Risks/Challenges','')}</div>
  <div class="news-val"><span class="news-field">Future outlook:</span> {row.get('Future Outlook','')}</div>
</div>""", unsafe_allow_html=True)

            # CSV download
            st.download_button(
                "Download news sentiment data as CSV",
                df.to_csv(index=False).encode(),
                "news_sentiment.csv", "text/csv")

    # SECTION 4 - SEC 10-Q ANALYSIS
    st.markdown('<div class="sec-title">SEC 10-Q Filing Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Pipeline:</strong> SEC EDGAR is queried for each '
        'company\'s most recent 10-Q quarterly filing (mandatory for all US-listed companies). '
        'The filing HTML is downloaded, the MD&A and financial highlights sections are extracted, '
        'then scored by the LLM on: Performance (current results), Growth Potential (forward '
        'guidance), Risks (10=very low risk), and Competitive Edge. If the LLM parse fails, '
        'scores are estimated from Yahoo Finance metrics so the heatmap is never empty. '
        'A heatmap compares all stocks at a glance.</div>', unsafe_allow_html=True)

    if st.button("Fetch and Analyse 10-Q Filings", type="primary"):
        results  = []
        prog     = st.progress(0)
        for i,stock in enumerate(stocks):
            ticker = stock.get("Ticker","?")
            prog.progress((i+1)/len(stocks), f"Analysing {ticker}...")
            url, date, link = get_10q(ticker)
            if url:
                text   = extract_text(url)
                # FIX: Pass stock metadata for heuristic fallback
                result = analyse_10q(groq_client, ticker, text, stock_meta=stock)
                result["filing_date"] = date or "Unknown"
                result["filing_link"] = link
            else:
                # No 10-Q found: use heuristic scores so heatmap is never all zeros
                heuristic = _heuristic_10q_scores(stock)
                result = {
                    "ticker":          ticker,
                    "filing_date":     "N/A",
                    "filing_link":     None,
                    "scores":          heuristic,
                    "explanations":    {
                        d: "Estimated from Yahoo Finance metrics (no 10-Q found on EDGAR)."
                        for d in SEC_DIMS
                    },
                    "overall_summary": (
                        "No 10-Q filing found on SEC EDGAR. Scores are estimated from "
                        "Yahoo Finance financial metrics as a fallback."
                    ),
                }
            results.append(result)
            time.sleep(0.8)
        prog.empty()
        st.session_state.sec = results

    sec = st.session_state.sec
    if sec:
        # Heatmap - FIX: use proper color scale with visible range
        st.markdown("**10-Q Score Heatmap (0-10, Risks: 10 = very safe)**")
        heat_df = pd.DataFrame(
            {d: [r["scores"].get(d,0) for r in sec] for d in SEC_DIMS},
            index=[r["ticker"] for r in sec],
        )
        fig_h = px.imshow(heat_df, text_auto=True, aspect="auto",
                          color_continuous_scale=[[0,"#07080d"],[0.3,"#1e3a5f"],
                                                  [0.6,"#3b82f6"],[1.0,"#a78bfa"]],
                          zmin=0, zmax=10, template="plotly_dark",
                          title="10-Q Scores (0-10) | Risks: 10 = very low risk")
        fig_h.update_layout(
            paper_bgcolor="#0c0e16", plot_bgcolor="#07080d",
            font_color="#8a9ab8",
            height=max(280, len(sec)*55+100),
            coloraxis_colorbar=dict(title="Score",tickvals=[0,2,4,6,8,10]))
        fig_h.update_traces(textfont=dict(color="white",family="IBM Plex Mono",size=13))
        st.plotly_chart(fig_h, use_container_width=True)

        # Filing detail cards
        st.markdown("**Filing Details by Company**")
        for r in sec:
            filed = r.get("filing_date","N/A")
            link  = r.get("filing_link")
            link_html = (f'[View 10-Q filings on SEC EDGAR]({link})' if link else "")
            with st.expander(f"{r['ticker']} - Filed: {filed}", expanded=False):
                if link:
                    st.markdown(link_html)
                st.markdown("**Key Insights from 10-Q:**")
                for dim in SEC_DIMS:
                    score = r["scores"].get(dim,0)
                    expl  = r["explanations"].get(dim,"")
                    st.markdown(bar_html(dim, score, expl), unsafe_allow_html=True)
                summary = r.get("overall_summary","")
                if summary:
                    st.markdown(
                        f'<div style="margin-top:12px;padding-top:12px;'
                        f'border-top:1px solid #1a1f2e;color:#8a9ab8;'
                        f'font-size:0.82rem;line-height:1.65;">{summary}</div>',
                        unsafe_allow_html=True)

    # SECTION 5 - STOCK RECOMMENDATIONS
    st.markdown('<div class="sec-title">Stock Recommendations</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="pipeline-info"><strong>Pipeline:</strong> Synthesizes all available data - '
        'Yahoo Finance metrics, news sentiment scores, and SEC 10-Q scores - into a '
        'BUY / HOLD / SELL recommendation with specific reasoning for each stock. '
        'Run Sections 3 and 4 first for the most comprehensive recommendation.</div>',
        unsafe_allow_html=True)

    if st.button("Generate Recommendations for All Stocks", type="primary"):
        # Build lookup maps for sentiment and SEC data
        sent_map = {}
        if st.session_state.sent_df is not None and not st.session_state.sent_df.empty:
            df_s = st.session_state.sent_df
            rel  = df_s[df_s["Is Relevant"]==True] if "Is Relevant" in df_s.columns else df_s
            for _, grp in (rel if not rel.empty else df_s).groupby("Ticker"):
                t = grp["Ticker"].iloc[0]
                sent_map[t] = round(grp["Sentiment Score"].mean(), 1)

        sec_map = {}
        for r in st.session_state.sec:
            sec_map[r["ticker"]] = r["scores"]

        prog = st.progress(0)
        for i, stock in enumerate(stocks):
            ticker = stock.get("Ticker","?")
            prog.progress((i+1)/len(stocks), f"Recommending {ticker}...")
            rec = generate_recommendation(
                groq_client, stock,
                sentiment=sent_map.get(ticker),
                sec=sec_map.get(ticker),
            )
            # Color-code the recommendation card
            if "BUY" in rec[:10]:
                border_color = "#4ade80"
            elif "SELL" in rec[:10]:
                border_color = "#f87171"
            else:
                border_color = "#facc15"

            st.markdown(f"""
<div class="rec-card" style="border-left:4px solid {border_color};">
  <div class="rec-ticker">{ticker} - {stock.get('Name','')}</div>
  <div class="rec-text">{rec}</div>
</div>""", unsafe_allow_html=True)

        prog.empty()

    st.markdown("---")
    st.markdown(
        '<div style="color:#1f2937;font-size:0.74rem;text-align:center;">'
        'Data: Yahoo Finance | Pinecone | NewsAPI | SEC EDGAR | Groq LLaMA'
        '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

