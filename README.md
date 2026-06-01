# Financial Analysis & Automation with LLMs

## Overview

This project is a financial intelligence platform that combines large language models, vector similarity search, real-time financial data, regulatory filings and news sentiment analysis into a single interactive application. The goal is to automate the kind of deep dive stock research that would normally take a professional analyst hours to complete manuall, and compress it into a workflow that runs in minutes.

The idea: raw financial numbers alone are not enough to make informed investment decisions also need to understand what a company actually does, how the market perceives it right now, what its own management is saying in official filings and how all of those signals combine into a coherent view. This project wires all of those inputs together and uses LLMs to synthesize them into structured, actionable analysis.

The user types a plain-language description of the kind of company they are looking for. For example: "semiconductors", "cloud SaaS with recurring revenue", or "biotech drug discovery". From that single input, the platform executes the following:

1. An LLM expands the short query into a detailed, vocabulary-rich paragraph that mirrors the language found in actual SEC filings and Yahoo Finance company descriptions.
2. That paragraph is converted into a 768-dimensional numerical vector using a sentence embedding model.
3. The vector is compared against over 10,000 pre-indexed stock descriptions stored in Pinecone using cosine similarity. The most semantically similar companies are returned.
4. For each matched company, Yahoo Finance provides live financial metrics including revenue growth, earnings growth, gross margins, EBITDA margins and 52-week price change.
5. The LLM analyses those metrics and generates a comparative financial summary identifying the strongest and weakest stocks in the result set.
6. Each stock is scored across six investment dimensions using the LLM: growth potential, financial health, market competition, innovation, industry trends and regulatory environment.
7. NewsAPI is queried for recent articles about each company. The LLM reads each article, determines whether it is genuinely relevant to that specific company and extracts a structured sentiment score from 0 to 100 along with key insights.
8. SEC EDGAR is queried for each company's most recent 10-Q quarterly filing. The filing HTML is downloaded and the most informative sections (Management Discussion and Analysis, Risk Factors, Financial Highlights) are extracted. The LLM scores the filing on four dimensions: performance, growth potential, risks and competitive edge.
9. All signals are synthesized into a final BUY, HOLD or SELL recommendation for each stock, with specific reasoning tied to actual numbers.

## Features

### Semantic Stock Search with RAG and Vector Embeddings

The stock search system is built on Retrieval-Augmented Generation (RAG) that combines a retrieval step over a large knowledge base with a generation step from an LLM.

During the ingestion phase, every company in the SEC EDGAR tickers list (over 10,000 US-listed companies) is fetched from Yahoo Finance. The business description for each company is converted into a 768-dimensional vector using the `sentence-transformers/all-mpnet-base-v2` model and stored in Pinecone along with the company's financial metadata. This creates a searchable semantic index of the entire US stock market.

At query time, rather than searching by ticker symbol or keyword match the system embeds the user's intent as a vector and finds the companies whose descriptions are most similar in vector space. This means a query like "AI chip designers" can correctly surface NVIDIA, AMD and Ambarella even though none of those words appear literally in the search term, because the semantic meaning of the query aligns with the meaning of their business descriptions.

The critical improvement over a naive embedding approach is LLM query expansion. A raw short query like "semiconductors" produces a weak embedding because it lacks the vocabulary depth found in real company filings. The LLM rewrites the query into a 5-7 sentence paragraph using the exact terminology found in SEC filings and Yahoo Finance descriptions: words like "fabless", "foundry", "ASIC", "IP licensing", "process node", "hyperscaler", and "AI inference accelerator". This dramatically improves cosine similarity matching and ensures that well-known companies in the target sector are reliably surfaced.

Cosine similarity is the mathematical measure used to compare vectors. Two vectors pointing in the same direction in 768-dimensional space have a cosine similarity of 1.0, meaning they describe the same concept. Pinecone uses this metric to rank all 10,000+ stored stock descriptions against the query vector and return the top candidates. Sector filtering is then applied in Python to narrow results if the user has selected a specific sector.

This approach is fundamentally different from using ChatGPT directly. A direct ChatGPT query about semiconductor companies would return a hallucinated or outdated list based on whatever the model learned during training. This system retrieves actual, live companies from a database of real SEC-registered businesses, with real Yahoo Finance financial metrics attached and uses the LLM only to interpret and rank the results rather than generate them from memory.

### Parallel Ingestion of 10,000+ Stocks
<img width="329" height="1616" alt="image" src="https://github.com/user-attachments/assets/c8442f8d-4aab-4396-bbb2-bc9648d147f0" />
<img width="1125" height="853" alt="image" src="https://github.com/user-attachments/assets/154714f3-4062-4365-aa91-486c4bddb904" />

Ingesting over 10,000 company descriptions into Pinecone sequentially would take hours because each stock requires an HTTP request to Yahoo Finance, a text embedding computation and an upsert to Pinecone. Sequential processing handles one stock at a time: fetch AAPL, wait for the response, store it, then move to GOOGL, wait, store and so on.

The project uses Python's `concurrent.futures.ThreadPoolExecutor` to run ingestion in parallel across multiple worker threads. With 10 workers, 10 stocks are processed simultaneously. Each worker independently fetches data from Yahoo Finance and writes to Pinecone. A thread-safe file lock manages the progress tracking files so workers do not corrupt each other's writes. The ingestion is also resumable: already-processed tickers are loaded from disk at startup and skipped, so the process can be interrupted and restarted without duplicating work.

Parallel processing is appropriate here because the bottleneck is network I/O (waiting for HTTP responses), not CPU computation. While one thread waits for Yahoo Finance to respond, other threads can make their own requests. This is the classic use case for threading: I/O-bound concurrent work.

### Yahoo Finance Financial Metrics

Yahoo Finance (accessed via the `yfinance` Python library) provides the quantitative foundation for all analysis in the platform. For each company, the following metrics are retrieved and stored: earnings growth, revenue growth, gross margins, EBITDA margins, 52-week price change, market capitalization, trailing P/E ratio, sector, industry, country and the full business summary text.

These metrics matter because they represent the objective financial reality of each company. Earnings growth tells you whether profits are expanding or contracting. Revenue growth tells you whether the business is growing. Gross margins reveal how efficiently the company produces its goods or services, and high gross margins (above 70%) typically indicate pricing power and a durable competitive advantage. EBITDA margins measure operational profitability. The 52-week price change reflects how the market has valued the company over the past year.

The LLM uses these numbers as the primary input for the AI financial comparison, radar scoring and final recommendation. By grounding the LLM in real numbers rather than asking it general questions about companies, the analysis is specific and verifiable rather than generic and vague.

### LLMs Financial Comparison

After the RAG search returns a set of matching stocks, the LLM receives all of their financial metrics in a single structured prompt and generates a comparative analysis. For each stock, it identifies the single most significant strength or weakness and cites the actual percentage. It then writes an investor takeaway that names the top performer, the highest-risk stock and a clear actionable conclusion.

This is one of the core ways the project goes beyond simply displaying financial data. Raw numbers require interpretation. A 53% gross margin is excellent in consumer hardware but merely average in enterprise software. A -96% earnings growth sounds catastrophic but may reflect a one-time accounting event. The LLM applies contextual financial reasoning to explain what the numbers mean in the context of each company's sector and business model, which is exactly what a human analyst does but cannot do at scale across dozens of companies simultaneously.

### Market Trend Radar with LLM Scoring

Each stock is scored on six investment dimensions by the LLM: growth potential, financial health, market competition, innovation, industry trends and regulatory environment. Each dimension receives a score from 0 to 10 with a one-sentence explanation grounded in the company's actual data.

The scoring uses the Yahoo Finance financial metrics plus the business description as input context. The system prompt provides explicit guidance on what each score level means: for market competition, 10 means the company is a near-monopoly dominant player, 5 means it operates in a competitive market and 1 means it faces extreme commoditization. For regulatory environment, 10 means the industry is largely unregulated and 1 means it faces heavy government oversight.

The scores are visualised as a radar (spider) chart with each stock rendered as a separate polygon in a distinct color. This allows investors to immediately see which companies are strong across all dimensions versus which have specific weaknesses.

This feature could not be replicated by simply asking ChatGPT about a company because it requires the actual financial data to be injected into the prompt. The LLM is acting as an interpreter of structured data, not as a source of facts.

### News Sentiment Analysis with LLM Extraction

The news sentiment pipeline fetches recent news articles from NewsAPI for each company and uses the LLM to extract structured information from each article. 

First, articles are searched by the full company name the LLM explicitly judges whether each article is genuinely about the target company before assigning a sentiment score. An article that mentions AMD in passing while primarily covering NVIDIA would be marked as not relevant and excluded from the average sentiment calculation. This prevents irrelevant articles from corrupting the sentiment signal. LLM extracts structured information from each article beyond just a sentiment score: the key recent development described, the main risk or challenge mentioned and any forward-looking statements. This transforms raw article text into a structured dataset that can be downloaded as a CSV for further analysis.

Sentiment scores range from 0 to 100. Articles scoring 61-80 are positive, 81-100 are very positive, 41-60 are neutral, 21-40 are negative, and 0-20 are very negative. The average sentiment across relevant articles for each company is computed and displayed as a bar chart and the LLM generates a ranked analysis explaining what each company's sentiment score suggests about its current market perception.

### SEC 10-Q Filing Analysis

Every publicly listed company in the United States is legally required to file a Form 10-Q with the Securities and Exchange Commission every quarter. The 10-Q contains management's discussion and analysis of financial results, forward-looking guidance, risk factors and audited financial statements. This is the most authoritative source of information about a company's current financial state because management is legally liable for what they write.

The SEC EDGAR pipeline works as follows. For each ticker, the company's CIK (Central Index Key) is looked up from the SEC's company tickers database. The CIK is used to fetch the company's full filing history from the EDGAR API. The most recent 10-Q filing is identified, and its index page is downloaded to find the primary HTML document. The document is downloaded, all HTML tags are stripped and the text is cleaned. The system searches the full text for the most informative sections: Management Discussion and Analysis, Results of Operations, Financial Condition and Risk Factors. Extraction starts from the beginning of the first section found, capturing the most relevant content for LLM analysis.

The LLM scores the filing on four dimensions from 1 to 10: Performance (current revenue and margin results), Growth Potential (forward guidance and expansion plans), Risks (where 10 means very low risk and 1 means severe red flags) and Competitive Edge (patents, customer lock-in, market share). 

The scores are displayed as a heatmap with all stocks on one axis and the four dimensions on the other, allowing immediate visual comparison across the entire result set. Detailed cards for each stock show the score bars with LLM explanations drawn directly from the filing text, including specific quotes and percentages.

The reason 10-Q analysis matters alongside Yahoo Finance data is that Yahoo Finance provides trailing metrics (what already happened), while the 10-Q contains forward-looking statements (what management expects to happen) and specific risk disclosures that are not captured in standard financial ratios. A company can have excellent trailing margins but disclose in its 10-Q that a major customer is departing or that litigation could impair the business. The LLM reads for those signals.

### BUY / HOLD / SELL Recommendations

The final section of the platform synthesizes all available data into an investment recommendation for each stock. The LLM receives Yahoo Finance metrics, the average news sentiment score and the four SEC 10-Q scores in a single prompt and is instructed to begin its response with one of BUY, HOLD or SELL followed by two to three sentences of specific reasoning citing actual numbers and concluding with one sentence identifying the main risk to the recommendation.

The prompt enforces directness and specificity. The system prompt instructs the model to avoid hedging language, generic statements and qualifications. The result is a concise, data-grounded investment thesis for each company that integrates quantitative financial metrics, qualitative market sentiment and regulatory risk signals in a way that would require significant manual work to replicate.

Recommendations are color-coded in the interface: green for BUY, yellow for HOLD, and red for SELL. Running the news sentiment and 10-Q sections before generating recommendations provides the fullest data set and the most comprehensive output.



## Setup and Configuration

- `PINECONE_API_KEY`: from pinecone.io, used to create and query the vector index
- `GROQ_API_KEY`: from console.groq.com, used for all LLM inference
- `NGROK_AUTH_TOKEN`: from ngrok.com, used to expose the Streamlit server publicly
- `NEWS_API_KEY`: from newsapi.org, used to fetch recent news articles


---

## Data Sources

| Source | Data Provided | Access Method |
|---|---|---|
| Yahoo Finance | Financial metrics, price history, business descriptions | yfinance Python library |
| Pinecone | Vector similarity search over 10,000+ stock embeddings | pinecone-client Python SDK |
| SEC EDGAR | Official 10-Q quarterly filings | EDGAR public REST API |
| NewsAPI | Recent news articles by company name | REST API with API key |
| Groq / LLaMA 3.1 | LLM inference for all analysis and generation | Groq Python SDK |

## Notes

This project is intended for my own educational and research purposes. The BUY, HOLD and SELL recommendations generated by the LLM are based on automated analysis of publicly available data and do not constitute financial advice. Investment decisions should always be made in consultation with a qualified financial professional.


News sentiment accuracy depends on NewsAPI coverage, which varies by company size. Large-cap companies like NVIDIA and Apple will have many relevant articles. Smaller or less-covered companies may return few or no articles, resulting in a sentiment score of zero by default.
