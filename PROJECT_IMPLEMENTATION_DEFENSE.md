# Factra — Comprehensive Technical Implementation & Viva Defense Guide

> **Confidential Technical Reference for Review Defense**  
> **Prepared for:** Yash · Soham · Prarthana · Mrudula  
> **Course:** Deep Learning / Machine Learning Project Review

---

## 1. Complete System Architecture & Data Flow

```
                      ┌────────────────────────────────────────┐
                      │              USER INPUT                │
                      │   (Web URL / Raw Paragraph / Claim)    │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │    CHROME EXTENSION    │                     │     DASHBOARD UI       │
     │  (popup.js / Manifest  │                     │ (index.html / app.js)  │
     │  V3 DOM Content Script)│                     │ Direct REST Trigger    │
     └────────────┬───────────┘                     └───────────┬────────────┘
                  │ POST /predict /predict_url                  │
                  └───────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            FLASK REST API              │
                      │              (api/app.py)              │
                      │  CORS-Scoped, Port 5000, /health Probe │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     ┌────────────────────────┐                     ┌────────────────────────┐
     │    SCRAPER MODULE      │                     │      i18n LAYER        │
     │    (model/scrape.py)   │                     │    (model/i18n.py)     │
     │  SSRF Filter (blocks   │                     │ langdetect (seed=0)    │
     │  127.0.0.1, RFC1918),  │                     │ + deep-translator      │
     │  BeautifulSoup + lxml  │                     │ (chunked <=4500 chars) │
     └────────────┬───────────┘                     └───────────┬────────────┘
                  │ Clean Text                                  │ English Text
                  └───────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │           PREDICTION ENGINE            │
                      │           (model/predict.py)           │
                      └───────────────────┬────────────────────┘
                                          │
         ┌──────────────────┬─────────────┴───────┬──────────────────┐
         ▼                  ▼                     ▼                  ▼
┌─────────────────┐ ┌───────────────┐ ┌───────────────────┐ ┌────────────────┐
│  NEURAL ENGINE  │ │  XAI ATTN MAP │ │  WEB CONSENSUS    │ │ LINGUISTIC CUE │
│ (bert-base-unc) │ │ CLS to Token  │ │ (Serper API News  │ │ Tone, Readability│
│ PyTorch GPU FP16│ │ Final Layer   │ │ + Fact Check API) │ │ & Objectivity  │
└────────┬────────┘ └───────┬───────┘ └─────────┬─────────┘ └────────┬───────┘
         │                  │                   │                    │
         └──────────────────┴─────────────┬─────┴────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        INTELLIGENCE AGGREGATOR         │
                      │  Credibility = 0.4(BERT) + 0.4(Web)    │
                      │              + 0.2(Reputation) - Damps │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │             JSON RESPONSE              │
                      │  Prediction, Confidence, Credibility,  │
                      │  Anomalies, Consensus Sources, Tone    │
                      └────────────────────────────────────────┘
```

---

## 2. Deep Learning Deep-Dive (Model & Mathematical Formulation)

### 2.1 Why `bert-base-uncased`?
- **Transformer Architecture:** 12 encoder layers (transformer blocks), 12 self-attention heads per layer, hidden dimensionality $d_{model} = 768$, and feedforward dimension $d_{ff} = 3072$. Total parameters $\approx 110\text{M}$.
- **Bidirectional Self-Attention:** Unlike autoregressive models (like GPT) which attend only to past tokens ($P(w_i | w_1, ..., w_{i-1})$), BERT computes full cross-token attention:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  where queries $Q$, keys $K$, and values $V$ are linear projections of input tokens. This captures non-local stylistic signals such as exaggerated claims followed by ungrounded attribution.

### 2.2 Classification Head Adaptation
- The special first token `[CLS]` aggregates aggregate sequence representation:
  $$h_{\text{CLS}} = \text{Transformer}(\text{Tokens})_{0} \in \mathbb{R}^{768}$$
- This vector is passed through a classification dropout layer ($p = 0.1$) and a linear classification projection:
  $$\mathbf{z} = W \cdot h_{\text{CLS}} + b \quad \text{where } W \in \mathbb{R}^{2 \times 768}, \; b \in \mathbb{R}^2$$
- Softmax is applied to obtain class probabilities for `[Real, Fake]`:
  $$P(\text{Class } i) = \frac{e^{z_i}}{\sum_{j=0}^{1} e^{z_j}}$$

### 2.3 Loss Function
- **Categorical Cross-Entropy Loss:**
  $$\mathcal{L} = -\sum_{i=0}^{1} y_i \log(P_i) = - \left( y \log(P_{\text{fake}}) + (1-y) \log(P_{\text{real}}) \right)$$
- Optimizer: **AdamW** (Adam with Decoupled Weight Decay, $\lambda = 0.01$, $\text{lr} = 2 \times 10^{-5}$).

### 2.4 Training Configuration
- **Dataset:** ~3,721 curated news articles after dropping null/empty texts and normalization.
- **Class Balance:** Balanced split (50.3% Fake / 49.7% Real).
- **Stratified Split:** 80% Training (2,976 samples), 20% Validation (745 samples).
- **Hardware Acceleration:** Native PyTorch with CUDA (`cu124`) on **NVIDIA GeForce RTX 3050 6GB Laptop GPU** utilizing mixed precision (FP16).
- **Converged Metrics:**
  - Training steps: 374 batches across 2 epochs
  - Evaluation Loss: `0.000755`
  - Validation Accuracy: `100%` (or ~99.87% on generalized test sets)
  - Precision: `1.0`, Recall: `1.0`, F1: `1.0`

---

## 3. Explainable AI (XAI) & Attention Extraction

### 3.1 Attention Mapping
- During inference, `model(..., output_attentions=True)` returns attention matrices across all 12 layers:
  $$\text{attentions} \in \mathbb{R}^{12 \times \text{batch} \times \text{heads} \times \text{seq\_len} \times \text{seq\_len}}$$
- We isolate the final layer ($\text{layer}_{11}$), average across all 12 heads:
  $$\bar{A} = \frac{1}{12}\sum_{h=1}^{12} A_{11, h} \in \mathbb{R}^{\text{seq\_len} \times \text{seq\_len}}$$
- We look at row 0 (the attention vector from `[CLS]` to every token $j$):
  $$w_j = \bar{A}_{0, j}$$
- Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) and subword fragments starting with `##` are filtered out.
- The top-$K$ tokens with the highest weights represent the words that most influenced the classification decision.

### 3.2 Linguistic Telemetry
1. **Tone Profiling:** Searches for alarmist and clickbait trigger terms (`"shocking"`, `"breaking"`, `"unbelievable"`, etc.) alongside exclamation marks to categorize tone as *Alarmist / Sensational*, *Heightened / Intense*, or *Neutral / Informative*.
2. **Complexity Index:** Evaluates average word length and sentence structure. Sensational fake news consistently exhibits lower syntactic complexity compared to formal journalism.
3. **Objectivity Vector:** Scans for subjective pronouns (*"I"*, *"we"*, *"they"*, *"our"*) and emotive superlatives to classify as *Objective (Factual framing)* or *Subjective (Emotionally driven)*.

---

## 4. Web Consensus & Source Reputation

### 4.1 Web Consensus Engine (Serper.dev)
- Traditional classifiers fail on novel events because they only know what they saw during training.
- Factra extracts the core claim and issues a real-time query to **Serper.dev** (Google News / Search Index).
- It scores **Consensus Density**:
  - Checks if reputable news domains appear in search results.
  - An exact hostname or valid subdomain match against reputable outlets (Reuters, AP, BBC, NYT, NPR, The Guardian) triggers high consensus ($>70\%$).
  - If a sensational claim yields zero global index footprint, the score drops ($<30\%$), alerting users to an isolated fabrication.

### 4.2 Composite Credibility Score
The final credibility metric is a weighted blend:
$$\text{Score} = \left( P(\text{Real}) \times 40 \right) + \left( \text{ConsensusScore} \times 0.40 \right) + \left( \text{SourceReputation} \times 0.20 \right) - \text{Penalties}$$
Penalties include:
- Clickbait keyword deductions ($-8\%$ per unique clickbait word)
- ALL-CAPS token penalties ($-3\%$ per instance)
- Consecutive exclamation mark penalties ($-5\%$)

---

## 5. Security & Engineering Architecture

1. **Server-Side Request Forgery (SSRF) Defense:**
   - In `model/scrape.py`, every requested URL is inspected via `_is_safe_url()`.
   - Hostnames are resolved to IP addresses and verified against `ipaddress.ip_address()`.
   - Private (`10.0.0.0/8`, `192.168.0.0/16`), loopback (`127.0.0.1`), link-local (`169.254.169.254`), and reserved subnets are strictly blocked.
2. **Chrome Manifest V3 CSP Compliance:**
   - Manifest V3 prohibits inline `<script>` tags in extension pages. All initialization and theme toggling logic is encapsulated in `popup.js`.
3. **Cross-Site Scripting (XSS) Sanitization:**
   - External links from Google News search are sanitized via `sanitizeUrl()` restricting protocols to `http:` and `https:`, with `rel="noopener noreferrer"`.
4. **Resilient Data Loading:**
   - Custom pure-Python streaming CSV loader handles dataset ingestion without C-extension DLL policy blocks on Windows Smart App Control.

---

## 6. Back-Question & Viva Defense Bank (Q&A by Member)

---

### 🛡️ For Yash (System Architecture & Pipeline)

**Q1: Why did you build a Chrome Extension instead of just a web app?**
> *"Misinformation spreads natively inside social feeds, articles, and blog posts. An external website forces users to copy-paste text, creating friction. A Chrome extension operates at the point of consumption, utilizing DOM TreeWalkers to inject highlights directly into the live page without modifying layout."*

**Q2: What happens if the article website is behind Cloudflare or renders via client-side JavaScript?**
> *"Our Chrome extension extracts text directly from the already-rendered browser DOM in the active tab. Only if the user submits a standalone URL does our server-side BeautifulSoup/lxml scraper take over. If scraping returns fewer than 80 characters, the extension flags the low density and falls back gracefully."*

**Q3: How do you prevent attackers from using your `/predict_url` endpoint to probe internal networks (SSRF)?**
> *"We implemented strict IP resolution checks in `model/scrape.py`. Before making any HTTP request, we parse the hostname, resolve its IPv4/IPv6 address via DNS, and verify that the target is not in private, loopback (127.0.0.1), or cloud metadata ranges (169.254.169.254). Any unsafe address returns an immediate rejection."*

---

### 🛡️ For Soham (Deep Learning & Modeling)

**Q1: Why BERT instead of RoBERTa, DeBERTa, or a lightweight LSTM?**
> *"LSTMs are recurrent and suffer from sequential bottlenecking, missing long-distance semantic dependencies. While RoBERTa or DeBERTa offer incremental improvements, `bert-base-uncased` provides the optimal balance of inference latency (~150ms per batch on GPU) and rich contextual bidirectional attention representations. For a production-ready extension where users expect sub-second classification, BERT is ideal."*

**Q2: Your validation accuracy is extremely high (~99.8%). Does this mean the model is overfitting?**
> *"This is a classic phenomenon in benchmark news classification datasets where stylistic markers (specific sensational words, reporting structures) are prominent. We explicitly documented this limitation in our project architecture. That is precisely why we do not rely solely on BERT; our final decision is grounded in our Web Consensus Engine (Pillar 2) and publisher reputation, ensuring robust real-world generalization."*

**Q3: Explain the mathematical intuition behind the `[CLS]` token.**
> *"In BERT, `[CLS]` is prepended to every input sequence. Because self-attention allows all tokens to attend to all other tokens, the final hidden state corresponding to `[CLS]` effectively aggregates pooled information across the entire sequence through 12 multi-head attention layers, making it a principled representation for whole-sentence classification."*

---

### 🛡️ For Prarthana (Web Consensus & APIs)

**Q1: If BERT is already 99% accurate, why do you need the Serper Web Consensus engine?**
> *"Because linguistic fluency does not equal truth. A sophisticated bad actor can write a completely fabricated story using formal, objective, academic language. BERT would predict 'Real' based on style alone. Web Consensus verifies real-world corroboration: if a massive claim has zero reporting across trusted international news wires, its credibility score is immediately penalized."*

**Q2: How is publisher reputation mapped and updated?**
> *"We maintain a domain reputation registry (`api/source_reputation.json`) scoring established publishers (Reuters: 98, AP: 98, BBC: 92) based on media reliability indices. Domains are matched via full hostname and subdomain parsing using `urlparse`. The data is cached in memory upon initial server startup to avoid repetitive disk I/O."*

**Q3: What if Serper API fails or rate-limits?**
> *"The consensus module wraps external calls in defensive try-except blocks with timeouts. If external APIs fail or keys are absent, the system gracefully degrades: it reports status as 'Unavailable' and computes credibility based on BERT confidence and linguistic heuristics without crashing."*

---

### 🛡️ For Mrudula (XAI, Frontend & Client Layer)

**Q1: How do you extract attention weights for Explainability without retraining?**
> *"BERT inherently computes attention matrices during every forward pass. By setting `output_attentions=True` during model instantiation, we tap directly into the 12th layer's attention tensor, average over the 12 attention heads, and extract the weight vector from `[CLS]` to each individual token. This gives us zero-overhead feature attribution."*

**Q2: How does the Chrome Extension highlight text without breaking the host webpage's layout?**
> *"Rather than using naive `innerHTML.replace()`—which can corrupt inline SVG icons, React event listeners, and nested HTML structures—we use a DOM `TreeWalker(NodeFilter.SHOW_TEXT)`. It inspects pure text nodes, splits them around exact regex match boundaries, and injects lightweight `<mark>` elements. Clicking 'Clear Highlights' simply replaces the `<mark>` nodes back with text nodes and normalizes the DOM."*

**Q3: How does the extension hand off results to the full dashboard?**
> *"When 'Full Analysis Dashboard' is clicked in the popup, the extension serializes the full scan findings—including prediction, credibility score, tone, and consensus sources—into URL search parameters and opens the dashboard tab (`/?handoff=1&...`). The dashboard parses these parameters on load, renders the animated SVG gauges, and saves the scan into persistent browser local storage."*
