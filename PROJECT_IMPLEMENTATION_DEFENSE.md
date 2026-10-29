# Factra — Comprehensive Technical Implementation & Deep Learning Viva Defense Master Guide

> **Confidential Academic Reference & Technical Defense Manual**  
> **Prepared for:** Yash · Soham · Prarthana · Mrudula  
> **Course:** Deep Learning (DL) Project Examination & Viva Voce  
> **Repository:** `https://github.com/Yashparmar1125/Factra`

---

# Table of Contents
1. [System Architecture & Multi-Layer Pipeline](#1-system-architecture--multi-layer-pipeline)
2. [Deep Learning Core & Mathematical Foundations](#2-deep-learning-core--mathematical-foundations)
3. [Explainable AI (XAI) & Attention Formulation](#3-explainable-ai-xai--attention-formulation)
4. [Real-World Grounding & Web Consensus Engine](#4-real-world-grounding--web-consensus-engine)
5. [Security, Performance & Full-Stack Engineering](#5-security-performance--full-stack-engineering)
6. [Comprehensive Viva Voce Question & Answer Bank](#6-comprehensive-viva-voce-question--answer-bank)
   - [Section A: Core Deep Learning & NLP Fundamentals (General for all)](#section-a-core-deep-learning--nlp-fundamentals)
   - [Section B: Individual Member Defense (Targeted per person)](#section-b-individual-member-defense)
     - [🛡️ Yash: Architecture, System Design & Ingestion Pipeline](#-yash-architecture-system-design--ingestion-pipeline)
     - [🛡️ Soham: Transformer Math, Model Training & Loss Mechanics](#-soham-transformer-math-model-training--loss-mechanics)
     - [🛡️ Prarthana: Consensus Engine, APIs & Credibility Formulation](#-prarthana-consensus-engine-apis--credibility-formulation)
     - [🛡️ Mrudula: XAI Attention Extraction, Client Engineering & DOM Highlighting](#-mrudula-xai-attention-extraction-client-engineering--dom-highlighting)
   - [Section C: Tough / Trap Questions & Recommended Responses](#section-c-tough--trap-questions--recommended-responses)

---

# 1. System Architecture & Multi-Layer Pipeline

```
                                  ┌──────────────────────────────────────────────┐
                                  │                 USER INPUT                   │
                                  │    • Raw News Paragraph / Claim              │
                                  │    • Active Webpage URL / Tab                │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                  ┌──────────────────────┴───────────────────────┐
                                  ▼                                              ▼
                    ┌───────────────────────────┐                  ┌───────────────────────────┐
                    │     CHROME EXTENSION      │                  │    FORENSIC DASHBOARD     │
                    │   (Manifest V3 popup.js)  │                  │   (index.html / app.js)   │
                    │  DOM Traversal & Content  │                  │ Direct In-Dashboard Input │
                    └─────────────┬─────────────┘                  └─────────────┬─────────────┘
                                  │                                              │
                                  │ HTTP POST /predict or /predict_url           │
                                  └──────────────────────┬───────────────────────┘
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │              FLASK REST API                  │
                                  │               (api/app.py)                   │
                                  │   CORS Scoped, Port 5000, /health Monitor    │
                                  └──────────────────────┬───────────────────────┘
                                                         │
                                  ┌──────────────────────┴───────────────────────┐
                                  ▼                                              ▼
                    ┌───────────────────────────┐                  ┌───────────────────────────┐
                    │      SSRF SCRAPER         │                  │        i18n LAYER         │
                    │    (model/scrape.py)      │                  │      (model/i18n.py)      │
                    │ Validates DNS against     │                  │ Language detection (seed=0│
                    │ private, loopback & cloud │                  │ + deep-translator chunked │
                    │ metadata IPs before fetch │                  │ to <= 4500 chars          │
                    └─────────────┬─────────────┘                  └─────────────┬─────────────┘
                                  │                                              │
                                  └──────────────────────┬───────────────────────┘
                                                         ▼
                                  ┌──────────────────────────────────────────────┐
                                  │              PREDICT ENGINE                  │
                                  │            (model/predict.py)                │
                                  └──────────────────────┬───────────────────────┘
                                                         │
               ┌───────────────────────┬─────────────────┴─────┬────────────────────────┐
               ▼                       ▼                       ▼                        ▼
     ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌────────────────────┐
     │   NEURAL ENGINE   │   │  XAI ATTN MAPPER  │   │   WEB CONSENSUS   │   │  LINGUISTIC ENGINE │
     │  bert-base-unc    │   │ Layer-12 [CLS] to │   │ Serper.dev Google │   │ • Tone Profiling   │
     │  PyTorch CUDA     │   │ token attention   │   │ News + FactCheck  │   │ • Complexity Index │
     │  FP16 Inference   │   │ weights extraction│   │ Claims API        │   │ • Objectivity Vec  │
     └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘   └──────────┬─────────┘
               │                       │                       │                        │
               └───────────────────────┼───────────────────────┴────────────────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   INTELLIGENCE FUSION     │
                         │ Credibility =             │
                         │   0.40 * P(Real) +        │
                         │   0.40 * ConsensusScore + │
                         │   0.20 * Reputation -     │
                         │   Clickbait & Caps Damps  │
                         └─────────────┬─────────────┘
                                       ▼
                         ┌───────────────────────────┐
                         │   STRUCTURED JSON OUT     │
                         │ Prediction, Confidence,   │
                         │ Credibility, Cues, Sources│
                         └───────────────────────────┘
```

---

# 2. Deep Learning Core & Mathematical Foundations

### 2.1 Why BERT over RNNs, LSTMs, and GRUs?
1. **Vanishing/Exploding Gradients:** RNNs suffer from exponential degradation of gradient signals across long sequences due to repeated matrix multiplications ($W_{hh}^T$). LSTMs mitigate this with gating mechanisms (input, forget, output gates), but still exhibit sequential bottlenecks.
2. **Sequential Computation vs. Parallelism:** LSTMs are inherently sequential ($O(n)$ time complexity per sequence), which prevents parallel computation on modern GPU tensor cores. Transformers process all $n$ tokens simultaneously ($O(1)$ sequential operations).
3. **Contextual Encoding:** In word embeddings (Word2Vec, GloVe), a word has one static vector regardless of context. BERT produces dynamic contextual embeddings: the token `"bank"` in *"river bank"* has a completely distinct representation from `"bank"` in *"investment bank"*.

### 2.2 Mathematical Anatomy of Bidirectional Multi-Head Self-Attention
For an input sequence embedded into matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$, where $n$ is sequence length ($256$) and $d_{\text{model}} = 768$:

1. **Linear Projections:** Three parameter matrices $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$ generate Queries ($Q$), Keys ($K$), and Values ($V$):
   $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

2. **Scaled Dot-Product Attention:**
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
   * **Why scale by $\sqrt{d_k}$?** For large values of $d_k$ ($64$ in BERT), the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients ($\approx 0$), causing gradient saturation. Dividing by $\sqrt{d_k}$ stabilizes the variance to $1.0$.

3. **Multi-Head Attention:** Instead of performing a single attention function, BERT computes $h = 12$ parallel attention heads, allowing the model to jointly attend to information from different representation subspaces at different positions:
   $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_{12}) W^O$$
   $$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

### 2.3 Classification Head Formulation
- The special first token `[CLS]` aggregates whole-sequence semantic representation through 12 stacked transformer layers:
  $$h_{\text{CLS}} = \text{Encoder}(X)_{0} \in \mathbb{R}^{768}$$
- A dropout layer is applied to prevent co-adaptation of features:
  $$\tilde{h}_{\text{CLS}} = \text{Dropout}(h_{\text{CLS}}, p = 0.1)$$
- A feed-forward linear projection outputs raw unnormalized logits $\mathbf{z} \in \mathbb{R}^2$:
  $$\mathbf{z} = W_c \cdot \tilde{h}_{\text{CLS}} + b_c \quad \text{where } W_c \in \mathbb{R}^{2 \times 768}, \; b_c \in \mathbb{R}^2$$
- Softmax maps logits to a categorical probability distribution over classes $\{\text{Real} (0), \text{Fake} (1)\}$:
  $$P(\text{Class } i) = \frac{e^{z_i}}{e^{z_0} + e^{z_1}}$$

### 2.4 Loss Function: Categorical Cross-Entropy
Given binary ground truth label $y \in \{0, 1\}$ ($0 = \text{Real}, 1 = \text{Fake}$):
$$\mathcal{L}_{\text{CE}} = - \sum_{c=0}^1 y_c \log(\hat{y}_c) = - \left[ (1 - y) \log(P_{\text{real}}) + y \log(P_{\text{fake}}) \right]$$

### 2.5 Optimization: AdamW
Unlike standard Adam which implements L2 regularization by adding weight decay directly to the gradient ($g_t = \nabla \mathcal{L} + \lambda \theta_{t-1}$), **AdamW** (Loshchilov & Hutter) decouples weight decay from the gradient update:
$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$
This prevents weights with large historical gradients from decaying slower than weights with small gradients, leading to superior generalization.

---

# 3. Explainable AI (XAI) & Attention Formulation

### 3.1 Attention Map Extraction
- When running inference with `output_attentions=True`, BERT returns a tuple of attention tensors for all 12 layers:
  $$\mathcal{A} \in \mathbb{R}^{12 \times B \times H \times N \times N}$$
  where $B = 1$ (batch size), $H = 12$ (attention heads), and $N \le 256$ (sequence length).
- We select the final layer ($\text{Layer}_{11}$), representing the highest-level semantic abstractions, and take the mean across all 12 heads:
  $$\bar{A} = \frac{1}{12} \sum_{h=1}^{12} \mathcal{A}_{11, 0, h} \in \mathbb{R}^{N \times N}$$
- To measure how much each input token influenced the sequence-level decision, we slice row 0 (the attention weights directed from the `[CLS]` token to token $j$):
  $$w_j = \bar{A}_{0, j} \quad \text{for } j \in \{0, \dots, N-1\}$$
- Filtering: Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) and subword fragments (tokens starting with `##`) are filtered out.
- The top-$K$ tokens with the highest attribution scores $w_j$ are selected as the model's anomalous cues.

### 3.2 Linguistic Telemetry Metrics
1. **Tone Profiling:** Computes the frequency of sensational/alarmist lexical cues:
   $$\text{Alarmist cues} = \{\text{"shocking"}, \text{"breaking"}, \text{"insane"}, \text{"unbelievable"}, \text{"horrifying"}, \text{"urgent"}\}$$
   If frequency $\ge 2$ or exclamation mark count $\ge 3$, tone is classified as *Alarmist / Sensational*.
2. **Complexity Index:** Measures mean token length and syntactic depth:
   $$\text{Complexity} = \frac{\sum_{i=1}^M \text{len}(\text{token}_i)}{M}$$
   Formal journalism typically averages $>5.2$ chars/token; sensational clickbait averages $<4.4$ chars/token.
3. **Objectivity Vector:** Calculates the ratio of first-person emotional pronouns ($I, we, our, me$) to third-person entities ($he, she, they, institutional nouns$).

---

# 4. Real-World Grounding & Web Consensus Engine

### 4.1 The Need for Grounding
A neural network only possesses knowledge frozen up to its training cutoff. A bad actor can write a completely fictitious headline using formal grammatical language (e.g., *"The Federal Reserve has officially abolished interest rates as of this morning"*). A pure BERT classifier would mark this as "Real" because the syntax is sophisticated and neutral. **Web Consensus is the empirical countermeasure.**

### 4.2 Serper.dev News Verification Workflow
1. The backend extracts the key factual assertion from the input text.
2. An HTTP POST request is dispatched to Serper.dev Google Search API (`https://google.serper.dev/news`).
3. Returned article titles, snippets, and publisher URLs are parsed.
4. **Domain Matching:** Rather than fragile substring checks (`"bbc"` in URL), domains are extracted using `urllib.parse.urlparse` and verified against exact root hostnames or subdomains (e.g., `news.bbc.co.uk` $\rightarrow$ `bbc.co.uk`).
5. **Consensus Density Scoring:**
   - 3+ reputable wire matches $\rightarrow$ Score = $85–100\%$
   - 1–2 secondary matches $\rightarrow$ Score = $50–70\%$
   - Zero matching index $\rightarrow$ Score = $0–25\%$ (*Isolated / Fabricated Footprint*)

### 4.3 Composite Credibility Score Equation
$$\text{Credibility} = \max\left(0, \min\left(100, \, \left(0.40 \cdot P_{\text{real}} \cdot 100\right) + \left(0.40 \cdot \text{Consensus}\right) + \left(0.20 \cdot \text{Reputation}\right) - \sum \text{Penalties}\right)\right)$$

Where penalties include:
- Clickbait keyword penalty: $-8.0$ points per unique clickbait term
- ALL-CAPS penalty: $-3.0$ points per fully capitalized word
- Exclamation mark penalty: $-5.0$ points if $\ge 2$ exclamation marks exist

---

# 5. Security, Performance & Full-Stack Engineering

1. **Server-Side Request Forgery (SSRF) Defense:**
   - An attacker could pass `http://169.254.169.254/latest/meta-data/` to `/predict_url` to steal cloud credentials, or `http://127.0.0.1:3306` to probe local databases.
   - We implemented `_is_safe_url()` in `model/scrape.py`: resolves domain via DNS, extracts IP, and checks against Python's `ipaddress.ip_address()`. Any IP in `is_private`, `is_loopback`, `is_link_local`, or `is_reserved` is rejected immediately.
2. **Chrome Manifest V3 Security:**
   - Manifest V3 strictly enforces `script-src 'self'`. Inline scripts (`<script>...theme...</script>`) violate Content Security Policy (CSP). All execution logic is packaged strictly inside `popup.js`.
3. **Non-Destructive In-Page Highlighting:**
   - Using `document.body.innerHTML.replace()` breaks modern web apps by destroying DOM event listeners and corrupting SVG icons.
   - We implemented a DOM `TreeWalker(NodeFilter.SHOW_TEXT)` that visits only raw text nodes, splits them around matched keywords, and wraps them in `<mark data-fnd-mark>` elements. Reverting simply unwraps the text node and calls `parent.normalize()`.

---

# 6. Comprehensive Viva Voce Question & Answer Bank

---

## Section A: Core Deep Learning & NLP Fundamentals

### Q1: What is the Transformer architecture and how does it differ from traditional CNNs and RNNs?
> **Answer:** Traditional CNNs process local spatial windows with fixed kernel sizes and lack global sequence memory. RNNs process tokens sequentially step-by-step, suffering from vanishing gradients across long dependencies and an inability to parallelize across tokens. Transformers rely entirely on **Self-Attention mechanisms**, discarding recurrent and convolutional layers. This enables $O(1)$ path length between any two arbitrary tokens in a sequence, capturing long-range semantic dependencies with full parallelization on GPU hardware.

### Q2: What is the purpose of Positional Encodings in BERT?
> **Answer:** Because the self-attention operation is permutation-invariant—meaning $\text{Attention}(Q, K, V)$ yields the exact same numerical result regardless of token order—the transformer has no innate concept of word order. BERT introduces learned **Positional Embeddings** ($\mathbb{R}^{256 \times 768}$) that are added element-wise to the Token Embeddings and Segment Embeddings before the first encoder layer. This ensures the model distinguishes between *"Dog bites man"* and *"Man bites dog"*.

### Q3: Why does BERT use WordPiece tokenization instead of word-level or character-level tokenization?
> **Answer:** Word-level tokenization produces massive vocabularies ($>500,000$ words) and fails on out-of-vocabulary (OOV) words. Character-level tokenization produces long sequences that dilute semantic meaning and explode computational costs ($O(n^2)$ attention). BERT's **WordPiece** tokenization is a subword algorithm that strikes an optimal balance: frequent words remain intact (`"news"`), while rare or inflected words are broken into meaningful subword units (`"misinformation"` $\rightarrow$ `["mis", "##information"]`). Any unseen word can be constructed from subword pieces, eliminating OOV tokens.

### Q4: Explain the difference between BERT Pre-training and Fine-Tuning.
> **Answer:** 
> - **Pre-training:** Self-supervised learning on billions of unannotated words (Wikipedia + BookCorpus) using two auxiliary tasks: **Masked Language Modeling (MLM)** where 15% of tokens are masked and predicted, and **Next Sentence Prediction (NSP)**. This teaches the model universal linguistic syntax and semantic representations.
> - **Fine-Tuning:** Supervised learning where the pre-trained weights are loaded as an initialization point, a task-specific classification head is appended, and all weights are gently updated on labeled fake/real news data with a low learning rate ($2 \times 10^{-5}$).

### Q5: What is the difference between Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5) architectures?
> **Answer:**
> - **Encoder-only (BERT):** Utilizes unmasked bidirectional self-attention. Excellent for sequence understanding, text classification, named-entity recognition, and extractive QA.
> - **Decoder-only (GPT):** Utilizes causal (masked) self-attention where tokens can only attend to prior tokens. Optimized for autoregressive text generation.
> - **Encoder-Decoder (T5 / BART):** Uses a bidirectional encoder and an autoregressive decoder. Optimized for sequence-to-sequence transformation tasks like translation and abstractive summarization.

---

## Section B: Individual Member Defense

---

### 🛡️ Yash: Architecture, System Design & Ingestion Pipeline

#### Q1: Walk us through the end-to-end data lifecycle when a user submits a URL to Factra.
> **Answer:** 
> 1. The client sends a `POST /predict_url` request with `{ "url": "..." }` to our Flask backend.
> 2. `model/scrape.py` validates the URL against SSRF filters to block private IP attacks.
> 3. BeautifulSoup and lxml scrape the main article text, stripping navigation menus, footers, and ads.
> 4. `model/i18n.py` detects the language with `langdetect`. If non-English, it translates it via `deep-translator` in chunks $\le 4500$ characters.
> 5. The text passes to `predict_text()` in `model/predict.py`. BERT generates class logits and probabilities; attention weights are extracted; linguistic heuristics are computed; and Serper.dev is queried for Google News consensus.
> 6. The aggregated credibility score is returned as a structured JSON object to the dashboard or extension.

#### Q2: What design pattern did you use for the backend, and why?
> **Answer:** We implemented the **Application Factory Pattern** (`create_app(model_dir)` in `api/app.py`). This guarantees that heavy PyTorch model weights ($417\text{ MB}$) and tokenizers are loaded into GPU memory **only once** at server startup, rather than per HTTP request. Routes operate as closures around the active model in memory, ensuring sub-200ms API response latency.

#### Q3: Why did you implement SSRF protection, and how does it work?
> **Answer:** In `predict_url`, our server performs an outbound HTTP request on behalf of the user. Without protection, an attacker could supply `http://localhost:5000/internal` or cloud metadata endpoints (`http://169.254.169.254`) to inspect our server's internal environment. We implemented `_is_safe_url()`: it extracts the hostname, uses `socket.getaddrinfo()` to resolve the underlying IP address, and verifies that the IP is not private, loopback, link-local, or reserved using Python's `ipaddress` library before issuing any `requests.get()`.

#### Q4: Why did you separate `text_raw` from `text_clean` in the prediction pipeline?
> **Answer:** In our codebase audit, we found a critical bug: `clean_text()` strips punctuation and lowercases all characters. If heuristics for ALL-CAPS words or exclamation marks run on `text_clean`, they will always return zero detections. We refactored `predict_text()` so that `text_clean` is fed into the BERT tokenizer (matching its pre-training format), while `text_raw` is preserved for stylistic heuristics and linguistic telemetry.

---

### 🛡️ Soham: Transformer Math, Model Training & Loss Mechanics

#### Q1: What were your exact training hyperparameters and hardware setup?
> **Answer:**
> - **Base Model:** `bert-base-uncased` (12 layers, 12 heads, 768 hidden size, 110M parameters).
> - **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6GB VRAM) via PyTorch with CUDA 12.4.
> - **Optimizer:** AdamW ($\text{lr} = 2 \times 10^{-5}$, weight decay $\lambda = 0.01$).
> - **Batch Size & Epochs:** Batch size of 16, sequence length capped at 256 tokens, trained over 2 epochs (374 optimization steps).
> - **Mixed Precision:** FP16 enabled via PyTorch CUDA.
> - **Final Validation Metrics:** Loss converged to $0.000755$, with accuracy and F1 score $>0.99$.

#### Q2: Your validation accuracy is 99.8%. Is this model overfitting? How do you defend this?
> **Answer:** In standard binary fake news benchmarks, stylistic features (such as clickbait syntax and partisan rhetoric) create stark separation, allowing deep transformers to achieve very high accuracy on in-distribution test splits. However, this is precisely why we explicitly argue that **BERT alone is insufficient for production deployment**. In-distribution fluency does not prevent false positives on well-written fake news. This is the entire justification for our 4-pillar architecture: BERT handles stylistic representation, while Web Consensus grounds claims in external real-world corroboration.

#### Q3: How did you fix the confidence score inversion bug in `predict.py`?
> **Answer:** Originally, the codebase assigned `confidence = prob_real`. This caused a catastrophic semantic bug: when the model was 99% certain an article was **Fake** ($P_{\text{fake}} = 0.99, P_{\text{real}} = 0.01$), it reported a confidence of `0.01` (1%). I refactored the logic to:
> ```python
> confidence = prob_real if pred_label == "Real" else (1.0 - prob_real)
> ```
> Now confidence correctly reflects the model's certainty in its chosen prediction.

#### Q4: Why did you rewrite the evaluation callback using pure PyTorch tensors instead of scikit-learn?
> **Answer:** On Windows machines running corporate or institutional Code Integrity policies (Smart App Control / WDAC), scikit-learn's compiled C-extension DLLs (specifically `_pairwise_fast.pyd`) were blocked from loading. To make Factra 100% resilient and self-contained, I implemented our confusion matrix and metric calculations (accuracy, precision, recall, F1) using pure PyTorch tensor operations (`torch.argmax`, boolean tensor masks, and sum reductions), completely eliminating the scikit-learn runtime dependency during model evaluation.

---

### 🛡️ Prarthana: Consensus Engine, APIs & Credibility Formulation

#### Q1: Explain the mathematical intuition behind your Composite Credibility Score.
> **Answer:** Rather than outputting raw softmax probability, we formulate credibility as a linear combination of three independent signals, modulated by heuristic penalties:
> $$\text{Credibility} = 0.40 \cdot \left(P_{\text{real}} \times 100\right) + 0.40 \cdot \text{ConsensusScore} + 0.20 \cdot \text{SourceReputation} - \text{Penalties}$$
> - $40\%$ weight on BERT evaluates linguistic and stylistic authenticity.
> - $40\%$ weight on Web Consensus verifies whether legitimate news wires corroborate the event.
> - $20\%$ weight on Source Reputation factors in the historical trust index of the publishing domain.
> - Penalties directly deduct points for sensational red flags: $-8$ per clickbait word, $-3$ per ALL-CAPS word, and $-5$ for excessive exclamation marks.

#### Q2: What happens if Serper API or Google Fact Check API experiences a network timeout or quota exhaustion?
> **Answer:** All external HTTP API calls in `model/predict.py` are wrapped in defensive `try-except` blocks with strict timeouts ($1.5$ to $5.0$ seconds). If an API fails or the API key is missing:
> - The consensus status is set to `"Unavailable / Isolated"`.
> - The credibility score falls back to the weighted combination of BERT confidence and linguistic heuristics without raising an unhandled exception or crashing the server.

#### Q3: How does your domain reputation lookup work, and how did you optimize it?
> **Answer:** We maintain a curated domain trust database in `api/source_reputation.json` mapping verified news organizations to trust scores (e.g., Reuters: 98, AP: 98, BBC: 92). In our performance audit, we discovered that `_get_source_reputation()` was reading and parsing this JSON file from disk on **every single prediction call**. I introduced a module-level global cache (`_SOURCE_REPUTATION_CACHE`): the JSON is parsed once into RAM on the first query, reducing disk I/O to zero for all subsequent predictions.

#### Q4: Why did you migrate domain matching to `urlparse` instead of substring matching?
> **Answer:** Substring matching (`if "bbc" in url`) is dangerous and prone to spoofing. An adversarial domain like `fake-bbc-news.com` or `phishing-reuters.org` would falsely evaluate to true. We replaced this with:
> ```python
> hostname = urlparse(link).hostname or ""
> is_reputable = any(hostname == dom or hostname.endswith("." + dom) for dom in reputable_domains)
> ```
> This guarantees that only genuine domains or legitimate subdomains (e.g., `edition.cnn.com`) are accredited.

---

### 🛡️ Mrudula: XAI Attention Extraction, Client Engineering & DOM Highlighting

#### Q1: How do you extract Explainable AI (XAI) attention weights without retraining BERT?
> **Answer:** Modern transformers inherently compute multi-head self-attention during the forward pass. By passing `output_attentions=True` to our model, the output object includes the raw attention matrices across all 12 layers. We extract layer 11 (the final layer), compute the mean across all 12 heads, and extract the attention weights between the `[CLS]` token and all input tokens. By sorting the non-special tokens by their `[CLS]` attention magnitude, we pinpoint the exact words that tipped the classification head toward its decision with zero additional training cost.

#### Q2: Explain the technical difference between naive DOM replacement and your TreeWalker highlighting algorithm.
> **Answer:** Many browser extensions use `document.body.innerHTML = document.body.innerHTML.replace(...)`. This is destructive: it destroys attached JavaScript event listeners, resets video players, breaks inline SVGs, and causes DOM re-rendering flashes. Instead, we use a DOM `TreeWalker`:
> ```javascript
> const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
> ```
> The TreeWalker iterates exclusively through pure text nodes. When a suspicious word matches, it splits the text node at the exact character offset and wraps only that substring in a `<mark data-fnd-mark>` tag, leaving all parent elements, event listeners, and styling completely unharmed.

#### Q3: Why did you have to remove the inline `<script>` tag from `extension/popup.html`?
> **Answer:** Under Google Chrome's **Manifest V3 Content Security Policy (CSP)**, executing inline JavaScript (`<script>...</script>` inside an HTML document) is strictly prohibited to prevent cross-site scripting (XSS) and code injection attacks. I refactored the theme detection logic into a self-executing IIFE at the top of `extension/popup.js`, ensuring full compliance with Chrome Web Store security standards.

#### Q4: How does data handoff work between the Chrome Extension and the full Dashboard?
> **Answer:** Chrome extensions run in an isolated origin (`chrome-extension://<id>`), while our dashboard runs on `http://127.0.0.1:5000`. Because `localStorage` is partitioned by origin, data saved in the extension is invisible to the dashboard. To achieve cross-origin handoff, when the user clicks *"Full Analysis Dashboard"*, the extension serializes the prediction, confidence, credibility score, and consensus data into URL search parameters (`/?handoff=1&p=Real&s=85...`) and opens a new browser tab. The dashboard parses these parameters, renders the animated SVG gauge, and saves the record into the dashboard's local history.

---

## Section C: Tough / Trap Questions & Recommended Responses

### Trap Q1: "Isn't BERT outdated? Why didn't you use Llama-3 or GPT-4o via an API?"
> **Strong Defense:** *"LLMs like Llama-3 or GPT-4o are generative autoregressive models with billions of parameters. Using them introduces three major disadvantages for our specific use case:
> 1. **Latency:** Calling a commercial LLM API introduces 1.5 to 3 seconds of network latency and token generation delays. Our fine-tuned BERT runs in under 150ms locally on an RTX 3050 GPU, allowing real-time browsing analysis.
> 2. **Operational Cost & Privacy:** Commercial APIs charge per token and expose sensitive user browsing history to third-party servers. Factra runs fully on-premise without recurring token fees.
> 3. **Determinism:** LLMs are prone to hallucinating explanations and non-deterministic classifications. BERT is a specialized sequence encoder with deterministic output logits, making it vastly superior for reproducible verification."*

### Trap Q2: "What if a fake news story is written with high complexity and neutral tone, and published on a new blog? Will your system fail?"
> **Strong Defense:** *"That is the exact failure mode of standalone classifiers that Factra was built to solve! If a story has neutral tone, BERT might predict a higher probability of being real. However, because it is published on an unverified blog and no legitimate news agency is reporting it, our **Web Consensus score will be 0%** and the **Source Reputation will be 0%**. Under our weighted formula:
> $$\text{Credibility} = (0.40 \times P_{\text{real}}) + (0.40 \times 0) + (0.20 \times 0) = \text{under } 40\%$$
> The dashboard will immediately flag it with an 'Isolated Fabricated Footprint' warning, successfully catching the deception."*

### Trap Q3: "What are the limitations of your system?"
> **Honest Academic Defense:** *"We have identified three key limitations:
> 1. **Multimodal Content:** Factra currently processes text and URLs; it does not yet analyze deepfake images or manipulated video clips embedded within articles.
> 2. **Breaking News Latency:** There is an inherent 10-to-15 minute propagation window between a breaking real-world event occurring and Google News indexing wire reports.
> 3. **Non-English Consensus:** While our i18n translation layer handles non-English input text, Serper consensus verification performs with highest accuracy on English-language global news indexes."*

---

## 🎯 Final Examiner Quick-Reference Card

| Topic | Metric / Fact | Source File |
|:---|:---|:---|
| **Model** | `bert-base-uncased`, 110M params, 12 layers, 12 heads | `model/predict.py` |
| **Max Sequence Length** | 256 tokens (truncated/padded) | `model/train.py` |
| **GPU Hardware** | NVIDIA RTX 3050 Laptop GPU (6GB VRAM), CUDA 12.4 | System Environment |
| **Training Steps** | 374 batches across 2 epochs, batch size = 16 | `model/train.py` |
| **Optimizer** | AdamW, lr = $2 \times 10^{-5}$, weight decay = $0.01$ | `model/train.py` |
| **Loss Function** | Categorical Cross-Entropy, Eval Loss = `0.000755` | `model/train.py` |
| **Credibility Formula** | $0.40(\text{BERT}) + 0.40(\text{Consensus}) + 0.20(\text{Reputation}) - \text{Damps}$ | `model/predict.py` |
| **Web Consensus API** | Serper.dev Google News Search API | `model/predict.py` |
| **Fact Check API** | Google Fact Check Tools API | `model/predict.py` |
| **SSRF Security** | IP resolution check blocking private/loopback/cloud ranges | `model/scrape.py` |
| **Extension Standard** | Manifest V3, CSP compliant, DOM TreeWalker highlighting | `extension/popup.js` |
