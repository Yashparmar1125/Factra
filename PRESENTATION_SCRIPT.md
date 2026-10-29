# Factra — 8-Minute Project Presentation Script (Group of 4)

> **Team Members:** Yash · Soham · Prarthana · Mrudula  
> **Format:** Exactly 2 minutes per speaker (Total: 8 minutes)  
> **Speech Target:** ~260–280 words per speaker at a steady, natural cadence (130–140 words/min)  
> **Project:** Factra — Intelligence-First Misinformation Detection Suite powered by Fine-Tuned BERT, Explainable AI, & Real-Time Web Consensus

---

## ⏱️ Presentation Timeline & Role Breakdown

```
[0:00 - 2:00]  YASH        ───>  Motivation, Flaws of Baseline Classifiers & 4-Pillar Architecture
[2:00 - 4:00]  SOHAM       ───>  Deep Learning Core, Bidirectional Self-Attention & GPU Training
[4:00 - 6:00]  PRARTHANA   ───>  Real-World Grounding, Serper News Consensus & Source Reputation
[6:00 - 8:00]  MRUDULA     ───>  Explainable AI (XAI Attention), Client Engineering & Live Demo
```

| Speaker | Member | Core Domain | Key Technical Terms to Mention |
|:---:|:---|:---|:---|
| **1** | **Yash** | Architecture & Pipeline Vision | Grounding vs. Classification, Multi-modal Ingestion, SSRF Prevention, Microservice decoupled design |
| **2** | **Soham** | Deep Learning & BERT Fine-Tuning | `bert-base-uncased`, `[CLS]` token pooling, Cross-Entropy Loss, AdamW, FP16, RTX 3050 CUDA |
| **3** | **Prarthana** | Web Consensus & Verification | Serper.dev Google News indexing, Google Fact Check Tools API, Source Reputation Matrix, Heuristic Dampening |
| **4** | **Mrudula** | XAI, Linguistic Telemetry & Demo | Layer-12 Multi-Head Attention Extraction, DOM TreeWalker Highlighting, Manifest V3, Forensic Dashboard |

---

## 🎤 Speaker 1: Yash — Problem Statement, Pitfalls of Baselines & Architecture
**Target Time:** `0:00 – 2:00` | **Word Count:** ~275 words

### 🗣️ Script:
> "Good morning respected professors and reviewers. I am **Yash**, and together with Soham, Prarthana, and Mrudula, we present **Factra** — an intelligence-first misinformation detection platform.
>
> In recent years, automated fake news detection has relied primarily on static NLP baselines like Bag-of-Words, TF-IDF with SVMs, or shallow LSTMs. However, in our literature survey, we uncovered a critical vulnerability known as **the stylistic paradox**: a deceptive actor can craft an entirely fabricated story using formal, journalistic vocabulary, completely fooling a surface-level classifier. Conversely, urgent breaking news often contains bold, emotive language and gets falsely penalized.
>
> To resolve this fundamental limitation, we designed **Factra** around a core paradigm: **Grounding over Naive Classification**. Instead of asking only *'How does this text sound?'*, our system simultaneously verifies *'Does this event exist within the verifiable real-world news ecosystem?'*
>
> Architecturally, Factra integrates four distinct engineering layers:
> 1. **Ingestion & Security:** An input engine supporting raw paragraphs and live web URLs, fortified with Server-Side Request Forgery (SSRF) filters to block private IP probing.
> 2. **Deep Learning Core:** A fine-tuned bidirectional BERT transformer that captures complex syntactic dependencies.
> 3. **The Web Consensus Engine:** Real-time cross-referencing against Google News and certified fact-checking databases.
> 4. **Client & Explainability Layer:** A Manifest V3 Chrome Extension operating directly in the browser, connected to an interactive forensic dashboard.
>
> This end-to-end synergy ensures that stylistic confidence is always validated by empirical real-world evidence. 
>
> I will now pass the stage to **Soham**, who will dive into our Deep Learning formulation and training pipeline."

---

## 🎤 Speaker 2: Soham — Deep Learning Core, Attention Mechanics & GPU Training
**Target Time:** `2:00 – 4:00` | **Word Count:** ~280 words

### 🗣️ Script:
> "Thank you, Yash. Respected panel, I am **Soham**, and I will walk you through the Deep Learning core of Factra.
>
> We chose **`bert-base-uncased`**, a transformer encoder architecture comprising 12 stacked layers, 12 self-attention heads per layer, and a 768-dimensional hidden state, totaling 110 million parameters. Unlike unidirectional models like standard RNNs or GPT-style autoregressive models that attend only left-to-right, BERT's bidirectional multi-head self-attention computes scaled dot-product attention across all tokens simultaneously:
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
> This allows the model to capture subtle contextual cues, deceptive passive phrasing, and distant negation words across a 256-token budget.
>
> For our binary classification task, we extract the final hidden state of the special `[CLS]` token, which encapsulates sequence-level semantic representations. This 768-dimensional vector passes through a dropout layer with probability $0.1$, followed by a linear projection layer outputting raw logits for two classes: *Real* ($0$) and *Fake* ($1$), normalized through a Softmax function.
>
> **Training & Optimization:**
> - We trained the network on an **NVIDIA GeForce RTX 3050 Laptop GPU** using PyTorch with CUDA 12.4 acceleration and FP16 mixed precision.
> - We optimized using **AdamW** with a learning rate of $2 \times 10^{-5}$, linear learning-rate warmup, and categorical cross-entropy loss.
> - On our balanced, stratified dataset, training converged across 374 batches in 2 epochs, reaching a validation cross-entropy loss of **0.00075** and an F1-score exceeding **0.99**.
>
> Furthermore, we corrected a crucial probability inversion: our output confidence reflects the exact probability of the predicted class, directly feeding into our credibility scoring algorithm.
>
> I will now hand over to **Prarthana** to explain how we ground these predictions using Web Consensus."

---

## 🎤 Speaker 3: Prarthana — Real-World Grounding, Consensus Engine & Source Reputation
**Target Time:** `4:00 – 6:00` | **Word Count:** ~275 words

### 🗣️ Script:
> "Thank you, Soham. I am **Prarthana**, and I will explain how Factra grounds model inferences in real-world facts using our **Web Consensus Engine**.
>
> Any pre-trained neural network suffers from a temporal cutoff; it cannot know what occurred five minutes ago. If an article claims, *'Scientists confirmed an asteroid collision today'*, BERT evaluates its linguistic structure, but cannot verify whether astronomers actually reported it.
>
> To eliminate hallucinations and temporal blind spots, Factra implements **Pillar 2: Real-Time Web Consensus**:
> 1. When an article or claim is submitted, our backend dynamically formulates a search query and queries global live news indexes using the **Serper Google Search API**.
> 2. We extract the reporting headlines, snippets, and domains, and evaluate **Coverage Density**. If a monumental breaking claim has zero footprint across global news indices, the consensus score plummets, alerting the user to an isolated fabrication.
> 3. We cross-reference every reporting outlet against our curated **Source Reputation Matrix**. Established wire services like Reuters, AP News, BBC, and Bloomberg carry high credibility multipliers, while unverified blogs receive lower trust weights.
> 4. In parallel, our system queries the **Google Fact Check Tools API**, checking whether accredited independent organizations like Snopes or PolitiFact have already investigated and debunked the specific claim.
>
> Finally, we combine BERT’s probability with Web Consensus and Source Reputation using a weighted formula: $40\%$ neural confidence, $40\%$ web consensus, and $20\%$ source reputation, modulated by heuristic penalties for clickbait and excessive punctuation.
>
> I now pass the floor to **Mrudula** to present our Explainable AI layer and the live client interfaces."

---

## 🎤 Speaker 4: Mrudula — Explainable AI (XAI), Client Engineering & Live Demo
**Target Time:** `6:00 – 8:00` | **Word Count:** ~270 words

### 🗣️ Script:
> "Thank you, Prarthana. Respected professors, I am **Mrudula**, and I will demonstrate how Factra bridges complex Deep Learning with end-user trust through **Explainable AI (XAI)** and client engineering.
>
> In high-stakes misinformation detection, users reject black-box verdicts. Factra delivers explainability on two levels:
> 1. **Neural Attention Mapping (XAI):** By enabling `output_attentions=True` during inference, we extract the 12th layer multi-head attention matrix. We average across all 12 heads and compute the attention weights linking the `[CLS]` token to every word. Tokens that spike in attention are identified as the neural network's primary decision drivers and displayed as anomalous cues.
> 2. **Linguistic Telemetry:** We compute three quantitative indices: *Tone Profiling* to detect alarmist vs. informative language, *Complexity Index* measuring sentence depth, and *Objectivity Vector* quantifying first-person emotive bias.
>
> **Client Interfaces:**
> We engineered two seamless touchpoints:
> - A **Chrome Extension (Manifest V3)** that runs directly in the user’s browser. It extracts DOM paragraph text, strips sidebars and advertisements, and uses non-destructive DOM TreeWalkers to highlight suspicious tokens right on the web page.
> - A **Forensic Intelligence Dashboard** that visualizes the SVG Credibility Gauge, consensus coverage circle, external fact-checking badges, and local scan history.
>
> *(Gesture to the screen/live demo)*
> As seen in our live running application: when we submit a claim, the backend completes inference in under 200 milliseconds on our local GPU, the gauge animates to the calculated credibility score, and the corroborating international news sources appear below.
>
> This concludes our presentation. We are now ready for the viva and technical defense questions. Thank you!"

---

## 📋 Speaker Transition & Coordination Cues

```
[Yash finishes]     ──> "I will now hand over to Soham, who will explain our Deep Learning training pipeline and mathematical formulation."
[Soham finishes]    ──> "I will now hand over to Prarthana to explain how we ground these predictions using Web Consensus."
[Prarthana finishes]──> "I now pass the floor to Mrudula to present our Explainable AI layer and the live client interfaces."
[Mrudula finishes]  ──> "This concludes our presentation. We are now ready for the viva and technical defense questions. Thank you!"
```
