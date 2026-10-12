# Factra — 8-Minute Project Presentation Script (Group of 4)

> **Team Members:** Yash · Soham · Prarthana · Mrudula  
> **Format:** 2 minutes per speaker (Total: ~8 minutes)  
> **Project:** Factra — Intelligence-First Misinformation Detection Suite powered by Fine-Tuned BERT & Real-Time Web Consensus

---

## ⏱️ Presentation Timeline Overview

| Speaker | Member | Core Domain | Key Talking Points | Duration |
|:---:|:---|:---|:---|:---:|
| **1** | **Yash** | Problem Statement & System Architecture | Problem definition, limitation of naive classifiers, high-level 4-pillar architecture | 0:00 – 2:00 |
| **2** | **Soham** | Deep Learning Core (BERT Fine-Tuning) | `bert-base-uncased`, self-attention, loss function, GPU training on RTX 3050, metrics | 2:00 – 4:00 |
| **3** | **Prarthana** | Web Consensus & Verification Engine | Serper.dev Google News indexing, Google Fact Check API, source reputation scoring | 4:00 – 6:00 |
| **4** | **Mrudula** | XAI, Linguistic Profiling & Live Demo | Attention weights for cue extraction, readability/tone metrics, extension-to-dashboard workflow | 6:00 – 8:00 |

---

## 🎤 Speaker 1: Yash — Problem Statement & Architecture Vision
**Time:** `0:00 – 2:00 (approx. 240 words)`

### 🗣️ Script:
> "Good morning respected professors and reviewers. I am **Yash**, and on behalf of our team with Soham, Prarthana, and Mrudula, we present **Factra** — an intelligence-first misinformation detection suite.
>
> Today, the digital ecosystem faces a massive proliferation of fake news. Traditional machine learning solutions treat misinformation purely as a static text-classification problem using TF-IDF or naive sentiment classifiers. The fatal flaw here is that well-written, professionally phrased fake news easily fools surface-level classifiers, while legitimate breaking news with bold words gets falsely flagged.
>
> To solve this, we engineered **Factra V2.0**. Factra does not just classify text; it performs **grounding**. It asks two critical questions:
> 1. *What linguistic and stylistic patterns does the neural network observe?*
> 2. *Is anyone else in the credible global news ecosystem actually reporting this claim?*
>
> Architecturally, Factra consists of four interconnected layers:
> - First, an ingestion pipeline supporting raw text and dynamic URL web-scraping.
> - Second, our fine-tuned transformer backbone based on **BERT**.
> - Third, an external **Web Consensus Engine** connecting to Google News through Serper API and Google Fact Check Tools.
> - And fourth, an explainability and user experience layer deployed as a **Chrome Extension** and a full **Forensic Dashboard**.
>
> I will now hand over to **Soham**, who will explain our Deep Learning training pipeline and mathematical formulation."

---

## 🎤 Speaker 2: Soham — Deep Learning Core & Model Fine-Tuning
**Time:** `2:00 – 4:00 (approx. 250 words)`

### 🗣️ Script:
> "Thank you, Yash. Respected reviewers, I am **Soham**, and I will walk you through the Deep Learning core of Factra.
>
> At the center of our pipeline is a fine-tuned **`bert-base-uncased`** transformer model. BERT uses bidirectional self-attention, which means for every token in a news headline or paragraph, it computes attention across all preceding and succeeding tokens simultaneously. This allows it to capture semantic subtleties, sarcasm, passive clickbait structures, and deceptive phrasing that unidirectional models miss.
>
> For our classification head, we take the `[CLS]` token representation from BERT's final hidden state ($768$ dimensions), pass it through a linear dropout layer, and feed it into a binary classification head with two output logits: Class 0 for *Real* and Class 1 for *Fake*.
>
> **Training specifics:**
> - We trained using the AdamW optimizer with a learning rate of $2 \times 10^{-5}$ and linear learning rate warmup.
> - We used cross-entropy loss with dynamic padding and a sequence length budget of $256$ tokens.
> - The model was trained and validated locally on an **NVIDIA GeForce RTX 3050 Laptop GPU** using mixed precision (FP16).
> - Across our stratified split of news datasets, the model achieved **high validation accuracy and an F1 score above 0.99**, with loss converging to **0.00075**.
>
> Crucially, rather than relying solely on raw softmax probabilities, we map the model's confidence into a normalized Credibility Score from 0 to 100, which is then dynamically adjusted by linguistic indicators and our consensus engine.
>
> I now invite **Prarthana** to explain the Web Consensus and Verification Engine."

---

## 🎤 Speaker 3: Prarthana — Web Consensus & Real-Time Verification
**Time:** `4:00 – 6:00 (approx. 250 words)`

### 🗣️ Script:
> "Thank you, Soham. I am **Prarthana**, and I will explain how Factra grounds model predictions against real-world news using our **Web Consensus Engine**.
>
> A standalone neural network has a knowledge cutoff and no awareness of real-time events. If an article claims *'NASA discovered life on Mars today'*, a linguistic classifier might mark it as credible purely because NASA is a formal, reputable word.
>
> To counter this, Factra introduces **Web Consensus Verification (Pillar 2)**:
> 1. When an article or URL is analyzed, our backend extracts the core claim and queries global news indexes in real-time via the **Serper.dev Google Search API**.
> 2. We extract the top reporting sources and evaluate coverage density. If a major breaking story has zero footprint across global wires, the consensus score drops sharply, triggering an *'Isolated Fabricated Claim'* warning.
> 3. We cross-reference reporting publishers against our curated **Source Reputation Database**, which scores established wire agencies like Reuters, AP News, BBC, and NPR with high trust multipliers.
> 4. We also query the **Google Fact Check Tools API** using extracted sentence claims to see if accredited independent fact-checkers like Snopes or PolitiFact have already reviewed the claim.
>
> By synthesizing model confidence ($0.4$), web consensus ($0.4$), and publisher reputation ($0.2$), Factra eliminates false positives on breaking news.
>
> I now pass the presentation to **Mrudula**, who will showcase our Explainable AI, linguistic telemetry, and a live demonstration."

---

## 🎤 Speaker 4: Mrudula — Explainable AI (XAI) & Live Demonstration
**Time:** `6:00 – 8:00 (approx. 250 words)`

### 🗣️ Script:
> "Thank you, Prarthana. I am **Mrudula**, and I will cover our Explainable AI (XAI) layer and the client interfaces.
>
> In high-stakes misinformation detection, a black-box percentage is not enough — users and moderators need to know *why* a decision was made.
>
> Factra implements **Dual-Layer Explainability**:
> - **Attention Mapping (XAI):** During the forward pass with `output_attentions=True`, we extract the attention weights between the `[CLS]` token and all input tokens from BERT's final layer. Tokens with anomalous attention spikes are tagged as influential cues.
> - **Linguistic Telemetry:** We compute three quantitative dimensions:
>   1. *Tone Profiling* — flags alarmist vs. informative tone based on sensational cue density.
>   2. *Complexity Index* — measures sentence length and structural sophistication.
>   3. *Objectivity Vector* — measures first-person subjectivity versus factual reporting.
>
> **User Deployment:**
> We engineered two intuitive entry points:
> 1. A **Chrome Extension (Manifest V3)** that allows reading any article on the web, extracting clean DOM paragraphs while stripping navigation ads, highlighting flagged suspicious terms directly in the webpage using non-destructive DOM TreeWalkers, and presenting immediate verdict metrics.
> 2. A **Deep-Dive Dashboard** featuring our animated SVG credibility vector gauge, real-time source consensus list, keyword chips, and local scan history.
>
> *(Point to screen or trigger quick demo on the dashboard)*
> As you can see, passing a real article displays high credibility and global source verification, while sensational claims trigger low credibility and highlight linguistic cues.
>
> Thank you, and we are now open for questions!"

---

## 🎯 Quick Delivery Checklist for Tomorrow
- [ ] **Yash:** Speak first, clear and authoritative. Set the big-picture context.
- [ ] **Soham:** Emphasize equations, GPU, BERT self-attention, and training details.
- [ ] **Prarthana:** Emphasize why AI alone isn't enough and how Serper API prevents hallucinations.
- [ ] **Mrudula:** Keep browser open on [http://127.0.0.1:5000/](http://127.0.0.1:5000/) and click "Analyze Now" during her segment.
