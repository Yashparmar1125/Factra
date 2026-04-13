from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data.dataset import clean_text


# Load environment variables (Basic parser without python-dotenv dependency)
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", "api", ".env")
    if not os.path.exists(env_path):
        return {}
    env_vars = {}
    with open(env_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                env_vars[key] = val
    return env_vars


_ENV = _load_env()
SERPER_API_KEY = _ENV.get("SERPER_API_KEY")
CLICKBAIT_WORDS = ["shocking", "breaking", "exclusive"]

_ALL_CAPS_RE = re.compile(r"\b[A-Z]{4,}\b")
_SENTINEL_PUNCT_RE = re.compile(r"!+")


def _find_clickbait_terms(text_lower: str) -> Tuple[List[str], int]:
    """
    Returns (unique_terms_found, total_occurrences_over_terms).
    """
    found: List[str] = []
    total = 0
    for w in CLICKBAIT_WORDS:
        # word boundary so we don't match "breaking-news" as two tokens, but still capture the word.
        pattern = re.compile(rf"(?<!\w){re.escape(w)}(?!\w)", re.IGNORECASE)
        matches = pattern.findall(text_lower)
        if matches:
            found.append(w)
            total += len(matches)
    return found, total


def _build_explanation(text_clean: str, pred_label: str, prob_real: float) -> Tuple[str, List[str]]:
    text_lower = text_clean.lower()
    suspicious_words, n_clickbait = _find_clickbait_terms(text_lower)

    exclamations = len(_SENTINEL_PUNCT_RE.findall(text_lower))
    allcaps_hits = len(_ALL_CAPS_RE.findall(text_clean))

    short_detail = len(text_lower.split()) < 80
    has_digits = any(ch.isdigit() for ch in text_clean)

    explanation_bits: List[str] = []
    explanation_bits.append(f"Model confidence that this is REAL is {prob_real:.2f}.")

    if suspicious_words:
        explanation_bits.append(
            "Contains sensational language: " + ", ".join(suspicious_words) + "."
        )
    else:
        explanation_bits.append("No strong sensational language cues detected.")

    penalties = []
    if n_clickbait > 0:
        penalties.append(f"{n_clickbait} clickbait-term occurrence(s)")
    if exclamations >= 2:
        penalties.append(f"{exclamations} exclamation group(s)")
    if allcaps_hits >= 1:
        penalties.append(f"{allcaps_hits} ALL-CAPS token(s)")
    if (short_detail and not has_digits) or (short_detail and len(text_clean) < 450):
        penalties.append("limited supporting details")

    if penalties:
        explanation_bits.append("Credibility reduced due to " + "; ".join(penalties) + ".")
    else:
        explanation_bits.append("No major credibility-dampening signals found.")

    # Heuristic phrasing requested by the user.
    if suspicious_words and (short_detail or (not has_digits)):
        explanation_bits.append("Contains sensational language and low factual density.")

    explanation = " ".join(explanation_bits).strip()
    return explanation, suspicious_words


def _compute_credibility(prob_real: float, text_clean: str) -> float:
    """
    credibility = confidence * 100, reduced by clickbait-like words.
    """
    text_lower = text_clean.lower()
    _, n_clickbait = _find_clickbait_terms(text_lower)

    exclamations = len(_SENTINEL_PUNCT_RE.findall(text_lower))
    allcaps_hits = len(_ALL_CAPS_RE.findall(text_clean))

    credibility = prob_real * 100.0

    # Clickbait penalty: saturates (keeps score in [0,100]).
    penalty_clickbait = min(40.0, 6.0 * float(n_clickbait))
    penalty_exclamations = 8.0 if exclamations >= 2 else 0.0
    penalty_allcaps = 8.0 if allcaps_hits >= 1 else 0.0

    credibility -= (penalty_clickbait + penalty_exclamations + penalty_allcaps)
    return float(max(0.0, min(100.0, credibility)))


def load_model(model_dir: str, device: str | None = None):
    if not model_dir or not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, output_attentions=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _analyze_tone(text: str) -> str:
    """Detects if news tone is Alarmist, Balanced, or Neutral."""
    alarmist_cues = ["shocking", "breaking", "insane", "unbelievable", "horrifying", "urgent", "must see"]
    text_lower = text.lower()
    score = sum(1 for w in alarmist_cues if w in text_lower)
    # Also check exclamation count
    excl = len(re.findall(r"!", text))
    if score >= 2 or excl >= 3:
        return "Alarmist / Sensational"
    if score == 1:
        return "Heightened / Intense"
    return "Neutral / Informative"


def _compute_complexity(text: str) -> str:
    """Simple readability index (Grade Level approximation)."""
    words = text.split()
    if not words: return "Minimal"
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len > 6.5: return "High (Academic/Formal)"
    if avg_len > 5.0: return "Medium (Standard News)"
    return "Low (Casual/Simplified)"


def _detect_objectivity(text: str) -> str:
    """Heuristic mapping of subjective markers."""
    subjective_markers = ["i believe", "my opinion", "in my view", "i think", "feel that", "we should"]
    text_lower = text.lower()
    score = sum(1 for m in subjective_markers if m in text_lower)
    if score >= 2:
        return "Subjective (Opinion-heavy)"
    if score == 1:
        return "Slightly Subjective"
    return "Objective (Factual framing)"


def _get_source_reputation(url: str) -> Dict[str, Any] | None:
    """Attempts to find a reputation entry for the given URL's domain."""
    if not url or not url.startswith("http"):
        return None
    
    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        here = os.path.dirname(__file__)
        data_path = os.path.join(here, "..", "api", "source_reputation.json")
        if not os.path.exists(data_path):
            return None
            
        with open(data_path, "r") as f:
            reputations = json.load(f)
            
        return reputations.get(domain)
    except Exception:
        return None


def _get_important_tokens(tokenizer, inputs, attentions, top_k: int = 5) -> List[str]:
    """
    Extracts Top-K important words based on BERT's final layer attention (average across heads).
    Only considers tokens from the original text (ignores [CLS], [SEP], and padding).
    """
    # final_layer_attn: (batch, heads, seq_len, seq_len)
    final_layer_attn = attentions[-1][0] 
    # Average across all heads
    avg_attn = final_layer_attn.mean(dim=0) # (seq_len, seq_len)
    # Get attention weights for the [CLS] token (represents global sequence importance)
    cls_attn = avg_attn[0, :] # Importance of each token to the classification
    
    input_ids = inputs["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Filter out special tokens and mapping to importance
    token_importance = []
    for i, token in enumerate(tokens):
        if token in tokenizer.all_special_tokens:
            continue
        # bert tokens starting with ## should be merged back or ignored for simplicity
        if token.startswith("##"):
            continue
        token_importance.append((token, cls_attn[i].item()))
    
    # Sort by importance and return top_k
    sorted_tokens = sorted(token_importance, key=lambda x: x[1], reverse=True)
    return [t[0] for t in sorted_tokens[:top_k]]


def _fetch_fact_checks(text: str, api_key: str | None = None) -> List[Dict[str, Any]]:
    """
    Uses Google Fact Check Explorer API to find recent debunkings.
    If no key is provided, returns an empty list.
    """
    if not api_key:
        return []
    
    # Focus on first few meaningful sentences as 'claims'
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20][:3]
    all_checks = []
    
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    for s in sentences:
        try:
            resp = requests.get(url, params={"query": s, "key": api_key, "languageCode": "en"}, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                if "claims" in data:
                    all_checks.extend(data["claims"])
        except Exception:
            continue
            
    return all_checks[:5] # Limit to top 5 hits


def _fetch_web_consensus(text: str, api_key: str | None = None) -> Dict[str, Any]:
    """
    Uses Serper.dev API to find if other sources are reporting similar stories.
    Returns { "consensus_score": float, "sources": List[Dict] }
    """
    if not api_key:
        return {"consensus_score": 0.0, "sources": [], "status": "No API Key"}
    
    # Extract keywords for query (simple: first 10 meaningful words)
    query = " ".join(re.findall(r"\w+", text))[:100]
    
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    payload = json.dumps({"q": query})

    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=2.5)
        if resp.status_code != 200:
            return {"consensus_score": 0.0, "sources": [], "status": f"Error: {resp.status_code}"}
        
        data = resp.json()
        results = data.get("organic", [])[:5]
        
        if not results:
            return {"consensus_score": 10.0, "sources": [], "status": "Low Consensus (No sources found)"}
        
        # Simple heuristic: how many "known/reputable" words in snippets?
        # In V2.0, we will use semantic similarity here.
        reputable_domains = ["bbc", "reuters", "nytimes", "apnews", "guardian", "aljazeera"]
        found_sources = []
        score_bonus = 0
        
        for r in results:
            link = r.get("link", "").lower()
            snippet = r.get("snippet", "").lower()
            title = r.get("title", "").lower()
            
            is_reputable = any(dom in link for dom in reputable_domains)
            if is_reputable:
                score_bonus += 20
            
            found_sources.append({
                "title": r.get("title"),
                "link": r.get("link"),
                "snippet": r.get("snippet"),
                "reputable": is_reputable
            })
            
        # Overall consensus score (base 40 + bonus, max 100)
        consensus_score = min(100.0, 40.0 + score_bonus + (len(results) * 5))
        
        return {
            "consensus_score": consensus_score,
            "sources": found_sources,
            "status": "Success"
        }
    except Exception as e:
        return {"consensus_score": 0.0, "sources": [], "status": f"Request Failed: {str(e)}"}


@torch.no_grad()
def predict_text(
    model,
    tokenizer,
    device: str,
    text: str,
    max_length: int = 256,
    fact_check_api_key: str | None = None,
):
    # Clean consistently with training.
    text_clean = clean_text(text)

    inputs = tokenizer(
        text_clean,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    logits = outputs.logits
    attentions = outputs.attentions
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]  # [real, fake]

    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    pred_label = "Real" if prob_real >= prob_fake else "Fake"
    confidence = prob_real  # confidence = probability of being REAL

    credibility_score = _compute_credibility(prob_real, text_clean)
    explanation, heuristic_suspicious = _build_explanation(text_clean, pred_label, prob_real)

    # Advanced Analysis
    tone = _analyze_tone(text)
    complexity = _compute_complexity(text)
    objectivity = _detect_objectivity(text)
    
    # Intelligence Upgrade: AI-Driven Suspicious Cues (XAI)
    ai_suspicious = _get_important_tokens(tokenizer, inputs, attentions)
    
    # Intelligence Upgrade: Real-Time Fact Check
    fact_checks = _fetch_fact_checks(text, fact_check_api_key)

    # Intelligence Upgrade: Source Reputation
    source_info = _get_source_reputation(text if text.startswith("http") else None)
    
    # Intelligence Upgrade: Web Consensus (V2.0)
    consensus_data = _fetch_web_consensus(text, SERPER_API_KEY)
    
    # Adjust credibility if source is known
    if source_info:
        # Heavily weigh source reputation if available (mix 50/50 with content score or override)
        credibility_score = (credibility_score * 0.4) + (source_info["score"] * 0.6)

    # Adjust credibility based on Web Consensus
    if consensus_data["status"] == "Success":
        # Consensus acts as a multiplier/dampener
        # If consensus is high, it pulls score up; if low, it potentially pulls it down
        c_score = consensus_data["consensus_score"]
        if c_score > 70:
            credibility_score = min(100.0, credibility_score + 10.0)
        elif c_score < 30:
            credibility_score = max(0.0, credibility_score - 15.0)

    # Combine heuristic and AI-detected words, prioritizing AI findings
    suspicious_words = list(set(ai_suspicious + heuristic_suspicious))

    return {
        "prediction": pred_label,
        "confidence": round(confidence, 4),
        "credibility_score": round(credibility_score, 2),
        "explanation": explanation,
        "suspicious_words": suspicious_words,
        "ai_suspicious_words": ai_suspicious,
        "fact_checks": fact_checks,
        "source_info": source_info,
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "tone": tone,
        "complexity": complexity,
        "objectivity": objectivity,
        "web_consensus": consensus_data
    }


