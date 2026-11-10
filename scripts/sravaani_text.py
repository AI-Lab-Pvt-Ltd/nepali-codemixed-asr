"""Text normalisation for fine-tuning SraVaani on Nepali / Nepali-English code-mixed ASR data.

Target convention (see reports/text_normalization.md):
  * Unicode NFKC; Nepali in Devanagari; English in Latin script, LOWERCASE.
  * No punctuation; hyphen/dash between words -> space; apostrophes inside words deleted.
  * Numbers / maths symbols spelled out as words (Nepali Devanagari or English Latin);
    no digits remain.
  * Single spaces, stripped, non-empty, and exactly encodable by the SraVaani
    SentencePiece tokenizer: no <unk>, and decode(encode(t)) == t.

Public API
  clean(text)                -> str   normalisation except numbers/maths spans (left verbatim)
  has_numeric(text)          -> bool  text contains a span that needs verbalising
                                      (digits, %, super/subscripts, + = ^ / $ # & @ ! etc.)
  candidates(text)           -> list[str]   (raises Unhandled for unhandled text)
  candidates_with_reason(t)  -> (list[str], reason or None)
  segments_with_reason(t)    -> (list[str | list[str]], reason)  per-span alternatives
  romanize(text)             -> coarse Devanagari->Latin phonetic key (Latin passes through)
  best_candidate(text, hyp)  -> (best_text, distance)  pick/compose the candidate closest to hyp
  is_encodable(text)         -> bool
  get_tokenizer()            -> lazily loaded sentencepiece processor
"""
from __future__ import annotations

import itertools
import os
import re
import unicodedata

from num2words import num2words

__all__ = [
    "clean", "has_numeric", "candidates", "candidates_with_reason", "segments_with_reason",
    "romanize", "best_candidate", "is_encodable", "is_final", "get_tokenizer", "Unhandled", "numeric_spans",
    "ne_int", "en_int", "MAX_CANDIDATES",
]

MAX_CANDIDATES = 32

_DEFAULT_TOK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "sravaani",
                            "tokenizer.model")
TOKENIZER_PATH = os.environ.get("SRAVAANI_TOKENIZER", os.path.normpath(_DEFAULT_TOK))
_SP = None


def get_tokenizer():
    global _SP
    if _SP is None:
        import sentencepiece as spm
        _SP = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    return _SP


def is_encodable(text: str) -> bool:
    """True iff text is non-empty, has no <unk> pieces and round-trips exactly."""
    if not text or not text.strip():
        return False
    sp = get_tokenizer()
    ids = sp.encode(text)
    unk = sp.unk_id()
    if unk in ids:
        return False
    return sp.decode(ids) == text


class Unhandled(ValueError):
    pass


# ---------------------------------------------------------------------------------------------
# Nepali number words.  Spellings follow the standard Nepali table; where the corpus itself uses
# a number word, its most frequent spelling (count >= 2, genuine number usage) is used — see
# reports/text_normalization.md for the counts.
# ---------------------------------------------------------------------------------------------
NE_0_99 = [
    "शून्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ",                      # 0-9
    "दस", "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार", "उन्नाइस",       # 10-19
    "बीस", "एक्काइस", "बाइस", "तेइस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताइस", "अट्ठाइस", "उनन्तीस",  # 20-29
    "तीस", "एकतीस", "बत्तीस", "तेत्तीस", "चौँतीस", "पैँतीस", "छत्तीस", "सैँतीस", "अठतीस", "उनन्चालीस",  # 30-39
    "चालीस", "एकचालीस", "बयालीस", "त्रिचालीस", "चवालीस", "पैँतालीस", "छयालीस", "सतचालीस", "अठचालीस", "उनन्चास",  # 40-49
    "पचास", "एकाउन्न", "बाउन्न", "त्रिपन्न", "चउन्न", "पचपन्न", "छपन्न", "सन्ताउन्न", "अन्ठाउन्न", "उनन्साठी",  # 50-59
    "साठी", "एकसट्ठी", "बयसट्ठी", "त्रिसट्ठी", "चौसठ्ठी", "पैँसट्ठी", "छयसट्ठी", "सतसट्ठी", "अठसट्ठी", "उनन्सत्तरी",  # 60-69
    "सत्तरी", "एकहत्तर", "बहत्तर", "त्रिहत्तर", "चौहत्तर", "पचहत्तर", "छयहत्तर", "सतहत्तर", "अठहत्तर", "उनासी",  # 70-79
    "असी", "एकासी", "बयासी", "त्रियासी", "चौरासी", "पचासी", "छयासी", "सतासी", "अठासी", "उनान्नब्बे",  # 80-89
    "नब्बे", "एकानब्बे", "बयानब्बे", "त्रियानब्बे", "चौरानब्बे", "पन्चानब्बे", "छयानब्बे", "सन्तानब्बे", "अन्ठानब्बे", "उनान्सय",  # 90-99
]
assert len(NE_0_99) == 100
NE_SAY = "सय"          # 10^2
NE_HAJAR = "हजार"      # 10^3
NE_LAKH = "लाख"        # 10^5
NE_KAROD = "करोड"      # 10^7
NE_ARBA = "अर्ब"        # 10^9
NE_KHARBA = "खरब"      # 10^11 (corpus: खरब 4 vs खर्ब 1)
NE_DECIMAL = "दशमलव"
NE_PERCENT = "प्रतिशत"
NE_ORD_1_10 = ["", "पहिलो", "दोस्रो", "तेस्रो", "चौथो", "पाँचौँ", "छैटौँ", "सातौँ", "आठौँ", "नवौँ", "दसौँ"]
NE_HALF = "आधा"
NE_DEDH = "डेढ"        # 1.5
NE_ADHAI = "अढाई"      # 2.5
NE_SADHE = "साढे"      # n.5 (n>=3)
NE_SAWA = "सवा"        # n:15
NE_PAUNE = "पौने"      # n:45 -> (n+1) minus a quarter
NE_BATA = "बटा"        # fraction "a बटा b"
NE_DEKHI = "देखि"      # range "a देखि b"

EN_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

_DEV_VOWEL_SIGNS = set("ािीुूृॄॅॆेैॉॊोौॢॣ")
_DEV_INDEP_VOWELS = set("अआइईउऊऋॠऌएऐऍऎओऔऑऒॲ")


def ne_int(n: int) -> str:
    """Nepali (Indian-system) cardinal in Devanagari words."""
    if n < 0:
        raise ValueError(n)
    if n < 100:
        return NE_0_99[n]
    parts = []
    for val, name in ((10 ** 11, NE_KHARBA), (10 ** 9, NE_ARBA), (10 ** 7, NE_KAROD), (10 ** 5, NE_LAKH),
                      (10 ** 3, NE_HAJAR), (10 ** 2, NE_SAY)):
        if n >= val:
            q, n = divmod(n, val)
            parts.append(f"{ne_int(q)} {name}")
    if n:
        parts.append(NE_0_99[n])
    return " ".join(parts)


def ne_hundreds(n: int) -> str | None:
    """'Fifteen-hundred' style for 1100..9999 with non-zero hundreds digit: 1988 -> उन्नाइस सय अठासी."""
    if 1100 <= n <= 9999 and (n // 100) % 10 != 0:
        s = f"{NE_0_99[n // 100]} {NE_SAY}"
        if n % 100:
            s += " " + NE_0_99[n % 100]
        return s
    return None


def ne_ordinal(n: int) -> str:
    if 1 <= n <= 10:
        return NE_ORD_1_10[n]
    w = ne_int(n)
    last = w[-1]
    if last in _DEV_VOWEL_SIGNS or last in _DEV_INDEP_VOWELS:
        return w + "औँ"
    return w + "ौँ"


def ne_digits(s: str) -> str:
    return " ".join(NE_0_99[int(c)] for c in s)


def _en_clean(s: str, keep_and: bool) -> str:
    s = s.replace(",", " ").replace("-", " ")
    toks = s.split()
    if not keep_and:
        toks = [t for t in toks if t != "and"]
    return " ".join(toks).lower()


def en_int(n: int, keep_and: bool = False) -> str:
    return _en_clean(num2words(n), keep_and)


def en_year(n: int, keep_and: bool = False) -> str:
    return _en_clean(num2words(n, to="year"), keep_and)


def en_lakh(n: int) -> str:
    return _en_clean(num2words(n, lang="en_IN"), False)


def en_ordinal(n: int) -> str:
    return _en_clean(num2words(n, to="ordinal"), False)


def en_digits(s: str) -> str:
    return " ".join(EN_DIGITS[int(c)] for c in s)


def _plural(phrase: str) -> str:
    words = phrase.split()
    w = words[-1]
    if w.endswith("y"):
        w = w[:-1] + "ies"
    elif w.endswith(("x", "s")):
        w = w + "es"
    else:
        w = w + "s"
    return " ".join(words[:-1] + [w])


def _dedupe(alts):
    seen, out = set(), []
    for t, st in alts:
        t = " ".join(t.split())
        if t not in seen:
            seen.add(t)
            out.append((t, st))
    return out


# ---------------------------------------------------------------------------------------------
# Alternatives for numeric spans.  Each alternative is (text, style) with style "ne" or "en".
# ---------------------------------------------------------------------------------------------
def int_alts(ds: str, indian_grouping: bool = False):
    """Alternatives for an unsigned integer given as a digit string."""
    n = int(ds)
    L = len(ds)
    if L > 1 and ds[0] == "0":                       # 00, 007, 046 (= BS 2046) ... digit by digit / value
        digits = [(ne_digits(ds), "ne"), (en_digits(ds), "en")]
        if L <= 3:
            digits.append(("oh " + en_digits(ds[1:]), "en"))
        if n == 0:
            return _dedupe(digits + [(ne_int(0), "ne"), (en_int(0), "en")])
        return _dedupe(int_alts(str(n)) + digits)
    ne, en, en_and = ne_int(n), en_int(n), en_int(n, True)
    bare = _bare_alts(n, ne, en, en_and)          # 100 -> "सय" / "hundred", 1000 -> "हजार" / "thousand"
    if L <= 2:
        if L == 2 and set(ds) <= {"0", "1"}:          # binary-looking 10 / 11 -> also "one zero"
            return [(ne, "ne"), (en, "en"), (en_digits(ds), "en"), (ne_digits(ds), "ne")]
        return [(ne, "ne"), (en, "en")]
    if L == 3:
        alts = [(ne, "ne"), (en, "en")] + bare + [(en_and, "en")]
        if ds[1] != "0":                           # 130 -> "one thirty", 365 -> "three sixty five"
            # (Nepali "एक तीस" was tried but loses to "एक सय तीस" on Bengali-script hypotheses that merge सय)
            alts += [(f"{en_int(n // 100)} {en_int(n % 100)}", "en")]
        alts += [(en_digits(ds), "en"), (ne_digits(ds), "ne")]
        return _dedupe(alts)
    if L == 4:
        nh = ne_hundreds(n)
        ey, ey_and = en_year(n), en_year(n, True)
        if 1100 <= n <= 1999:                          # years / "fifteen hundred"
            alts = [(nh, "ne"), (ne, "ne"), (ey, "en"), (en, "en"), (ey_and, "en"), (en_and, "en")] + bare
        elif 2000 <= n <= 2099:
            alts = [(ne, "ne"), (ey, "en"), (en, "en"), (en_and, "en")]
            if nh:
                alts.insert(1, (nh, "ne"))
        else:
            alts = [(ne, "ne"), (en, "en"), (en_and, "en")]
            if nh:
                alts.insert(1, (nh, "ne"))
            alts.append((ey, "en"))
        if (1000 <= n <= 1099 or 2000 <= n <= 2099) and n % 100 >= 10:   # 1010 -> "दस दस", 2022 -> "बीस बाइस"
            alts.append((f"{NE_0_99[n // 100]} {NE_0_99[n % 100]}", "ne"))
        if 1000 <= n <= 1099:
            alts += bare
        alts += [(en_digits(ds), "en"), (ne_digits(ds), "ne")]
        return _dedupe([a for a in alts if a[0]])
    # 5+ digits
    if L >= 11:                                              # ids / account numbers: digits only
        return _dedupe([(en_digits(ds), "en"), (ne_digits(ds), "ne")])
    if n >= 100000:
        lk = (en_lakh(n), "en")
        card = [(ne, "ne")] + ([lk, (en, "en")] if indian_grouping else [(en, "en"), lk]) + [(en_and, "en")]
    else:
        card = [(ne, "ne"), (en, "en"), (en_and, "en")]
    digits = [(en_digits(ds), "en"), (ne_digits(ds), "ne")]
    if L >= 7 and n % 1000 != 0 and not indian_grouping:     # phone numbers / ids: digit by digit first
        return _dedupe(digits + card)
    return _dedupe(card + digits)


def _bare_alts(n, ne, en, en_and):
    """Readings without the leading one/एक: 100-199 ("सय पचास", "hundred fifty"), 1000-1999."""
    out = []
    if 100 <= n <= 199 or 1000 <= n <= 1999:
        for full, st in ((ne, "ne"), (en, "en"), (en_and, "en")):
            for one in ("एक ", "one "):
                if full.startswith(one) and full.split()[1] in ("सय", "हजार", "hundred", "thousand"):
                    out.append((full[len(one):], st))
    return out


def decimal_alts(ip: str, fp: str):
    i = int(ip)
    ne_i, en_i = ne_int(i), en_int(i)
    if len(ip) > 1 and ip[0] == "0":
        ne_i, en_i = ne_digits(ip), en_digits(ip)
    ne_f, en_f = ne_digits(fp), en_digits(fp)
    alts = [(f"{ne_i} {NE_DECIMAL} {ne_f}", "ne"), (f"{en_i} point {en_f}", "en"),
            (f"{ne_i} point {ne_f}", "ne")]
    if i == 0:
        alts.append((f"point {en_f}", "en"))
    alts += [(f"{ne_i} {ne_f}", "ne"), (f"{en_i} {en_f}", "en")]          # 19.03 -> "उन्नाइस शून्य तीन"
    if len(fp) >= 2 and fp[0] != "0" and len(fp) <= 3:                    # 1.37 -> "एक सैँतीस" / "one thirty seven"
        ne_n, en_n = ne_int(int(fp)), en_int(int(fp))
        alts += [(f"{ne_i} {ne_n}", "ne"), (f"{en_i} {en_n}", "en"), (f"{ne_i} {NE_DECIMAL} {ne_n}", "ne"),
                 (f"{en_i} point {en_n}", "en")]
    if fp == "5":
        if i == 1:
            alts.append((NE_DEDH, "ne"))
        elif i == 2:
            alts.append((NE_ADHAI, "ne"))
        elif i >= 3:
            alts.append((f"{NE_SADHE} {ne_i}", "ne"))
    return _dedupe(alts)


def _bare(alts, style):
    """Primary reading of a style, but without a leading one/एक before सय/हजार/hundred/thousand if offered."""
    p = _primary(alts, style)
    w = p.split()
    if len(w) >= 2 and w[0] in ("एक", "one") and w[1] in ("सय", "हजार", "hundred", "thousand"):
        q = " ".join(w[1:])
        if any(t == q for t, _ in alts):
            return q
    return p


def _primary(alts, style):
    for t, st in alts:
        if st == style:
            return t
    return alts[0][0]


def number_alts(s: str):
    """Alternatives + pattern label for a raw number token (digits with optional , and .)."""
    if "," in s:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
            return number_alts(s.replace(",", ""))[0], "thousands_sep"
        if re.fullmatch(r"\d{1,2}(?:,\d{2})*,\d{3}(?:\.\d+)?", s):
            body = s.replace(",", "")
            if "." in body:
                return number_alts(body)[0], "indian_sep"
            return int_alts(body, indian_grouping=True), "indian_sep"
        return None, "comma_list"
    if s.count(".") == 1:
        ip, fp = s.split(".")
        return decimal_alts(ip, fp), "decimal"
    if s.count(".") >= 2:
        parts = s.split(".")
        en = [en_int(int(p)) if not (len(p) > 1 and p[0] == "0") else en_digits(p) for p in parts]
        ne = [ne_int(int(p)) if not (len(p) > 1 and p[0] == "0") else ne_digits(p) for p in parts]
        return _dedupe([(" point ".join(en), "en"), (" dot ".join(en), "en"), (" point ".join(ne), "ne")]), "version"
    L = len(s)
    if L > 1 and s[0] == "0":
        lab = "leading_zero"
    elif L <= 2:
        lab = "int_1-2digit"
    elif L == 3:
        lab = "int_3digit"
    elif L == 4:
        lab = "int_4digit_1000-2099" if int(s) <= 2099 else "int_4digit_other"
    elif L <= 6:
        lab = "int_5-6digit"
    else:
        lab = "int_7+digit"
    return int_alts(s), lab


# operator alternatives
ALT_EQ = [("equals to", "en"), ("is equal to", "en"), ("equals", "en"), ("equal to", "en"), ("बराबर", "ne"),
          ("", "en")]
ALT_EQ_MULT = ALT_EQ + [("जा", "ne")]          # multiplication facts "2 x 3 = 6" are recited "टू थ्री जा सिक्स"
ALT_EQEQ = [("equals equals", "en"), ("double equals", "en"), ("equals to", "en")]
ALT_POW = [("to the power", "en"), ("को power", "ne"), ("power", "en"), ("raise to", "en")]
ALT_DIV = [("by", "en"), ("divided by", "en"), ("over", "en"), ("upon", "en"), (NE_BATA, "ne")]
ALT_TIMES = [("into", "en"), ("x", "en"), ("times", "en"), ("", "en")]
ALT_SQ = [("square", "en"), ("को square", "ne")]
ALT_CUBE = [("cube", "en"), ("को cube", "ne")]
ALT_PCT = [("percent", "en"), (NE_PERCENT, "ne"), ("modulus", "en"), ("mod", "en")]
ALT_AT = [("at", "en"), ("at the rate", "en")]
ALT_SUB0 = [("not", "en"), ("zero", "en"), ("naught", "en"), (NE_0_99[0], "ne")]
ALT_DOT = [("dot", "en"), ("", "en")]
ALT_POINT_OR_GAP = [("point", "en"), ("", "en")]
ALT_SLASH_LAT = [("", "en"), ("by", "en"), ("per", "en"), ("slash", "en")]
ALT_PER = [("per", "en"), ("", "en")]
ALT_MINUS_OR_GAP = [("minus", "en"), ("", "en"), ("dash", "en")]
ALT_MINUS_STANDALONE = [("minus", "en"), ("", "ne"), ("dash", "en")]
ALT_PRIME = [("dash", "en"), ("", "en"), ("prime", "en")]
ALT_NEG = [("minus", "en"), ("negative", "en")]
ALT_FACT = [("factorial", "en"), ("", "en")]
ALT_STAR = [("into", "en"), ("times", "en"), ("star", "en")]
SYMBOL_WORDS = {  # deterministic verbalisations of rare symbols
    "&": "and", "×": "into", "÷": "divided by", "√": "root", "π": "pi", "θ": "theta", "°": "degree",
    "∞": "infinity", "≠": "not equal to", "≤": "less than or equal to", "≥": "greater than or equal to",
    "<": "less than", ">": "greater than", "±": "plus minus", "∆": "delta", "Δ": "delta", "α": "alpha",
    "β": "beta", "λ": "lambda", "μ": "mu", "σ": "sigma", "Ω": "ohm", "ω": "omega", "₹": "rupees",
}

# ---------------------------------------------------------------------------------------------
# Pre-normalisation
# ---------------------------------------------------------------------------------------------
_SUP = {"⁰": 0, "¹": 1, "²": 2, "³": 3, "⁴": 4, "⁵": 5, "⁶": 6, "⁷": 7, "⁸": 8, "⁹": 9}
_SUB = {"₀": 0, "₁": 1, "₂": 2, "₃": 3, "₄": 4, "₅": 5, "₆": 6, "₇": 7, "₈": 8, "₉": 9}
SUP_BASE, SUP_MINUS, SUP_N = 0xE000, "\uE00A", "\uE00C"
SUB_BASE = 0xE010
INV = "\uE020"
CUR_RS = "\uE030"          # "Rs 100" / "रु. 100" currency prefix
ONE_CL = "\uE031"          # "1" directly before the classifier वटा/टा/ओटा (-> एउटा)
PRIME = "\uE032"           # prime mark after a variable
CL_SUFFIXES = ["मात्रै", "मात्र", "भन्दा", "लाई", "बाट", "सँग", "को", "का", "की", "ले", "मा", "नै", "हरू", "हरु"]
_PRIV_BACK = {chr(SUP_BASE + v): k for k, v in _SUP.items()}
_PRIV_BACK.update({chr(SUB_BASE + v): k for k, v in _SUB.items()})
_PRIV_BACK.update({SUP_MINUS: "⁻", SUP_N: "ⁿ", INV: "⁻¹", CUR_RS: "rs ", ONE_CL: "1 ", PRIME: "'"})

_DEV_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_DEV = "ऀ-ॿ"


def _prenorm(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = re.sub(r"[-−⁻]¹", INV, t)
    t = "".join(chr(SUP_BASE + _SUP[c]) if c in _SUP else chr(SUB_BASE + _SUB[c]) if c in _SUB
                else SUP_MINUS if c == "⁻" else SUP_N if c == "ⁿ" else _DEV_DIGITS.get(c, c) for c in t)
    t = unicodedata.normalize("NFKC", t)
    # drop format / control characters (LRM, ZWJ, ZWNJ, BOM, soft hyphen ...)
    t = "".join(c for c in t if unicodedata.category(c) not in ("Cf", "Cc") or c in "\t\n\r")
    t = re.sub(r"(?i)(?<![A-Za-z])(?:n?rs|inr|npr)\.?\s*(?=\d)", CUR_RS, t)
    t = re.sub(r"(?<![\u0900-\u097F])रु\.\s*(?=\d)", CUR_RS, t)
    t = re.sub(r"(?<![\w.,:/^\-])1\s*(?=(?:ओटा|वटा|टा)(?:%s)?(?![\u0900-\u097F]))" % "|".join(CL_SUFFIXES), ONE_CL, t)
    t = re.sub(r"(?<=\d)\s+(?=औ[ँं]?(?![\u0900-\u097F]))", "", t)    # "16 औँ" -> "16औँ" (Nepali ordinal)
    t = re.sub("[।॥॰]", " ", t)
    # apostrophes / quotes;  prime notation X' / YC' / W'Z' / x'z -> PRIME (spoken "dash")
    t = re.sub(r"(?<![A-Za-z])([A-Za-z]{1,3})[’′'](?=$|[^A-Za-z]|[A-Z])", lambda m: m.group(1) + PRIME, t)
    t = re.sub(r"(?<![A-Za-z])([A-HJ-NP-Za-hj-np-xz])[’′'](?=[a-z](?![a-z]))", lambda m: m.group(1) + PRIME, t)
    t = re.sub(r"(?<=[A-Za-z])['’ʼ`](?=[A-Za-z])", "", t)        # don't -> dont, teacher's -> teachers
    t = re.sub(r"(?<=\d)['’ʼ](?=s(?![A-Za-z]))", "", t)          # 1990's -> 1990s, 1's -> 1s
    t = re.sub(rf"(?<=[{_DEV}])['’ʼ](?=[{_DEV}])", "", t)
    t = re.sub(r"['’‘ʼ`\"“”«»„]", " ", t)
    # dashes / hyphens
    t = t.replace("−", "-").replace("‐", "-").replace("‑", "-")
    t = re.sub(r"(?<=\d)[–—](?=\d)", "-", t)
    t = re.sub(r"[–—―]", " ", t)
    t = re.sub(rf"(?<=[A-Za-z{_DEV}])-+(?=[A-Za-z{_DEV}])", " ", t)   # multi-dimensional -> multi dimensional
    t = re.sub(rf"(?<=[{_DEV}])/(?=[{_DEV}])", " ", t)              # कारण/आधार -> कारण आधार
    return " ".join(t.split())


# ---------------------------------------------------------------------------------------------
# Chunk parser.  A chunk is a whitespace-free piece of the pre-normalised text.  It is turned into
# a list of items:
#   ("lit", text, src, verbal, glued_to_prev)
#   ("alt", [(text, style), ...], src, True, glued_to_prev, pattern)
#   ("sp",)                  -> separator (removed punctuation)
#   ("glue",)                -> join neighbours without space (abbreviations like I.Sc)
# ---------------------------------------------------------------------------------------------
_TOK_RE = re.compile(
    rf"(?P<num>\d+(?:[.,]\d+)*)|(?P<lat>[A-Za-z]+)|(?P<dev>[{_DEV}]+)"
    r"|(?P<sup>[\uE000-\uE00C]+)|(?P<sub>[\uE010-\uE019]+)|(?P<inv>\uE020)|(?P<cur>\uE030)|(?P<one>\uE031)|(?P<prime>\uE032)|(?P<o>.)", re.S)
_ORD_RE = re.compile(r"(?i)st|nd|rd|th")
_MATHY_RE = re.compile(r"[0-9\uE000-\uE020+=^/%*()]")


def _mathy(chunk):
    """Neighbour of a standalone '-' looks like maths: digits/operators or a short Latin variable."""
    if not chunk:
        return False
    if _MATHY_RE.search(chunk):
        return True
    return re.fullmatch(r"[A-Za-z]{1,3}[.,?]?", chunk) is not None


class _Item:
    __slots__ = ("kind", "text", "alts", "src", "verbal", "glued", "pat", "chunk")

    def __init__(self, kind, text=None, alts=None, src="", verbal=False, glued=False, pat=None):
        self.kind, self.text, self.alts, self.src = kind, text, alts, src
        self.verbal, self.glued, self.pat = verbal, glued, pat
        self.chunk = None


# words after "1" where "एउटा" (one thing) is not a plausible reading: measure words, scale words,
# copulas/conjunctions, units and operators
_NO_EUTA_DEV = ("जना", "वर्ष", "बर्ष", "दिन", "महिना", "हप्ता", "घण्टा", "घन्टा", "मिनेट", "मिनट", "सेकेन्ड", "बजे", "गते",
                "पटक", "पल्ट", "चोटि", "चोटी", "सय", "हजार", "लाख", "करोड", "अर्ब", "खरब", "प्रतिशत", "साल", "रुपैयाँ",
                "रूपैयाँ", "किलो", "मिटर", "लिटर", "कक्षा", "डिग्री", "देखि", "सम्म", "भन्दा", "र", "वा", "अथवा", "छ", "छन्",
                "हो", "होइन", "भयो", "हुन्छ", "हुने", "भनेको", "भन्ने", "भनेर", "नै", "पनि", "त", "है", "आयो", "आउँछ", "जोड",
                "गुणा", "भाग", "घटाउ", "बराबर", "ओटा", "वटा", "टा", "औ", "न", "नि", "अनि", "भए", "भएको", "गर्दा")
_NO_EUTA_LAT = {"second", "seconds", "sec", "hour", "hours", "hr", "minute", "minutes", "min", "year", "years", "day",
                "days", "month", "months", "week", "weeks", "mark", "marks", "byte", "bytes", "bit", "bits", "kg", "km",
                "cm", "mm", "kb", "mb", "gb", "to", "is", "into", "and", "or", "minus", "plus", "by", "times", "nf",
                "point", "percent", "lakh", "crore", "million", "billion", "thousand", "hundred", "dollar", "dollars",
                "rupees", "rs", "degree", "degrees", "unit", "units", "the", "of", "equals", "equal", "over", "upon",
                "power", "square", "cube", "root", "st", "nd", "rd", "th", "am", "pm", "base", "complement", "bar"}


def _euta_ok(chunk):
    """'1' followed by this word may be spoken एउटा (a noun / postposition follows, not a measure word)."""
    if not chunk:
        return False
    m = re.match(rf"[{_DEV}]+|[A-Za-z]+", chunk)
    if not m:
        return False
    w = m.group()
    if w[0].isascii():
        return len(w) > 1 and w.lower() not in _NO_EUTA_LAT
    return not any(w == x or (len(x) > 1 and w.startswith(x) and x not in ("र", "न", "त", "छ", "हो", "भए", "औ"))
                   for x in _NO_EUTA_DEV)


def _src(s):
    return "".join(_PRIV_BACK.get(c, c) for c in s)


def _parse_chunk(chunk: str, prev_chunk: str | None, next_chunk: str | None, items: list, pats: list,
                 strict: bool = True, prev2_chunk: str | None = None):
    toks = [(m.lastgroup, m.group()) for m in _TOK_RE.finditer(chunk)]
    n = len(toks)
    i = 0
    first_item = True

    def add(item):
        nonlocal first_item
        item.glued = not first_item and item.kind in ("lit", "alt")
        item.chunk = chunk
        first_item = False
        items.append(item)
        if item.pat:
            pats.append(item.pat)

    def lit(text, src=None, verbal=False, pat=None):
        add(_Item("lit", text=text, src=src if src is not None else text, verbal=verbal, pat=pat))

    def alt(alts, src, pat):
        alts = _dedupe(alts)
        if len(alts) == 1:
            add(_Item("lit", text=alts[0][0], src=src, verbal=True, pat=pat))
        else:
            add(_Item("alt", alts=alts, src=src, verbal=True, pat=pat))

    def sep():
        nonlocal first_item
        items.append(_Item("sp"))
        first_item = True

    def T(k):
        return toks[k] if 0 <= k < n else (None, "")

    while i < n:
        ty, s = toks[i]
        pty, ps = T(i - 1)
        nty, ns = T(i + 1)
        if ty == "num":
            # ---- composite numeric patterns ----
            # range a-b(-c...)
            if ns == "-" and T(i + 2)[0] == "num" and not (pty == "o" and ps == "-"):
                parts, j = [s], i + 2
                while T(j)[0] == "num":
                    parts.append(T(j)[1])
                    if T(j + 1)[1] == "-" and T(j + 2)[0] == "num":
                        j += 2
                    else:
                        break
                pa = [number_alts(p)[0] for p in parts]
                if any(a is None for a in pa):
                    pa = [int_alts(p.replace(",", "")) for p in parts]
                ne = [_primary(a, "ne") for a in pa]
                en = [_primary(a, "en") for a in pa]
                alts = [(" ".join(ne), "ne"), (f" {NE_DEKHI} ".join(ne), "ne"), (" ".join(en), "en"),
                        (" to ".join(en), "en")]
                ne_b = [_bare(a, "ne") for a in pa]
                en_b = [_bare(a, "en") for a in pa]
                if ne_b != ne:                                   # 100-200 -> "सय दुई सय"
                    alts += [(" ".join(ne_b), "ne"), (f" {NE_DEKHI} ".join(ne_b), "ne")]
                if en_b != en:
                    alts += [(" ".join(en_b), "en"), (" to ".join(en_b), "en")]
                alt(alts, "-".join(parts), "range")
                i = j + 1
                continue
            # time / ratio a:b
            if ns == ":" and T(i + 2)[0] == "num" and re.fullmatch(r"\d+", s) and re.fullmatch(r"\d+", T(i + 2)[1]):
                a, b = s, T(i + 2)[1]
                ia, ib = int(a), int(b)
                if ia <= 24 and len(b) == 2 and ib < 60:
                    if ib == 0:
                        alts = [(ne_int(ia), "ne"), (en_int(ia), "en"), (en_int(ia) + " o clock", "en"),
                                (f"{ne_int(ia)} {ne_int(0)} {ne_int(0)}", "ne"), (f"{en_int(ia)} hundred", "en")]
                    else:
                        en_b = en_int(ib) if ib >= 10 else "oh " + en_int(ib)
                        alts = [(f"{ne_int(ia)} {ne_int(ib)}", "ne"), (f"{en_int(ia)} {en_b}", "en")]
                        if ib == 30:
                            alts.append((f"{NE_SADHE} {ne_int(ia)}", "ne"))
                        elif ib == 15:
                            alts.append((f"{NE_SAWA} {ne_int(ia)}", "ne"))
                        elif ib == 45:
                            alts.append((f"{NE_PAUNE} {ne_int(ia + 1)}", "ne"))
                    alt(alts, f"{a}:{b}", "time")
                else:
                    aa, ba = int_alts(a), int_alts(b)
                    alt([(f"{_primary(aa, 'ne')} {_primary(ba, 'ne')}", "ne"),
                         (f"{_primary(aa, 'en')} {_primary(ba, 'en')}", "en"),
                         (f"{_primary(aa, 'en')} is to {_primary(ba, 'en')}", "en")], f"{a}:{b}", "ratio")
                i += 3
                continue
            # fraction a/b
            if ns == "/" and T(i + 2)[0] == "num" and T(i + 3)[1] != "/":
                a, b = s, T(i + 2)[1]
                aa, ba = number_alts(a)[0], number_alts(b)[0]
                if aa is not None and ba is not None:
                    ane, aen, bne, ben = _primary(aa, "ne"), _primary(aa, "en"), _primary(ba, "ne"), _primary(ba, "en")
                    alts = [(f"{aen} by {ben}", "en"), (f"{ane} {NE_BATA} {bne}", "ne"), (f"{aen} over {ben}", "en"),
                            (f"{aen} {ben}", "en"), (f"{ane} {bne}", "ne"), (f"{aen} divided by {ben}", "en"),
                            (f"{aen} upon {ben}", "en")]
                    if a == "1" and b == "2":
                        alts += [("half", "en"), (NE_HALF, "ne")]
                    if T(i + 3)[0] == "lat" and T(i + 3)[1].lower() in ("th", "st", "nd", "rd") and re.fullmatch(r"\d+", b):
                        ordw = en_ordinal(int(b))
                        alts = [(f"{aen} {ordw if a == '1' else _plural(ordw)}", "en")] + alts
                        alt(alts, f"{a}/{b}{T(i + 3)[1]}", "fraction")
                        i += 4
                        continue
                    alt(alts, f"{a}/{b}", "fraction")
                    i += 3
                    continue
            base, lab = number_alts(s)
            if base is None:                               # "5,13" -> list of numbers
                for k, p in enumerate(s.split(",")):
                    if k:
                        sep()
                    a, l2 = number_alts(p)
                    alt(a, p, "comma_list")
                i += 1
                continue
            # percent
            if ns == "%":
                alts = []
                for t_, st in base:
                    if st == "ne":
                        alts += [(f"{t_} {NE_PERCENT}", "ne"), (f"{t_} percent", "ne")]
                    else:
                        alts.append((f"{t_} percent", "en"))
                alt(alts, s + "%", "percent")
                i += 2
                continue
            # ordinal 1st 2nd 3rd 4th
            if nty == "lat" and _ORD_RE.fullmatch(ns) and re.fullmatch(r"\d+", s):
                v = int(s)
                alt([(en_ordinal(v), "en"), (ne_ordinal(v), "ne")], s + ns, "ordinal")
                i += 2
                continue
            # Nepali ordinal suffix glued to digits: 10औँ / १०औं
            if nty == "dev" and re.match("औ[ँं]?", ns) and re.fullmatch(r"\d+", s):
                v = int(s)
                rest = re.sub("^औ[ँं]?", "", ns)
                alt([(ne_ordinal(v), "ne"), (en_ordinal(v), "en")], s + ns[:len(ns) - len(rest)], "ordinal")
                if rest:
                    lit(rest)
                i += 2
                continue
            # decades / plurals 1990s 20s 1s
            if nty == "lat" and ns in ("s", "S") and re.fullmatch(r"\d+", s):
                v = int(s)
                if 1000 <= v <= 9999:
                    w = en_year(v)
                elif v == 0:
                    w = "zero"
                else:
                    w = en_int(v)
                alt([(_plural(w), "en")], s + ns, "decade_plural")
                i += 2
                continue
            # plain number (possibly glued to letters: S3, 4K, 2x, IPv4 ...)
            if s == "1" and n == 1 and _euta_ok(next_chunk):           # "1 पसलले" / "1 example" -> एउटा
                base = base + [("एउटा", "ne")]
            if pty == "lat" or nty == "lat":
                lab = "alnum"
                if s == "0" and pty == "lat":                    # Q0, x0 -> "q not"
                    base = base + [("not", "en"), ("naught", "en")]
            elif nty == "dev":
                lab = "glued_devanagari"
            alt(base, s, lab)
            i += 1
            continue
        if ty == "lat":
            if len(s) == 1 and ns == "." and T(i + 2)[0] == "lat" and len(T(i + 2)[1]) == 1:
                letters, j = [s], i + 2                          # U.S.A / B.P. / A.B (vector dot?)
                while T(j)[0] == "lat" and len(T(j)[1]) == 1:
                    letters.append(T(j)[1])
                    if T(j + 1)[1] == "." and T(j + 2)[0] == "lat" and len(T(j + 2)[1]) == 1:
                        j += 2
                    else:
                        break
                glued = "".join(letters).lower()
                if len(letters) == 2:
                    alt([(glued, "en"), (" dot ".join(x.lower() for x in letters), "en")], ".".join(letters),
                        "abbrev_or_dot")
                else:
                    lit(glued)
                i = j + 1
                continue
            if n == 1 and s in ("x", "X") and prev_chunk and next_chunk and re.search(r"\d[^A-Za-z]*$", prev_chunk) \
                    and re.match(r"[-(]?\d", next_chunk):
                alt(ALT_TIMES, s, "times_x")                     # "2 x 3" -> two into three
            else:
                lit(s.lower())
            i += 1
            continue
        if ty == "dev":
            lit(s)
            i += 1
            continue
        if ty == "sup":
            body = s
            neg = False
            if body.startswith(SUP_MINUS):
                neg, body = True, body[1:]
            if body == chr(SUP_BASE + 2) and not neg:
                alt(ALT_SQ, _src(s), "superscript_square")
            elif body == chr(SUP_BASE + 3) and not neg:
                alt(ALT_CUBE, _src(s), "superscript_cube")
            else:
                if body == SUP_N:
                    exp = [("n", "en")]
                else:
                    ds = "".join(str(ord(c) - SUP_BASE) for c in body if SUP_BASE <= ord(c) <= SUP_BASE + 9)
                    exp = int_alts(ds) if ds else [("", "en")]
                pre = "minus " if neg else ""
                alts = [(f"{pw} {pre}{_primary(exp, pst)}", pst) for pw, pst in ALT_POW]
                alts += [(f"{pw} {pre}{e}", st) for pw, _ in ALT_POW[:1] for e, st in exp]
                alt(alts, _src(s), "superscript_power")
            i += 1
            continue
        if ty == "sub":
            ds = "".join(str(ord(c) - SUB_BASE) for c in s)
            if ds == "0":
                alt(ALT_SUB0, _src(s), "subscript")
            else:
                alt(int_alts(ds), _src(s), "subscript")
            i += 1
            continue
        if ty == "one":                                          # 1 वटा -> एक वटा / एउटा / one वटा
            cl = re.match("ओटा|वटा|टा", ns) if nty == "dev" else None
            if cl and ns[cl.end():] in [""] + CL_SUFFIXES:
                alt([(f"एक {ns}", "ne"), ("एउटा" + ns[cl.end():], "ne"), (f"one {ns}", "en")], "1 " + ns,
                    "one_classifier")
                i += 2
            else:
                alt(int_alts("1"), "1", "int_1-2digit")
                i += 1
            continue
        if ty == "prime":
            alt(ALT_PRIME, "'", "prime")
            i += 1
            continue
        if ty == "cur":
            base = number_alts(ns)[0] if nty == "num" else None
            if base is None:
                base = int_alts(ns.replace(",", "").split(".")[0]) if nty == "num" else None
            if base is None:
                alt([("rupees", "en"), ("रुपैयाँ", "ne")], "rs", "currency_rs")
                i += 1
                continue
            if T(i + 2)[1] == "-" and T(i + 3)[0] == "num" and number_alts(T(i + 3)[1])[0]:
                b = number_alts(T(i + 3)[1])[0]
                ane, aen, bne, ben = _primary(base, "ne"), _primary(base, "en"), _primary(b, "ne"), _primary(b, "en")
                alts = [(f"{ane} {bne} रुपैयाँ", "ne"), (f"रुपैयाँ {ane} {bne}", "ne"), (f"{ane} देखि {bne} रुपैयाँ", "ne"),
                        (f"rupees {aen} to {ben}", "en"), (f"{aen} to {ben} rupees", "en"), (f"rs {aen} {ben}", "en")]
                abne, abbne = _bare(base, "ne"), _bare(b, "ne")
                if (abne, abbne) != (ane, bne):
                    alts += [(f"{abne} {abbne} रुपैयाँ", "ne"), (f"{abne} देखि {abbne} रुपैयाँ", "ne")]
                alt(alts, f"rs {ns}-{T(i + 3)[1]}", "currency_rs")
                i += 4
                continue
            alts = []
            for t_, st in base:
                if st == "ne":
                    alts += [(f"{t_} रुपैयाँ", "ne"), (f"रुपैयाँ {t_}", "ne")]
                else:
                    alts += [(f"rupees {t_}", "en"), (f"{t_} rupees", "en"), (f"rs {t_}", "en")]
            alt(alts, "rs " + ns, "currency_rs")
            i += 2
            continue
        if ty == "inv":
            lit("inverse", _src(s), True, "inverse")
            i += 1
            continue
        # ---- single "other" character ----
        c = s
        if c == "+":
            lit("plus", c, True, "plus")
        elif c == "=":
            if ns == "=":
                alt(ALT_EQEQ, "==", "equals")
                i += 2
                continue
            mult = bool(prev2_chunk and re.fullmatch(r"[xX×*]", prev2_chunk)) or \
                bool(re.search(r"\d\s*[xX×*]\s*\d+$", chunk[:i])) or bool(prev_chunk and re.search(r"\d[xX×*]\d", prev_chunk))
            alt(ALT_EQ_MULT if mult else ALT_EQ, c, "equals")
        elif c == "^":
            alt(ALT_POW, c, "caret_power")
        elif c == "%":
            alt(ALT_PCT, c, "percent_standalone")
        elif c == "$":
            if nty == "num":
                base, _ = number_alts(ns)
                if base is None:
                    base = int_alts(ns.replace(",", ""))
                alts = []
                for t_, st in base:
                    if st == "en":
                        alts += [(f"{t_} dollars", "en"), (f"{t_} dollar", "en")]
                    else:
                        alts.append((f"{t_} dollar", "ne"))
                alt(alts, c + ns, "dollar")
                i += 2
                continue
            lit("dollar", c, True, "dollar")
        elif c == "#":
            if pty == "lat":
                lit("sharp", c, True, "hash_sharp")
            else:
                lit("hash", c, True, "hash_sharp")
        elif c == "@":
            alt(ALT_AT, c, "at_sign")
        elif c == "*":
            alt(ALT_STAR, c, "star")
        elif c in SYMBOL_WORDS:
            lit(SYMBOL_WORDS[c], c, True, "symbol_word")
        elif c == "-":
            if pty is None and nty is None:                       # standalone " - ": minus / pause / "dash"
                alt(ALT_MINUS_STANDALONE, c, "minus_standalone")
            elif nty in ("num", "lat") and (pty is None or (pty == "o" and ps in "^=(+*/[{<>")):
                alt(ALT_NEG, c, "negative")
            elif pty in ("lat", "sup", "sub", "num") and nty == "num" and pty != "num":
                if pty == "lat" and len(ps) > 3:
                    sep()
                else:
                    alt(ALT_MINUS_OR_GAP, c, "hyphen_minus")
            elif pty == "num" and nty == "lat":
                sep()
            elif pty in ("num", "lat", "sup", "sub") and nty is None:
                if pty == "lat" and len(ps) > 2:                 # "non- violent" style dangling hyphen
                    sep()
                else:
                    alt(ALT_MINUS_OR_GAP, c, "hyphen_minus")
            elif pty in ("sup", "sub", "num") and nty in ("lat", "num"):
                alt(ALT_MINUS_OR_GAP, c, "hyphen_minus")
            else:
                sep()
        elif c == "/":
            if pty == "num" or nty == "num" or pty in ("sup", "sub"):
                alt(ALT_DIV, c, "slash_divide")
            elif pty is None and nty is None:
                alt(ALT_DIV, c, "slash_divide")
            elif pty == "lat" and nty == "lat":
                if len(ps) <= 3 and len(ns) <= 3:
                    alt(ALT_SLASH_LAT, c, "slash_letters")
                else:
                    sep()
            elif (pty == "lat" and nty is None) or (pty is None and nty == "lat"):
                other = ps if pty == "lat" else ns
                alt(ALT_DIV if len(other) <= 2 else ALT_SLASH_LAT, c, "slash_divide")
            elif {pty, nty} == {"lat", "dev"}:
                alt(ALT_PER, c, "slash_letters")
            else:
                sep()
        elif c == ".":
            if pty == "lat" and nty == "lat":
                if len(ps) == 1 and len(ns) > 1 and ns[0].isupper():
                    items.append(_Item("glue"))                   # I.Sc / U.S.A / B.Com abbreviations
                elif ns[0].islower():
                    alt(ALT_DOT, c, "dot_code")
                else:
                    sep()
            elif pty == "num" and nty == "lat" and ns[0].islower():
                if T(i - 2)[0] == "lat":
                    alt(ALT_DOT, c, "dot_code")                  # lb2.settext
                else:
                    alt(ALT_POINT_OR_GAP, c, "dot_code")         # 0.something
            else:
                sep()
        elif c == "!":
            if pty == "num" or (pty == "lat" and len(ps) == 1):
                alt(ALT_FACT, c, "factorial")
            else:
                sep()
        elif c in "_~|\\":
            sep()
        else:
            cat = unicodedata.category(c)
            if cat.startswith("P") or cat.startswith("Z") or c.isspace():
                sep()
            elif cat.startswith("S"):
                if strict:
                    raise Unhandled(f"unhandled symbol {c!r} (U+{ord(c):04X})")
                lit(c, c, True, "unknown_symbol")
            else:
                lit(c)                                           # other-script letters: let encodability decide
        i += 1


def _parse(text: str, strict: bool = True):
    t = _prenorm(text)
    chunks = t.split(" ") if t else []
    items, pats = [], []
    for k, ch in enumerate(chunks):
        if not ch:
            continue
        items.append(_Item("sp"))
        _parse_chunk(ch, chunks[k - 1] if k else None, chunks[k + 1] if k + 1 < len(chunks) else None, items, pats,
                     strict, chunks[k - 2] if k >= 2 else None)
    return items, pats


def _render_clean(items) -> str:
    out = []
    for it in items:
        if it.kind == "sp":
            out.append(" ")
        elif it.kind == "glue":
            out.append("\x01")
        else:
            txt = it.src if it.verbal else it.text
            if it.verbal and txt.isascii():
                txt = txt.lower()
            out.append(txt if it.glued else " " + txt)
    return _finish("".join(out))


def _finish(s: str) -> str:
    s = re.sub(r"\s*\x01\s*", "", s)
    return " ".join(s.split())


def _segments(items):
    """Collapse items into segments: str (fixed text, may be empty) or list[(text, style)]."""
    segs, buf = [], []
    for it in items:
        if it.kind == "sp":
            buf.append(" ")
        elif it.kind == "glue":
            buf.append("\x01")
        elif it.kind == "lit":
            buf.append(" " + it.text)
        else:
            segs.append("".join(buf))
            buf = []
            segs.append(it.alts)
    segs.append("".join(buf))
    return segs


def _render(segs, choice) -> str:
    """choice: one index per alternative segment (in order)."""
    out, k = [], 0
    for s in segs:
        if isinstance(s, str):
            out.append(s)
        else:
            out.append(" " + s[choice[k]][0] + " ")
            k += 1
    return _finish("".join(out))


def _choices(segs, cap=MAX_CANDIDATES):
    """Candidate choice vectors: style-consistent first, full product if small, else consistent
    combinations at each rank plus single-span substitutions."""
    altsegs = [s for s in segs if not isinstance(s, str)]
    if not altsegs:
        return [()]

    def styled(style, rank=0):
        ch = []
        for a in altsegs:
            pool = [k for k, x in enumerate(a) if x[1] == style] or list(range(len(a)))
            ch.append(pool[min(rank, len(pool) - 1)])
        return tuple(ch)

    total = 1
    for a in altsegs:
        total *= len(a)
        if total > cap:
            break
    out = [styled("ne"), styled("en")]
    if total <= cap:
        out += list(itertools.product(*[range(len(a)) for a in altsegs]))
    else:
        maxrank = max(len(a) for a in altsegs)
        for rank in range(1, maxrank):
            out += [styled("ne", rank), styled("en", rank)]
        for st in ("ne", "en"):
            base = styled(st)
            for k, a in enumerate(altsegs):
                for x in range(len(a)):
                    ch = list(base)
                    ch[k] = x
                    out.append(tuple(ch))
    return out


def _expand(text: str):
    """-> (segs, verbal, reason)."""
    try:
        items, _ = _parse(text)
    except Unhandled as e:
        return None, True, str(e)
    verbal = any(it.verbal for it in items if it.kind in ("lit", "alt"))
    return (_segments(items) if verbal else [_render_clean(items)]), verbal, None


def clean(text: str) -> str:
    """Full normalisation except that numeric/maths spans are left verbatim (digits stay)."""
    items, _ = _parse(text, strict=False)
    return _render_clean(items)


def has_numeric(text: str) -> bool:
    """True if the text contains any span that must be verbalised (digits, %, ², +, =, ^, /, $ ...)."""
    items, _ = _parse(text, strict=False)
    return any(it.verbal for it in items if it.kind in ("lit", "alt"))


def numeric_patterns(text: str) -> list:
    """Pattern labels of all verbalised spans (for reporting)."""
    try:
        _, pats = _parse(text)
    except Unhandled as e:
        return ["UNHANDLED:" + str(e)]
    return pats


def numeric_spans(text: str) -> list:
    """[(pattern, source_span, [alternatives], chunk)] for every verbalised span (reporting/debugging)."""
    try:
        items, _ = _parse(text)
    except Unhandled as e:
        return [("UNHANDLED", str(e), [], "")]
    out = []
    for it in items:
        if it.kind == "alt":
            out.append((it.pat, _src(it.src), [a for a, _ in it.alts], _src(it.chunk or "")))
        elif it.kind == "lit" and it.verbal:
            out.append((it.pat, _src(it.src), [it.text], _src(it.chunk or "")))
    return out


def segments_with_reason(text: str):
    """Return (segments, reason). Each segment is a fixed str or a list of alternative strings;
    joining one choice per segment with single spaces (then collapsing whitespace) gives a
    candidate. Lets a selector compose per-span choices beyond the capped candidate list."""
    segs, verbal, reason = _expand(text)
    if reason:
        return None, reason
    out = []
    for s in segs:
        if isinstance(s, str):
            f = _finish(s)
            if f:
                out.append(f)
        else:
            out.append([x[0] for x in s])
    return out, None


_FINAL_OK = re.compile(r"[a-z\u0900-\u0963\u0971-\u097F ]+")   # lowercase Latin, Devanagari (no digits/danda/॰), space


def is_final(text: str) -> bool:
    """Full target-convention check: allowed characters only, normalised spacing, encodable."""
    return (bool(text) and _FINAL_OK.fullmatch(text) is not None and text == " ".join(text.split())
            and unicodedata.normalize("NFKC", text) == text and is_encodable(text))


def _candidates_and_choices(text: str, cap=MAX_CANDIDATES):
    segs, verbal, reason = _expand(text)
    if reason:
        return [], [], segs, reason
    if not verbal:
        c = segs[0]
        if not c:
            return [], [], segs, "empty after cleaning"
        if not is_final(c):
            return [], [], segs, "unencodable after cleaning: " + _bad_chars(c)
        return [c], [()], segs, None
    seen, cands, chs = set(), [], []
    first_bad = None
    for ch in _choices(segs, cap):
        r = _render(segs, ch)
        if r in seen:
            continue
        seen.add(r)
        if is_final(r):
            cands.append(r)
            chs.append(ch)
            if len(cands) >= cap:
                break
        elif first_bad is None:
            first_bad = r
    if not cands:
        return [], [], segs, ("unencodable candidates: " + _bad_chars(first_bad)) if first_bad else "empty"
    return cands, chs, segs, None


def candidates_with_reason(text: str):
    """Return (candidates, reason). reason is None when handled; candidates is [] when unhandled."""
    cands, _, _, reason = _candidates_and_choices(text)
    return cands, reason


def candidates(text: str) -> list:
    c, reason = candidates_with_reason(text)
    if reason:
        raise Unhandled(reason)
    return c


def _bad_chars(t: str) -> str:
    sp = get_tokenizer()
    bad = sorted({ch for ch in set(t) if ch != " " and (sp.unk_id() in sp.encode(ch) or not _FINAL_OK.fullmatch(ch))})
    if re.search(r"\d", t):
        bad.append("<digit>")
    return repr("".join(bad)) if bad else repr(t[:60])


# ---------------------------------------------------------------------------------------------
# Romanisation (coarse phonetic key) for comparing hypotheses with candidates
# ---------------------------------------------------------------------------------------------
_R_CONS = {
    "क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n", "च": "c", "छ": "c", "ज": "j", "झ": "j", "ञ": "n",
    "ट": "t", "ठ": "t", "ड": "d", "ढ": "d", "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n",
    "ऩ": "n", "प": "p", "फ": "p", "ब": "b", "भ": "b", "म": "m", "य": "i", "र": "r", "ऱ": "r", "ल": "l",
    "ळ": "l", "ऴ": "l", "व": "b", "श": "s", "ष": "s", "स": "s", "ह": "h",
    "क़": "k", "ख़": "k", "ग़": "g", "ज़": "j", "ड़": "r", "ढ़": "r", "फ़": "p", "य़": "i",
}
_R_VOW = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ॠ": "ri", "ऌ": "li",
          "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऒ": "o", "ऍ": "e", "ऎ": "e", "ॲ": "a"}
_R_MATRA = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "ॄ": "ri", "े": "e", "ै": "ai",
            "ो": "o", "ौ": "au", "ॉ": "o", "ॊ": "o", "ॅ": "e", "ॆ": "e", "ॢ": "li"}
_VIRAMA, _NUKTA = "्", "़"


def _deva_word(w: str) -> str:
    w = unicodedata.normalize("NFC", w)
    w = w.replace(_NUKTA, "")
    out = []
    L = len(w)
    for k, ch in enumerate(w):
        if ch in _R_CONS:
            out.append(_R_CONS[ch])
            nxt = w[k + 1] if k + 1 < L else ""
            if nxt and nxt not in _R_MATRA and nxt != _VIRAMA and nxt not in "ंँः":
                out.append("a")
            elif nxt and nxt in "ंँ":
                out.append("a")
        elif ch in _R_MATRA:
            out.append(_R_MATRA[ch])
        elif ch in _R_VOW:
            out.append(_R_VOW[ch])
        elif ch in "ंँ":
            out.append("n")
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


# Pronunciation respellings (as Nepali speakers / SraVaani's Devanagari would render them) for the
# English words that candidates generate, so "two" ~ टू, "eight" ~ एट, "square" ~ स्क्वायर.
_EN_PRON = {
    "zero": "jiro", "one": "wan", "two": "tu", "three": "thri", "four": "phor", "five": "phaib", "six": "siks",
    "seven": "seben", "eight": "et", "nine": "nain", "ten": "ten", "eleven": "ileben", "twelve": "twelb",
    "thirteen": "thartin", "fourteen": "phortin", "fifteen": "phiphtin", "sixteen": "sikstin",
    "seventeen": "sebentin", "eighteen": "etin", "nineteen": "naintin", "twenty": "twenti", "thirty": "tharti",
    "forty": "phorti", "fifty": "phiphti", "sixty": "siksti", "seventy": "sebenti", "eighty": "eti",
    "ninety": "nainti", "hundred": "handred", "thousand": "thaujend", "million": "milian", "billion": "bilian",
    "lakh": "lakh", "crore": "karor", "first": "pharst", "second": "sekend", "third": "thard", "fourth": "phorth",
    "fifth": "phiphth", "eighth": "etth", "ninth": "nainth", "twelfth": "twelph", "point": "point",
    "percent": "parsent", "plus": "plas", "minus": "mainas", "into": "intu", "by": "bai", "equal": "ikwal",
    "equals": "ikwals", "is": "ij", "to": "tu", "the": "da", "power": "pawar", "square": "skwayar",
    "cube": "kyub", "root": "rut", "half": "haph", "dot": "dat", "not": "nat", "naught": "nat", "over": "obar",
    "upon": "apan", "times": "taims", "divided": "dibaided", "dollar": "dalar", "dollars": "dalars", "and": "end",
    "at": "et", "rate": "ret", "oh": "o", "per": "par", "slash": "slyas", "inverse": "inbars", "raise": "rej",
    "clock": "klak", "negative": "negetib", "factorial": "phyaktorial", "hash": "hyas", "sharp": "sarp",
}


def _latin_word(w: str) -> str:
    w = w.lower()
    w = _EN_PRON.get(w, w)
    for a, b in (("ch", "C"), ("sh", "s"), ("th", "t"), ("ph", "p"), ("kh", "k"), ("gh", "g"), ("dh", "d"),
                 ("bh", "b"), ("jh", "j"), ("ck", "k"), ("qu", "kb"), ("x", "ks"), ("ee", "i"), ("oo", "u"),
                 ("ce", "se"), ("ci", "si"), ("cy", "si"), ("c", "k"), ("C", "c"), ("f", "p"), ("v", "b"), ("w", "b"), ("z", "j"), ("q", "k"), ("y", "i")):
        w = w.replace(a, b)
    return w


def romanize(text: str) -> str:
    """Coarse, forgiving phonetic key: Devanagari transliterated to Latin, aspiration / vowel length /
    sibilants / v-w-b collapsed, virama dropped; Latin lowercased and folded the same way. Other Brahmic
    scripts (Bengali, Gujarati, Oriya ... which SraVaani sometimes emits) are first folded onto Devanagari
    by Unicode block offset."""
    words = []
    text = unicodedata.normalize("NFKC", text)
    text = "".join(chr(0x0900 + (ord(c) - 0x0900) % 0x80) if "\u0980" <= c <= "\u0D7F" else c for c in text)
    for tok in re.findall(rf"[{_DEV}]+|[A-Za-z]+|\d+", text):
        if tok[0].isascii():
            r = _latin_word(tok) if tok[0].isalpha() else tok
        else:
            r = _deva_word(tok)
        r = re.sub(r"(.)\1+", r"\1", r)
        if r:
            words.append(r)
    return " ".join(words)


def _lev(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b)
    except ImportError:  # pragma: no cover
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def _hyp_range(ops, s, e, ext):
    """Hyp index range aligned to candidate chars [s, e) given Levenshtein opcodes; with ext, hyp chars
    inserted right at the span boundaries are included too. None if nothing aligns."""
    hs = he = None
    for tag, i1, i2, j1, j2 in ops:
        if tag == "insert":
            if s < i1 < e or (ext and s <= i1 <= e):
                hs = j1 if hs is None else min(hs, j1)
                he = j2 if he is None else max(he, j2)
            continue
        lo, hi = max(i1, s), min(i2, e)
        if lo >= hi:
            continue
        if tag == "delete":
            a = b = j1
        else:
            a, b = j1 + (lo - i1), j1 + (hi - i1)
        hs = a if hs is None else min(hs, a)
        he = b if he is None else max(he, b)
    return None if hs is None else (hs, he)


def _local_refine(segs, choice, h, trace=None):
    """One pass over the alternative spans, scoring each alternative *locally*: the romanised string
    "previous word + alternative + next word" against the hypothesis window aligned to that stretch
    (window ends may slide by up to 3 characters). Unlike the whole-sentence distance, noise elsewhere
    in the hypothesis cannot make a short reading ("छ") beat a matching longer one ("six").
    An empty alternative is ruled out when a long (>= 6 chars) non-empty alternative is clearly
    present in the window. The current choice is replaced only if the local distance improves by at
    least max(2, 25% of the local string)."""
    from rapidfuzz import fuzz as _fz
    from rapidfuzz.distance import Levenshtein as _L
    rom_seg = [romanize(_finish(x)) if isinstance(x, str) else [romanize(t) for t, _ in x] for x in segs]
    alt_pos = [k for k, x in enumerate(segs) if not isinstance(x, str)]
    choice = list(choice)
    M = 3

    def build(ch):
        parts, offs, pos, j = [], [], 0, 0
        for x in rom_seg:
            if isinstance(x, list):
                p = x[ch[j]]
                j += 1
            else:
                p = x
            if p:
                if parts:
                    pos += 1
                offs.append((pos, pos + len(p)))
                parts.append(p)
                pos += len(p)
            else:
                offs.append((pos, pos + 1) if parts else (0, 0))     # gap = the joining space
        return " ".join(parts), offs

    changed = False
    for j, k in enumerate(alt_pos):
        alts = rom_seg[k]
        if len(alts) < 2:
            continue
        full, offs = build(choice)
        s0, e0 = offs[k]
        ps = full.rfind(" ", 0, max(0, s0 - 1)) + 1                 # previous word start
        prev_w = full[ps:max(ps, s0 - 1)] if s0 > 0 else ""
        ne_ = full.find(" ", min(len(full), e0 + 1))
        ne_ = len(full) if ne_ < 0 else ne_
        next_w = full[e0 + 1:ne_] if e0 < len(full) else ""
        rg = _hyp_range(_L.opcodes(full, h), ps, ne_, True)
        if rg is None:
            continue
        W = h[max(0, rg[0] - M): rg[1] + M]
        Wn = W.replace(" ", "")
        lead = len(h[max(0, rg[0] - M):rg[0]].replace(" ", ""))
        trail = len(h[rg[1]:rg[1] + M].replace(" ", ""))
        dists, present = [], False
        for x in range(len(alts)):
            q = (prev_w + alts[x] + next_w).replace(" ", "")
            best = None
            for l in range(0, 2 * lead + 1 if lead else 1):
                for r in range(0, 2 * trail + 1 if trail else 1):
                    w = Wn[l:len(Wn) - r] if r else Wn[l:]
                    d = _L.distance(q, w)
                    best = d if best is None else min(best, d)
            dists.append(best)
            a = alts[x].replace(" ", "")
            if len(a) >= 6 and len(a) <= len(Wn) and _fz.partial_ratio(a, Wn) >= 70:
                present = True
        if present:
            dists = [10 ** 6 if not alts[x].strip() else d for x, d in enumerate(dists)]
        cur = choice[j]
        bx = min(range(len(alts)), key=lambda x: (dists[x], x != cur, x))
        need = max(2, 0.25 * len((prev_w + alts[bx] + next_w).replace(" ", "")))
        if trace is not None:
            trace.append((j, dists, cur, bx))
        if bx != cur and dists[cur] - dists[bx] >= need:
            choice[j] = bx
            changed = True
    return choice, changed


def best_candidate(text: str, hyp: str, max_rounds: int = 3, local: bool = False, return_info: bool = False):
    """Choose the verbalisation of `text` that best matches ASR hypothesis `hyp`:
      1. best of the candidate list by romanised whole-sentence edit distance;
      2. per-span coordinate descent on that distance (reaches mixed Nepali/English readings);
      3. (local=True, EXPERIMENTAL, off by default: not yet validated) per-span local re-scoring
         against the aligned hypothesis window (see _local_refine).
    Returns (best_text, distance) where distance is the whole-sentence romanised distance of the
    returned text, or (None, reason) if the text is unhandled. With return_info=True a dict is
    returned instead: text, dist, list_text, list_dist, list_second_dist, global_text, global_dist,
    local_changed."""
    cands, chs, segs, reason = _candidates_and_choices(text)
    if reason:
        return (None, reason) if not return_info else {"text": None, "reason": reason}
    h = romanize(hyp)
    scored = sorted(((_lev(romanize(c), h), k) for k, c in enumerate(cands)))
    best_d, bk = scored[0]
    list_text, list_d = cands[bk], best_d
    second_d = scored[1][0] if len(scored) > 1 else None
    best, choice = cands[bk], list(chs[bk])
    altsegs = [s for s in segs if not isinstance(s, str)] if choice else []
    cur_d = best_d
    for _ in range(max_rounds if altsegs else 0):
        improved = False
        for k, a in enumerate(altsegs):
            for x in range(len(a)):
                if x == choice[k]:
                    continue
                trial = list(choice)
                trial[k] = x
                r = _render(segs, trial)
                d = _lev(romanize(r), h)
                if d < cur_d and is_final(r):
                    choice, cur_d, best, improved = trial, d, r, True
        if not improved:
            break
    global_text, global_d = best, cur_d
    local_changed = False
    if local and altsegs:
        ch2, changed = _local_refine(segs, choice, h)
        if changed:
            r = _render(segs, ch2)
            if is_final(r) and r != best:
                best, cur_d, local_changed = r, _lev(romanize(r), h), True
    if return_info:
        return {"text": best, "dist": cur_d, "list_text": list_text, "list_dist": list_d,
                "list_second_dist": second_d, "global_text": global_text, "global_dist": global_d,
                "local_changed": local_changed}
    return best, cur_d
