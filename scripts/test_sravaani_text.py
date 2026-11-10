"""Tests for sravaani_text (plain asserts; run: .venv/bin/python scripts/test_sravaani_text.py)."""
import os
import random
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sravaani_text as T  # noqa: E402

# Independently re-typed reference for Nepali 0-99 (by decade) to catch typos in the module table.
REF_DECADES = {
    0: "शून्य एक दुई तीन चार पाँच छ सात आठ नौ",
    10: "दस एघार बाह्र तेह्र चौध पन्ध्र सोह्र सत्र अठार उन्नाइस",
    20: "बीस एक्काइस बाइस तेइस चौबीस पच्चीस छब्बीस सत्ताइस अट्ठाइस उनन्तीस",
    30: "तीस एकतीस बत्तीस तेत्तीस चौँतीस पैँतीस छत्तीस सैँतीस अठतीस उनन्चालीस",
    40: "चालीस एकचालीस बयालीस त्रिचालीस चवालीस पैँतालीस छयालीस सतचालीस अठचालीस उनन्चास",
    50: "पचास एकाउन्न बाउन्न त्रिपन्न चउन्न पचपन्न छपन्न सन्ताउन्न अन्ठाउन्न उनन्साठी",
    60: "साठी एकसट्ठी बयसट्ठी त्रिसट्ठी चौसठ्ठी पैँसट्ठी छयसट्ठी सतसट्ठी अठसट्ठी उनन्सत्तरी",
    70: "सत्तरी एकहत्तर बहत्तर त्रिहत्तर चौहत्तर पचहत्तर छयहत्तर सतहत्तर अठहत्तर उनासी",
    80: "असी एकासी बयासी त्रियासी चौरासी पचासी छयासी सतासी अठासी उनान्नब्बे",
    90: "नब्बे एकानब्बे बयानब्बे त्रियानब्बे चौरानब्बे पन्चानब्बे छयानब्बे सन्तानब्बे अन्ठानब्बे उनान्सय",
}

PASSED = []


def check(name):
    def deco(fn):
        fn()
        PASSED.append(name)
        return fn
    return deco


def has(text, expected):
    c = T.candidates(text)
    assert expected in c, f"{expected!r} not in candidates of {text!r}: {c}"
    return c


@check("nepali table 0-99 vs independent reference")
def _():
    for d, words in REF_DECADES.items():
        ws = words.split()
        assert len(ws) == 10, d
        for k, w in enumerate(ws):
            assert T.ne_int(d + k) == w, (d + k, T.ne_int(d + k), w)
    assert len(set(T.NE_0_99)) == 100
    for w in T.NE_0_99:
        assert unicodedata.normalize("NFKC", w) == w and re.fullmatch(r"[ऀ-ॿ]+", w), w
        assert T.is_encodable(w), w


@check("nepali cardinal spot checks")
def _():
    spot = {
        0: "शून्य", 10: "दस", 11: "एघार", 19: "उन्नाइस", 21: "एक्काइस", 29: "उनन्तीस", 39: "उनन्चालीस",
        49: "उनन्चास", 59: "उनन्साठी", 69: "उनन्सत्तरी", 79: "उनासी", 89: "उनान्नब्बे", 99: "उनान्सय",
        100: "एक सय", 101: "एक सय एक", 250: "दुई सय पचास", 1000: "एक हजार", 1001: "एक हजार एक",
        1999: "एक हजार नौ सय उनान्सय", 2025: "दुई हजार पच्चीस", 100000: "एक लाख",
        1500000: "पन्ध्र लाख", 10000000: "एक करोड", 12345678: "एक करोड तेइस लाख पैँतालीस हजार छ सय अठहत्तर",
        1000000000: "एक अर्ब",
    }
    for n, w in spot.items():
        assert T.ne_int(n) == w, (n, T.ne_int(n), w)
    assert T.ne_hundreds(1999) == "उन्नाइस सय उनान्सय"
    assert T.ne_hundreds(1500) == "पन्ध्र सय"
    assert T.ne_hundreds(2025) is None and T.ne_hundreds(999) is None


@check("english cardinal / year / ordinal")
def _():
    assert T.en_int(0) == "zero"
    assert T.en_int(21) == "twenty one"
    assert T.en_int(101) == "one hundred one"
    assert T.en_int(101, keep_and=True) == "one hundred and one"
    assert T.en_int(2025) == "two thousand twenty five"
    assert T.en_int(1500000) == "one million five hundred thousand"
    assert T.en_year(2025) == "twenty twenty five"
    assert T.en_year(1999) == "nineteen ninety nine"
    assert T.en_lakh(1500000) == "fifteen lakh"
    assert T.en_ordinal(21) == "twenty first"


@check("number candidates for required values")
def _():
    for n, ne, en in [(0, "शून्य", "zero"), (10, "दस", "ten"), (11, "एघार", "eleven"), (19, "उन्नाइस", "nineteen"),
                      (21, "एक्काइस", "twenty one"), (99, "उनान्सय", "ninety nine"), (100, "एक सय", "one hundred"),
                      (101, "एक सय एक", "one hundred one"), (1000, "एक हजार", "one thousand"),
                      (1999, "उन्नाइस सय उनान्सय", "nineteen ninety nine"),
                      (2025, "दुई हजार पच्चीस", "twenty twenty five"), (100000, "एक लाख", "one lakh"),
                      (1500000, "पन्ध्र लाख", "one million five hundred thousand"),
                      (10000000, "एक करोड", "ten million")]:
        c = has(f"{n} वटा", f"{ne} वटा")
        assert f"{en} वटा" in c, (n, c)
    c = has("1999 मा", "एक हजार नौ सय उनान्सय मा")
    assert "one thousand nine hundred ninety nine मा" in c
    c = has("2025 मा", "two thousand twenty five मा")


@check("decimals")
def _():
    c = has("2.5 kg", "दुई दशमलव पाँच kg")
    assert "two point five kg" in c and "अढाई kg" in c
    has("1.5 घण्टा", "डेढ घण्टा")
    has("0.25", "zero point two five")
    has("7.62", "सात दशमलव छ दुई")
    has("3.5", "साढे तीन")


@check("percent")
def _():
    c = has("10% छ", "दस प्रतिशत छ")
    assert "ten percent छ" in c and "दस percent छ" in c
    has("99.9%", "ninety nine point nine percent")


@check("ordinals")
def _():
    c = has("1st र 2nd", "पहिलो र दोस्रो")
    assert "first र second" in c
    has("21st century", "twenty first century")
    has("11th", "एघारौँ")
    has("3rd", "third")


@check("ranges, times, fractions, thousands")
def _():
    c = has("10-15 जना", "दस पन्ध्र जना")
    assert "ten to fifteen जना" in c and "दस देखि पन्ध्र जना" in c
    has("6:30 बजे", "साढे छ बजे")
    has("6:30 बजे", "six thirty बजे")
    has("1/2", "one by two")
    has("1/2", "आधा")
    has("1,000 रुपैयाँ", "एक हजार रुपैयाँ")
    has("1,00,000", "one lakh")
    has("1,00,000", "एक लाख")
    has("1940s मा", "nineteen forties मा")


@check("glued devanagari suffixes and devanagari digits")
def _():
    has("15जना", "पन्ध्र जना")
    has("2025मा", "twenty twenty five मा")
    c = has("१०औँ", "दसौँ")
    assert "tenth" in c
    has("21औं शताब्दी", "एक्काइसौँ शताब्दी")
    has("16 औँ शताब्दीमा", "सोह्रौँ शताब्दीमा")
    has("19 औं", "nineteenth")
    has("१५ जना", "fifteen जना")


@check("superscripts, subscripts, powers, operators")
def _():
    c = has("x²", "x square")
    assert "x को square" in c
    has("n³", "n cube")
    has("m² area", "m square area")
    has("H₂O", "h two o")
    has("x₀", "x not")
    has("2^3", "two to the power three")
    has("a + b = c", "a plus b equals to c")
    has("a + b = c", "a plus b बराबर c")
    has("n - 1", "n minus one")
    has("-5", "minus five")
    has("cos-¹ x", "cos inverse x")
    has("C++ र C#", "c plus plus र c sharp")
    has("Q&A", "q and a")
    has("$5", "five dollars")
    has("Rs 2,00,000 खर्च", "दुई लाख रुपैयाँ खर्च")
    has("Rs. 2500", "rupees two thousand five hundred")
    has("O(n2)", "o n two")
    has("4 x 3 = 12", "four into three equals to twelve")
    has("2 x 0", "दुई x शून्य")
    assert T.candidates("A.B") == ["ab", "a dot b"]
    assert T.candidates("B.Com पढें") == ["bcom पढें"]
    has("system.out.println", "system dot out dot println")
    has("046 सालको", "छयालीस सालको")
    has("10 10", "one zero one zero")


@check("alphanumerics")
def _():
    has("5G network", "five g network")
    has("4K video", "four k video")
    has("MP3", "mp three")
    has("COVID-19", "covid nineteen")
    has("IPv4", "ipv four")
    has("3D", "three d")
    has("i7 processor", "i seven processor")
    has("1080p", "ten eighty p")
    has("S3 bucket", "s three bucket")
    has("9841234567", "nine eight four one two three four five six seven")


@check("punctuation, apostrophes, hyphens, lowercase")
def _():
    assert T.clean("नमस्ते, तपाईंलाई कस्तो छ? I'm fine!") == "नमस्ते तपाईंलाई कस्तो छ im fine"
    assert T.clean("Don't worry, it's the teacher's day.") == "dont worry its the teachers day"
    assert T.clean("well-known राम-श्याम") == "well known राम श्याम"
    assert T.clean("AWS Lambda “quoted” (paren) [x]; a: b") == "aws lambda quoted paren x a b"
    assert T.clean("  धेरै   space  ") == "धेरै space"
    assert T.clean("ＡＢＣ") == "abc"                       # NFKC
    assert T.clean("I.Sc. पढ्दा") == "isc पढ्दा"
    assert T.clean("कारण/आधार") == "कारण आधार"
    assert T.clean("फारसी ‎जसलाई") == "फारसी जसलाई"
    assert T.candidates("Hello World!") == ["hello world"]
    assert not T.has_numeric("Hello, World!")
    assert T.has_numeric("10 वटा") and T.has_numeric("x²") and T.has_numeric("a = b")
    assert T.clean("10 वटा, 2.5%") == "10 वटा 2.5%"         # numbers left verbatim by clean()


@check("hard requirement: encodable, no digits, non-empty, capped")
def _():
    texts = ["भनेपछि हामी 15 जना छौँ. don't worry!", "x² + 2x + 1 = 0", "2^10 = 1024", "1,00,000 र 10,000",
             "I0, I1, I2, I3, I4, I5, I6, I7 र Q0 Q1 Q2 Q3 S0 S1 S2 S3", "Error 404 page", "1.5x speed",
             "7000-8000 वर्ष", "45:55 ratio", "PL/16 र d/dx", "9 - 9! गर्दै", "system.out.println"]
    sp = T.get_tokenizer()
    for t in texts:
        c, reason = T.candidates_with_reason(t)
        assert reason is None and c, (t, reason)
        assert len(c) <= T.MAX_CANDIDATES
        assert len(set(c)) == len(c)
        for x in c:
            assert x and not re.search(r"\d", x) and x == " ".join(x.split()), x
            ids = sp.encode(x)
            assert sp.unk_id() not in ids and sp.decode(ids) == x, x
            assert x == x.strip() and x == unicodedata.normalize("NFKC", x)
            assert not re.search(r"[A-Z]", x), x
    rnd = random.Random(0)
    for n in list(range(0, 1001)) + [rnd.randrange(10 ** k) for k in range(2, 13) for _ in range(30)]:
        for a, _ in T.int_alts(str(n)):
            assert T.is_encodable(a), (n, a)


@check("audit fixes (bare 100/1000, एउटा, pair readings, empty =/x, dash, decimals, currency guard, prime)")
def _():
    # 1 bare hundred / thousand (full forms kept)
    c = has("100% मुख्य", "hundred percent मुख्य")
    assert "one hundred percent मुख्य" in c and "सय प्रतिशत मुख्य" in c and "एक सय प्रतिशत मुख्य" in c
    has("100 वर्ष", "सय वर्ष")
    has("1000 छ", "हजार छ")
    has("1000 छ", "thousand छ")
    has("120 degree", "hundred twenty degree")
    has("Rs 100-200 मा", "सय दुई सय रुपैयाँ मा")
    # 2 एउटा for 1 + classifier (absorbs it), not for measure words
    c = has("जस्तै 1 वटा होइन", "जस्तै एउटा होइन")
    assert "जस्तै एक वटा होइन" in c
    has("1 वटामा", "एउटामा")
    has("1 टा मात्रै", "एउटा मात्रै")
    assert not any("एउटा" in x for x in T.candidates("1 जना"))
    assert not any("एउटा" in x for x in T.candidates("1 वर्ष"))
    assert not any("एउटा" in x for x in T.candidates("21 वटा"))
    assert not any("उको" in x and "एउटा" in x for x in T.candidates("1 टाउको"))
    has("1 पसलले", "एउटा पसलले")
    # 3 three-digit pair readings
    has("130 days", "one thirty days")
    has("180", "one eighty")
    has("365 दिन", "three sixty five दिन")
    assert "one zero five" in T.candidates("105") and not any(x == "one five" for x in T.candidates("105"))
    # 1010 / 2022 Nepali pair years
    has("ईपू 1010", "ईपू दस दस")
    has("2022 मा", "बीस बाइस मा")
    # 4 standalone dash
    c = has("9 - को factorial", "nine को factorial")
    assert "nine minus को factorial" in c and "nine dash को factorial" in c
    # 5 '=' and multiplication 'x' can be silent; जा only in multiplication facts
    has("2 + 3 = 5", "दुई plus तीन पाँच")
    c = has("2 x 3 = 6", "two three six")
    assert "दुई तीन जा छ" in c
    assert not any(" जा " in x for x in T.candidates("y = x + 1"))
    # 6 decimals without point / fraction as a number
    has("1.37 प्रतिशत", "एक सैँतीस प्रतिशत")
    has("1.37 प्रतिशत", "one thirty seven प्रतिशत")
    has("19.03 रहेको", "उन्नाइस शून्य तीन रहेको")
    has("2.0 नाम", "दुई शून्य नाम")
    # 7 currency regex needs a word start
    assert T.candidates("भन्छन् यिनीहरु. 47 सालमा")[0] == "भन्छन् यिनीहरु सतचालीस सालमा"
    has("रु. 500", "पाँच सय रुपैयाँ")
    # 8 prime notation
    c = has("YC + YC'", "yc plus yc dash")
    assert "yc plus yc" in c
    has("W'Z' को लागि", "w dash z dash को लागि")
    has("x'z", "x dash z")
    assert T.candidates("don't I'm it's students' marks") == ["dont im its students marks"]


@check("unhandled reporting")
def _():
    c, reason = T.candidates_with_reason("copyright © 2020")
    assert c == [] and reason
    c, reason = T.candidates_with_reason("...")
    assert c == [] and reason
    try:
        T.candidates("©")
        raise AssertionError("expected Unhandled")
    except T.Unhandled:
        pass


@check("segments")
def _():
    segs, reason = T.segments_with_reason("2 वटा x²")
    assert reason is None
    assert segs == [["दुई", "two"], "वटा x", ["square", "को square"]], segs


@check("romanize and best_candidate")
def _():
    assert T.romanize("ट्वेन्टी") == T.romanize("twenty")
    assert T.romanize("थ्री") == T.romanize("three")
    assert T._lev(T.romanize("फाइभ"), T.romanize("five")) <= 2
    assert T.romanize("ABC") == T.romanize("abc")
    best, d = T.best_candidate("2025 मा 15 जना", "ट्वेन्टी ट्वेन्टी फाइभ मा पन्ध्र जना")
    assert best == "twenty twenty five मा पन्ध्र जना", best
    best, d = T.best_candidate("2025 मा 15 जना", "दुई हजार पच्चीस मा फिफ्टिन जना")
    assert best == "दुई हजार पच्चीस मा fifteen जना", best


if __name__ == "__main__":
    print(f"{len(PASSED)} test groups passed:")
    for p in PASSED:
        print("  ok -", p)
