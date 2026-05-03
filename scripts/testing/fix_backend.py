"""
Comprehensive backend fix:
1. Sanitizes all bad Unicode from main.py
2. Sets correct thresholds (0.60 balanced)
3. Ensures utf-8 encoding everywhere
"""
import json
import re

# ── 1. Read + sanitize main.py ─────────────────────────────────────────────
with open('backend/main.py', 'r', encoding='utf-8') as f:
    text = f.read()

# All known corrupted sequences
bad = [
    ('\ufeff',       ''),
    ('\x9d',         ''),
    ('\x00',         ''),
    ('â[ERROR]Œ',   '[ERROR]'),
    ('â[OK]',        '[OK]'),
    ('âœ…',           '[OK]'),
    ('â€"',          ' - '),
    ('â€™',          "'"),
    ('\u2014',       ' - '),
    ('\u2013',       '-'),
    ('\u2019',       "'"),
    ('\u201c',       '"'),
    ('\u201d',       '"'),
    ('❌',           '[ERROR]'),
    ('✅',           '[OK]'),
    ('⚠️',           '[WARN]'),
]

for old, new in bad:
    if old in text:
        count = text.count(old)
        print(f"  Fixed {count}x: {repr(old)} -> {repr(new)}")
        text = text.replace(old, new)

# verify no remaining non-ASCII that would break cp1252
bad_lines = []
for i, line in enumerate(text.splitlines(), 1):
    try:
        line.encode('cp1252')
    except UnicodeEncodeError as e:
        bad_lines.append((i, str(e), line.strip()[:80]))

if bad_lines:
    print(f"\n  Still {len(bad_lines)} bad lines:")
    for lineno, err, content in bad_lines:
        print(f"    Line {lineno}: {err}")
        print(f"    >>> {content}")
else:
    print("  No remaining cp1252 issues!")

with open('backend/main.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("\nmain.py written successfully\n")

# ── 2. Set balanced thresholds ─────────────────────────────────────────────
thresholds = {
    "thresholds": {
        "dr":           0.60,
        "glaucoma":     0.60,
        "amd":          0.60,
        "cataract":     0.60,
        "hypertensive": 0.60,
        "myopic":       0.60
    }
}
with open('backend/models/threshold_config.json', 'w', encoding='utf-8') as f:
    json.dump(thresholds, f, indent=2)
print(f"threshold_config.json -> {thresholds['thresholds']}")
print("\nAll fixes applied. Ready to start.")
