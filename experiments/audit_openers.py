import re
from collections import Counter
import numpy as np

with open('springer_project/TabStress.tex', 'r') as f:
    text = f.read()

# 1. Total lines, characters, words in raw latex
total_raw_lines = len(text.splitlines())
total_raw_words = len(text.split())

# Strip comments
lines = [l for l in text.splitlines() if not l.strip().startswith('%')]
clean_text = '\n'.join(lines)

# Split body
body = clean_text.split(r'\begin{document}')[1].split(r'\end{document}')[0]

# Extract sections
raw_sections = re.findall(r'\\section\*?\{([^}]+)\}', body)
print(f"=== SECTIONS ({len(raw_sections)}) ===")
for idx, s in enumerate(raw_sections, 1):
    print(f"  {idx}. {s}")

# Environments to eliminate:
env_list = ['tabular', 'table', 'table*', 'subfigure', 'figure', 'equation', 
            'smallmatrix', 'longtable', 'tikzpicture', 'algorithm', 'algorithmic', 
            'gather', 'tabular*', 'proof', 'proposition', 'definition', 'remark']

for env in env_list:
    escaped_env = env.replace('*', r'\*')
    pattern = rf'\\begin\{{{escaped_env}\}}[\s\S]*?\\end\{{{escaped_env}\}}'
    body = re.sub(pattern, ' ', body)

body = re.sub(r'\\\[[\s\S]*?\\\]', ' ', body)
body = re.sub(r'\$[^\$]+\$', ' ', body)
body = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^\}]*\})*', ' ', body)
body = re.sub(r'[{}\[\]\\]', ' ', body)
body = re.sub(r'\s+', ' ', body).strip()

raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', body)
sentences = [s.strip() for s in raw_sentences if len(s.split()) >= 3]

openers = [s.split()[0].lower() for s in sentences if s.split()]
opener_counts = Counter(openers)
print('\n=== SENTENCE OPENER DISTRIBUTION (TOP 15) ===')
for word, count in opener_counts.most_common(15):
    print(f"  '{word}': {count} ({count/len(sentences)*100:.1f}%)")

# Check for repetitive transitions
transitions = ['however', 'therefore', 'consequently', 'similarly', 'specifically', 'in contrast', 'further', 'moreover', 'furthermore']
print('\n=== TRANSITION ADVERB COUNTS ===')
for tr in transitions:
    count = len(re.findall(r'\b' + tr + r'\b', body.lower()))
    print(f"  '{tr}': {count}")
