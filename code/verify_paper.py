"""Check manuscript references, generated assets, and the stated protocol."""
import hashlib
import json
import re
from publication_paths import PAPER


def main():
    tex = (PAPER/'main.tex').read_text()
    # Match literal LaTeX commands rather than regular-expression escapes.
    abstract = tex.split(r'\begin{abstract}', 1)[1].split(r'\end{abstract}', 1)[0]
    words = len(abstract.split())
    assert words <= 250, words
    keywords = tex.split(r'\begin{keyword}', 1)[1].split(r'\end{keyword}', 1)[0].split(r'\sep')
    assert 1 <= len(keywords) <= 7
    bib = (PAPER/'references.bib').read_text()
    keys = re.findall(r'@\w+\{([^,]+),', bib)
    assert len(keys) == len(set(keys))
    cites = set()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex):
        cites.update(group.split(','))
    assert cites <= set(keys), cites-set(keys)
    assert 'elsarticle-num' in tex
    assert not re.search(r'\b(?:draft|preliminary|midnight|CAL-REG|RIDGE)\b', tex, re.I)
    assert 'March--May 2025' in tex
    inputs = re.findall(r'\\input\{([^}]+)\}', tex)
    for name in inputs:
        assert (PAPER/name).is_file(), name
    figures = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', tex)
    for name in figures:
        assert (PAPER/'figures'/name).is_file(), name
    assert len(figures) == 5
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels))
    refs = re.findall(r'\\ref\{([^}]+)\}', tex)
    assert set(refs) <= set(labels), set(refs)-set(labels)
    report = dict(abstract_words_whitespace=words, keywords=len(keywords), cited_references=len(cites), figures=len(figures), main_tables=tex.count(r'\begin{table}'), generated_table_inputs=len(inputs), protocol='single pre-March fit, March--May 2025 holdout', source_sha256=hashlib.sha256(tex.encode()).hexdigest())
    (PAPER/'publication_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
