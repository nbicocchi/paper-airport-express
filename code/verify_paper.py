"""Check journal constraints, references and generated publication assets.

Uses Python's standard library; PDF text/layout checks remain a separate build QA.
"""
from pathlib import Path
import json
import re
import hashlib

from publication_paths import PAPER
P=PAPER

def main():
    tex=(P/'main.tex').read_text()
    abstract=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',tex,re.S).group(1)
    words=len(abstract.split())
    assert words<=250, words
    keywords=re.search(r'\\begin\{keyword\}(.*?)\\end\{keyword\}',tex,re.S).group(1).split(r'\sep')
    assert 1<=len(keywords)<=7
    highlights=(P/'highlights.txt').read_text().splitlines()
    assert 3<=len(highlights)<=5
    assert all(0<len(x)<=85 for x in highlights), [(x,len(x)) for x in highlights]
    bib=(P/'references.bib').read_text()
    keys=re.findall(r'@\w+\{([^,]+),',bib)
    cites=set()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}',tex):cites.update(group.split(','))
    assert len(keys)==len(set(keys))==30
    assert cites==set(keys), (cites-set(keys),set(keys)-cites)
    assert 'IEEEtran' not in tex
    assert 'elsarticle-num' in tex
    inputs=re.findall(r'\\input\{([^}]+)\}',tex)
    for x in inputs:assert (P/x).is_file(),x
    figures=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',tex)
    for x in figures:assert (P/'figures'/x).is_file(),x
    assert len(figures)==5
    supp=(P/'supplement.tex').read_text()
    assert supp.count(r'\begin{table}')==13
    # Output figures and source files must remain editable/vector, not raster tables.
    assert r'\begin{tabular}' in supp
    report={'abstract_words_whitespace':words,'keywords':len(keywords),
            'highlights_lengths':[len(x) for x in highlights],'cited_references':len(cites),
            'figures':len(figures),'main_tables':tex.count(r'\begin{table*}'),
            'supplement_tables':13,'submission_status':'draft: author completion and approval required',
            'source_sha256':hashlib.sha256(tex.encode()).hexdigest()}
    (P/'publication_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
