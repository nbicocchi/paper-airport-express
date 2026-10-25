"""Create a deterministic flat Editorial Manager draft archive.

No account access or submission is performed. Author prompts are deliberately
retained until human metadata, declarations and approval are supplied.
"""
from pathlib import Path
import hashlib
import json
import re
import zipfile

from publication_paths import PAPER
P=PAPER

def main():
    content={}
    main=(P/'main.tex').read_text().replace(r'\graphicspath{{figures/}}','')
    main=re.sub(r'\\input\{tables/([^}]+)\}',r'\\input{\1}',main)
    content['main.tex']=main.encode()
    for name in ['review.tex','supplement.tex','references.bib',
                 'elsarticle.cls','elsarticle-num.bst','ELSARTICLE-LICENSE','main.bbl','review.bbl',
                 'main.pdf','review.pdf','supplement.pdf','highlights.txt','cover_letter.md',
                 'author_completion.md','journal_requirements.md','publication_checks.json']:
        content[name]=(P/name).read_bytes()
    for path in re.findall(r'\\input\{([^}]+)\}',main):
        content[path]=(P/'tables'/path).read_bytes()
    for path in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',main):
        content[path]=(P/'figures'/path).read_bytes()
    content['BUILD.txt']=(
        'DRAFT PACKAGE: author metadata/declarations, data archival confirmation and human approval are pending.\n'
        'Compile main.tex or review.tex with pdflatex, bibtex, pdflatex, pdflatex.\n'
        'Compile supplement.tex twice with pdflatex. All files are at one folder level.\n'
        'The two manuscript PDFs contain the same text in different layouts; select one manuscript for submission.\n'
        'PDFs are reading outputs. Editable LaTeX sources and separate vector figures are included.\n'
        'Do not upload the author worksheet or preparation checklist as publication content.\n'
        'Complete author_completion.md and remove draft prompts before actual submission.\n'
    ).encode()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in sorted(content.items())}
    manifest={'status':'draft, not submitted','files_sha256':hashes,'flat_layout':True}
    content['MANIFEST.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    dest=P/'submission-draft.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,data in sorted(content.items()):
            assert '/' not in name and '\\' not in name,name
            info=zipfile.ZipInfo(name,date_time=(2026,10,10,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,data)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,sha in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==sha
    (P/'submission_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Built {dest.name}: {len(content)} flat files, {dest.stat().st_size} bytes; draft only.')

if __name__=='__main__':main()
