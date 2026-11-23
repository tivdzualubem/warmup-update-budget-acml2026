"""Apply the journal-approved linked-supplement delivery wording."""
from pathlib import Path
R=Path(__file__).resolve().parent
VERSION='sncs-submission-2026-11-23'
REPO='https://github.com/tivdzualubem/warmup-update-budget-acml2026'
URL=REPO+'/tree/'+VERSION
DOWNLOAD=REPO+'/raw/refs/tags/'+VERSION+'/submission/sncs-2026-11-23/'
def build():
    p=R/'manuscript/Research_Manuscript.md'
    s=p.read_text(encoding='utf8')
    begin=s.index('### Data and code availability')
    end=s.index('## References',begin)
    previous=s[begin:end]
    tail=previous[previous.index('The larger prediction'):]
    section=('### Data and code availability\n\n'
       'The supplementary materials are available through the following links in the '
       '[versioned research repository]('+URL+') (version `'+VERSION+'`).\n\n'
       '[Online Resource 1: supplementary results and provenance]('+DOWNLOAD+'ESM_1.pdf) '
       'contains the complete seed-level endpoint results, recorded training trajectories, budgets, provenance and evaluation limitations.\n\n'
       '[Online Resource 2: reproducibility package]('+DOWNLOAD+'ESM_2.zip) '
       'contains the code, configurations, retained experimental records, tests, execution logs and editable manuscript sources used to regenerate the reported tables and figures. '
       'The same files can be browsed in the repository’s `sncs/` directory. Earlier experiment and rebuttal code and the previous Computing release are retained separately.\n\n'
       +tail)
    p.write_text(s[:begin]+section+s[end:],encoding='utf8')
    p=R/'manuscript/All_Seed_Results_and_Provenance.md'
    s=p.read_text(encoding='utf8')
    start=s.index('Preserved scientific code, retained records and earlier editable sources:')
    end=s.index('\n',start)
    s=s[:start]+('Research code, retained records and editable sources: [version '+VERSION+']('+URL+'). '
       '[Download Online Resource 2: reproducibility package]('+DOWNLOAD+'ESM_2.zip).')+s[end:]
    p.write_text(s,encoding='utf8')
if __name__=='__main__':build()
