"""Build the main paper using Springer Nature's unmodified recommended class."""
from pathlib import Path
import os, re, subprocess, json, hashlib, shutil
from build_sncs_latex_legacy import render_md, COMMON
R=Path(__file__).resolve().parent; M=R/'manuscript'
MAIN_COMMON=r'''
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc,xurl,xcolor,needspace}
\let\burl\url
\hypersetup{hidelinks,pdfauthor={Tivdzua Lubem Noah; Bader Rasheed; Dmitriy S. Vatolin}}
\raggedbottom
'''
def main_body(text,pandoc):
    body=render_md(text,pandoc)
    body=re.sub(r'(\\(?:sub)*section\{)\d+(?:\.\d+)*\.?\s+',r'\1',body)
    body=re.sub(r'\\Needspace\{\d+\\baselineskip\}',lambda m:r'\Needspace{10\baselineskip}',body)
    body=body.replace(r'\tightlist',r'\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}')
    body=body.replace(r'\section{Statements and Declarations}',r'\backmatter'+ '\n'+r'\section*{Statements and Declarations}')
    for name in ['Funding','Competing interests','Author contributions','Data and code availability']:
        body=body.replace(r'\subsection{'+name+'}',r'\bmhead{'+name+'}')
    body=re.sub(r'(\\caption\{[^\n]+)\.\}',r'\1}',body)
    for i,old in enumerate(['fig1_computing_stable_progress.pdf','fig2_computing_legacy_progress.pdf','fig3_computing_peak_progress.pdf'],1):
        shutil.copyfile(R/'figures'/old,M/f'Fig{i}.pdf')
        body=body.replace('../figures/'+old,f'Fig{i}.pdf')
    return body

def build():
    pandoc=os.environ['PANDOC'];tectonic=os.environ['TECTONIC']
    for name in ['sn-jnl.cls','sn-mathphys-num.bst']:shutil.copyfile(R/'journal_template'/name,M/name)
    report={'format':'sn-jnl referee sn-mathphys-num','template_version':'3.1 December 2024','regular_paper_page_limit':35,'documents':[]}
    for stem in ['Research_Manuscript','All_Seed_Results_and_Provenance']:
        md=(M/(stem+'.md')).read_text(encoding='utf8'); title,body=md.split('\n',1);title=title.removeprefix('# ')
        if stem=='Research_Manuscript':
            abstract=body.split('## Abstract\n\n',1)[1].split('\n\n**Keywords:**',1)[0]
            kws=body.split('**Keywords:** ',1)[1].split('\n',1)[0].rstrip('.')
            body='## 1. Introduction'+body.split('## 1. Introduction',1)[1];body=body.split('\n## References',1)[0]
            head=r'\documentclass[referee,pdflatex,sn-mathphys-num]{sn-jnl}'+'\n'+MAIN_COMMON+'\n'+r'\begin{document}'+'\n'
            head+='\\title[First-step faults and scheduler delay in Adam]{'+title+'}\n'
            head+=r'''
\author*[1]{Tivdzua Lubem Noah}\email{l.tivdzua@innopolis.university}
\author[1]{Bader Rasheed}\email{b.rasheed@innopolis.university}
\author[1]{Dmitriy S. Vatolin}\email{dmitriy@graphics.cs.msu.ru}
\affil[1]{\orgname{Innopolis University}, \orgaddress{\city{Innopolis}, \country{Russia}}}
\affil[]{ORCID (Tivdzua Lubem Noah): \href{https://orcid.org/0009-0009-2511-1800}{0009-0009-2511-1800}}
'''
            head+='\\abstract{'+render_md(abstract,pandoc)+'}\n\\keywords{'+kws.replace(';',',')+'}\n\\maketitle\n'
            tex=head+main_body(body,pandoc)+'\n\\bibliography{references}\n\\end{document}\n'
        else:
            head=r'\documentclass[11pt,a4paper]{article}'+'\n'+r'\usepackage[margin=24mm]{geometry}'+'\n'+COMMON+'\n\\renewcommand{\\thefigure}{S\\arabic{figure}}\n\\begin{document}\n\\begin{center}{\\Large\\bfseries '+title+'}\\end{center}\n'
            head+='\\AtBeginEnvironment{longtable}{\\small}\n'
            tex=head+render_md(body,pandoc)+'\n\\end{document}\n'
        path=M/(stem+'.tex');path.write_text(tex,encoding='utf8')
        result=subprocess.run([tectonic,'--untrusted','--keep-logs','--keep-intermediates',path.name],cwd=M,capture_output=True,text=True,encoding='utf8',errors='replace')
        if stem=='Research_Manuscript' and result.returncode==0:
            # The official style prints an invented ??? location for online
            # proceedings without a publisher city. Omit that placeholder only;
            # preserve the bibliography facts, class and official style files.
            bbl=(M/(stem+'.bbl')).read_text(encoding='utf8')
            removed=bbl.count(r', \blocation{???}')
            bbl=bbl.replace(r', \blocation{???}','')
            assert '???' not in bbl
            tex=tex.replace(r'\bibliography{references}',bbl)
            path.write_text(tex,encoding='utf8')
            (M/(stem+'.bbl')).write_text(bbl,encoding='utf8')
            report['bibliography']={'embedded_for_portability':True,'unavailable_publisher_location_placeholders_omitted':removed,'bibliographic_facts_preserved':True}
            result=subprocess.run([tectonic,'--untrusted','--keep-logs','--keep-intermediates',path.name],cwd=M,capture_output=True,text=True,encoding='utf8',errors='replace')
        (R/'evidence/sncs'/(stem+'_build.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
        if result.returncode:raise RuntimeError(stem+' build failed: '+result.stderr[-2500:])
        report['documents'].append({'stem':stem,'tex_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256((M/(stem+'.pdf')).read_bytes()).hexdigest()})
    (R/'evidence/sncs/latex_build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':build()
