"""Computing's required SVJour3 smallextended format; no layout compression."""
from pathlib import Path
import os,re,subprocess,json,hashlib,shutil
import build_latex as previous
R=Path(__file__).resolve().parent;M=R/'manuscript'
C=dict(previous.CITATIONS)
C.update({'Ben Braiek and Khomh, 2023':'benbraiek2023deepchecker','Wardat et al., 2022':'wardat2022deepdiagnosis','Kingma and Ba, 2015':'kingma2015adam','Shazeer and Stern, 2018':'shazeer2018adafactor'})
COMMON=r'''
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc}
\usepackage{etoolbox,xurl,float,needspace}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[unicode,hidelinks]{hyperref}
\hypersetup{pdfauthor={},pdfsubject={Research manuscript}}
\setcounter{secnumdepth}{0}
\setlength{\emergencystretch}{3em}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>0.72\textheight0.72\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
'''
def render_md(text,pandoc):
    text=text.replace('\u2013','-').replace('\u2014','; ').replace('\u2011','-').replace('\u2192',r'$\rightarrow$').replace('±',r'$\pm$')
    text=re.sub(r'(?<![\w`])([\w/.-]+\.csv)(?![\w`])',lambda m:r'\path{'+m[1]+'}',text)
    text=re.sub(r'(?m)^(?=## [^\n]+\n\n\*\*Table )',lambda m:'\\Needspace{20\\baselineskip}\n\n',text)
    text=re.sub(r'(?m)^(?=## Table S|\*\*Table )',lambda m:'\\Needspace{16\\baselineskip}\n\n',text)
    for old,key in sorted(C.items(),key=lambda p:-len(p[0])):
        replacement=r'\citep{'+key+'}'
        if '(' in old:
            replacement=old.split(' (',1)[0]+' '+replacement
        text=text.replace(old,replacement)
    text=re.sub(r'\((\\citep\{[^}]+\})\)',r'\1',text)
    proc=subprocess.run([pandoc,'--from=markdown+tex_math_dollars+raw_tex','--to=latex','--natbib','--wrap=none','--shift-heading-level-by=-1'],input=text,text=True,encoding='utf-8',capture_output=True,check=True)
    body=proc.stdout
    body=re.sub(r'\\begin\{figure\}(?:\[[^\]]*\])?',lambda m:r'\begin{figure}[htbp]',body)
    body=re.sub(r'\\includegraphics\[([^\]]*)\]',lambda m:'\\includegraphics['+m[1]+',height=0.72\\textheight,keepaspectratio]',body)
    return body
def build():
    pandoc=os.environ['PANDOC'];tectonic=os.environ['TECTONIC'];report={'format':'svjour3 smallextended','page_limit':25,'author_contacts_complete':True,'competing_interests_confirmed':True,'documents':[]}
    for f in ['svjour3.cls','svglov3.clo','spmpsci.bst']:shutil.copy2(R/'journal_template'/f,M/f)
    for stem in ['Research_Manuscript','All_Seed_Results_and_Provenance']:
        md=(M/(stem+'.md')).read_text(encoding='utf-8');title,body=md.split('\n',1);title=title.removeprefix('# ')
        if stem=='Research_Manuscript':
            abstract=body.split('## Abstract\n\n',1)[1].split('\n\n**Keywords:**',1)[0]
            kws=body.split('**Keywords:** ',1)[1].split('\n',1)[0].rstrip('.')
            body='## 1. Introduction'+body.split('## 1. Introduction',1)[1];body=body.split('\n## References',1)[0]
            head=r'\documentclass[smallextended]{svjour3}'+'\n'+COMMON+'\n\\journalname{Computing}\n\\begin{document}\n'
            head+='\\title{'+title+'}\n\\titlerunning{First-step faults and scheduler delay in Adam}\n'
            head+='\\author{Tivdzua Lubem Noah \\and Bader Rasheed \\and Dmitriy S. Vatolin}\n\\authorrunning{Tivdzua Lubem Noah et al.}\n'
            head+=r'''\institute{Tivdzua Lubem Noah (corresponding author) \\ Innopolis University, Innopolis, Russia \\ \email{l.tivdzua@innopolis.university} \\ ORCID: \href{https://orcid.org/0009-0009-2511-1800}{0009-0009-2511-1800} \\[3pt] Bader Rasheed \\ Innopolis University, Innopolis, Russia \\ \email{b.rasheed@innopolis.university} \\[3pt] Dmitriy S. Vatolin \\ Innopolis University, Innopolis, Russia \\ \email{dmitriy@graphics.cs.msu.ru}}
\date{}
\maketitle
'''
            head+='\\begin{abstract}\n'+render_md(abstract,pandoc)+'\n\\keywords{'+kws.replace(';',r' \and')+'}\n\\end{abstract}\n'
            ending='\n\\bibliographystyle{spmpsci}\n\\bibliography{references}\n'
        else:
            head=r'\documentclass[11pt,a4paper]{article}'+'\n'+r'\usepackage[margin=24mm]{geometry}'+'\n'+COMMON+'\n\\renewcommand{\\thefigure}{S\\arabic{figure}}\n\\begin{document}\n\\begin{center}{\\Large\\bfseries '+title+'}\\end{center}\n'
            # The supplement has wide machine-readable result tables; normal article layout.
            head+='\\AtBeginEnvironment{longtable}{\\small}\n';ending=''
        tex=head+render_md(body,pandoc)+ending+'\n\\end{document}\n'
        path=M/(stem+'.tex');path.write_text(tex,encoding='utf-8')
        result=subprocess.run([tectonic,'--untrusted','--keep-logs','--keep-intermediates',path.name],cwd=M,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (R/'evidence/computing'/(stem+'_build.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError(stem+' build failed: '+result.stderr[-1500:])
        report['documents'].append({'stem':stem,'tex_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256((M/(stem+'.pdf')).read_bytes()).hexdigest()})
    (R/'evidence/computing/latex_build.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':build()
