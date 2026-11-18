"""Export complete editable LaTeX/BibTeX sources and optionally compile them.

Requires Pandoc for source regeneration; shipped .tex sources compile directly
with Tectonic or XeLaTeX/BibTeX without Python or Pandoc.
"""
from pathlib import Path
import argparse, hashlib, json, os, re, shutil, subprocess

ROOT=Path(__file__).resolve().parent
M=ROOT/'manuscript'
CITATIONS={
 'Abuduweili and Liu (2025)':'abuduweili2025initial',
 'Alimisis, Islamov and Lucchi, 2026':'alimisis2026warmup',
 'Bai et al., 2026':'bai2026spikes',
 'Gotmare et al. (2019)':'gotmare2019heuristics',
 'Guo et al., 2017':'guo2017calibration',
 'Huang et al. (2020)':'huang2020initialization',
 'Kalra and Barkeshli (2024)':'kalra2024warmup',
 'Kingma and Ba (2015)':'kingma2015adam',
 'Kosson, Messmer and Jaggi (2024)':'kosson2024warmup',
 'Kosson et al. (2024)':'kosson2024warmup',
 'Liu et al. (2020)':'liu2020variance',
 'Loshchilov and Hutter (2019)':'loshchilov2019decoupled',
 'Loshchilov and Hutter, 2019':'loshchilov2019decoupled',
 'Ma and Yarats (2021)':'ma2021untuned',
 'Shazeer and Stern (2018)':'shazeer2018adafactor',
 'Topollai and Choromanska, 2026, preprint':'topollai2026horizons',
 'Vaswani et al. (2017)':'vaswani2017attention',
 'Xiong et al. (2020)':'xiong2020layernorm',
}
PREAMBLE=r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=24mm]{geometry}
\usepackage{fontspec}
\setmainfont{texgyretermes-regular.otf}[BoldFont=texgyretermes-bold.otf,ItalicFont=texgyretermes-italic.otf,BoldItalicFont=texgyretermes-bolditalic.otf]
\setsansfont{texgyreheros-regular.otf}[BoldFont=texgyreheros-bold.otf]
\setmonofont{lmmono10-regular.otf}[Scale=MatchLowercase]
\usepackage{amsmath,amssymb,graphicx,longtable,booktabs,array,calc}
\usepackage{etoolbox,xcolor,xurl,caption,float,needspace}
\usepackage[round,authoryear]{natbib}
\usepackage[unicode,colorlinks=true,linkcolor=black,citecolor=black,urlcolor=blue]{hyperref}
\hypersetup{pdfauthor={},pdfsubject={Research manuscript},pdfkeywords={Adam; learning-rate warmup}}
\setcounter{secnumdepth}{0}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt}
\setlength{\emergencystretch}{3em}
\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000
\predisplaypenalty=10000
\setlength{\LTpre}{5pt}
\setlength{\LTpost}{8pt}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{4pt}}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\makeatletter
\def\maxwidth{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}
\def\maxheight{\ifdim\Gin@nat@height>0.72\textheight0.72\textheight\else\Gin@nat@height\fi}
\makeatother
\setkeys{Gin}{width=\maxwidth,height=\maxheight,keepaspectratio}
\date{}
\author{}
'''

def tool(name, explicit):
    found=explicit or os.environ.get(name.upper()) or shutil.which(name)
    if not found:
        raise SystemExit(f'{name} not found. Set {name.upper()} to its executable or add it to PATH. Existing .tex files can be compiled directly.')
    return str(found)

def run(pandoc, tectonic=None):
    evidence=ROOT/'evidence/verification'; evidence.mkdir(parents=True,exist_ok=True)
    report={'source_format':'LaTeX with BibTeX','pandoc_version':subprocess.check_output([pandoc,'--version'],text=True).splitlines()[0],'documents':[]}
    for stem in ('Research_Manuscript','All_Seed_Results_and_Provenance'):
        text=(M/(stem+'.md')).read_text(encoding='utf-8')
        title,text=text.split('\n',1); title=title.removeprefix('# ')
        if stem=='Research_Manuscript':
            text=text.split('\n## References',1)[0]
            for original,key in sorted(CITATIONS.items(),key=lambda p:-len(p[0])):
                command='citealp' if '(' not in original else 'citet'
                text=text.replace(original,'\\'+command+'{'+key+'}')
        text=text.replace('\u2013','-').replace('\u2014','; ').replace('\u2011','-')
        text=re.sub(r'(?m)^(?=## Table S|\*\*Table )',lambda m:'\\Needspace{14\\baselineskip}\n\n',text)
        # Pandoc consumes Markdown math/tables; source content remains editable.
        proc=subprocess.run([pandoc,'--from=markdown+tex_math_dollars+raw_tex','--to=latex','--natbib','--wrap=none'],input=text,text=True,encoding='utf-8',capture_output=True,check=True)
        body=proc.stdout
        body=re.sub(r'([^\n]+)\n\n(\\\[[\s\S]*?\\\])',lambda m:'\\begin{samepage}\n'+m[1]+'\n\n'+m[2]+'\n\\end{samepage}',body)
        body=re.sub(r'\\begin\{figure\}(?:\[[^\]]*\])?',lambda m:r'\begin{figure}[H]',body)
        # Constrain tall figure panels on the page and keep captions readable.
        body=re.sub(r'\\includegraphics\[([^\]]*)\]',lambda m:'\\includegraphics['+m[1]+',height=0.69\\textheight,keepaspectratio]',body)
        ending=''
        if stem=='Research_Manuscript':
            ending=r'\bibliographystyle{plainnat}'+'\n'+r'\nocite{*}'+'\n'+r'\bibliography{references}'+'\n'
        extra='' if stem=='Research_Manuscript' else r'\renewcommand{\thefigure}{S\arabic{figure}}'+'\n'
        tex=PREAMBLE+'\n\\begin{document}\n'+extra+'\\begin{center}{\\LARGE\\bfseries '+title+'}\\end{center}\n\\vspace{0.5em}\n'+body+ending+'\\end{document}\n'
        path=M/(stem+'.tex'); path.write_text(tex,encoding='utf-8')
        item={'source':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if tectonic:
            command=[tectonic,'--untrusted','--keep-logs','--keep-intermediates',path.name]
            compiled=subprocess.run(command,cwd=M,text=True,encoding='utf-8',errors='replace',capture_output=True)
            (evidence/(stem+'_latex_build.log')).write_text(compiled.stdout+'\n'+compiled.stderr,encoding='utf-8')
            if compiled.returncode:
                raise SystemExit(f'LaTeX build failed for {stem}; see evidence/verification/{stem}_latex_build.log')
            pdf=M/(stem+'.pdf')
            item.update(pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest())
        report['documents'].append(item)
    if tectonic: report['tectonic_version']=subprocess.check_output([tectonic,'--version'],text=True).strip()
    (evidence/'latex_build.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pandoc');p.add_argument('--tectonic');p.add_argument('--sources-only',action='store_true')
    a=p.parse_args(); run(tool('pandoc',a.pandoc),None if a.sources_only else tool('tectonic',a.tectonic))
