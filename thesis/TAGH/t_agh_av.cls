%% TAGH/t_agh_av.cls
%% A class for thesis documents.
%%
%% AV/AGH-IT/2023
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{TAGH/t_agh_av}[27-07-2023 v1.0 AV AGH-IT thesis Class]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{report}}
\ProcessOptions\relax
\LoadClass[a4paper,twoside,openright]{report}

%% Packages
\RequirePackage{fontspec}
\RequirePackage{geometry}
\RequirePackage{graphicx}
\RequirePackage{setspace}
\RequirePackage{xcolor}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{mathtools}
\RequirePackage{amsfonts,amsmath,amsthm}
\RequirePackage{url}
\RequirePackage{tocloft}
\RequirePackage[font=small, labelfont=bf, labelsep=period]{caption}
\RequirePackage{subcaption}
\RequirePackage{siunitx}
\RequirePackage{enumerate}
\RequirePackage{enumitem}
\RequirePackage[all,defaultlines=2]{nowidow}
\RequirePackage[english,polish]{babel}
\RequirePackage[
  style=numeric,
  sorting=none,
  language=autobib,
  autolang=other,
  urldate=iso,
  backref=false,
  isbn=true,
  url=false,
  maxbibnames=9,
  seconds=true,
  backend=biber
]{biblatex}
\RequirePackage{csquotes}
\RequirePackage{courier}
\RequirePackage{listings}
\RequirePackage{array}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{booktabs}
\RequirePackage{makecell}
\RequirePackage[flushleft]{threeparttable}

%% lstset
\lstloadlanguages{TeX}
\lstset{
  literate=
    {a}{{\k{a}}}1
    {c}{{\'{c}}}1
    {e}{{\k{e}}}1
    {o}{{\'{o}}}1
    {n}{{\'{n}}}1
    {l}{{\l{}}}1
    {s}{{\'{s}}}1
    {z}{{\'{z}}}1
    {z}{{\.z}}1
    {A}{{\k{A}}}1
    {C}{{\'{C}}}1
    {E}{{\k{E}}}1
    {O}{{\'{O}}}1
    {N}{{\'{N}}}1
    {L}{{\L{}}}1
    {S}{{\'{S}}}1
    {Z}{{\'{Z}}}1
    {Z}{{\.Z}}1,
  basicstyle=\footnotesize\ttfamily,
}

\newcolumntype{C}[1]{>{\hsize=#1\hsize\centering\arraybackslash}X}
\setlength{\cftsecnumwidth}{10mm}
\setcounter{secnumdepth}{4}
\brokenpenalty=10000\relax

\fancypagestyle{plain}{%
  \fancyhf{}
  \renewcommand{\headrulewidth}{0pt}
  \renewcommand{\footrulewidth}{0pt}
}

\setcounter{tocdepth}{2}
\setmainfont{Times New Roman}
\setstretch{1.5}
\geometry{
  a4paper,
  lmargin=30mm,
  rmargin=20mm,
  tmargin=30mm,
  bmargin=30mm,
  foot=10mm,
  head=10mm
}

%% User variables
\newcommand*\titlePL[1]{\def\@titlePL{#1}}
\newcommand*\titleEN[1]{\def\@titleEN{#1}}
\def\shorttitlePL#1 {\gdef\@shorttitlePL{#1}}
\def\shorttitleEN#1 {\gdef\@shorttitleEN{#1}}
\def\shortauthor#1 {\gdef\@shortauthor{#1}}
\def\faculty#1 {\gdef\@faculty{#1}}
\def\thesistype#1 {\gdef\@thesistype{#1}}
\def\supervisor#1 {\gdef\@supervisor{#1}}
\def\acknowledgements#1 {\gdef\@acknowledgements{#1}}
\def\department#1 {\gdef\@department{#1}}
\def\degreeprogramme#1 {\gdef\@degreeprogramme{#1}}

\renewcommand{\labelitemi}{--}

\DeclareOption{pl}{%
  \def\thesisheaders{%
    \fancyfoot[LE,RO]{\small \@shortauthor\quad\textit{\@shorttitlePL}}%
  }
  \def\thesistable{%
    \begin{tabular}{p{45mm}l}
      Autor:                    & {\itshape \@author}         \\[-1mm]
      Kierunek studi\'{o}w:     & {\itshape \@degreeprogramme}\\[-1mm]
      Opiekun pracy:            & {\itshape \@supervisor}     \\
    \end{tabular}%
  }
}

\DeclareOption{en}{%
  \def\thesisheaders{%
    \fancyfoot[LE,RO]{\small \@shortauthor\quad\textit{\@shorttitleEN}}%
  }
  \def\thesistable{%
    \begin{tabular}{p{45mm}l}
      Author:           & {\itshape \@author}         \\[-1mm]
      Degree programme: & {\itshape \@degreeprogramme}\\[-1mm]
      Supervisor:       & {\itshape \@supervisor}     \\
    \end{tabular}%
  }
}

\ProcessOptions

\newcommand{\titlepages}{%
  \newpage\titlepage
  \begin{center}
    \vspace*{0mm}
    \includegraphics[scale=0.6]{TAGH/agh}
    \vspace*{2mm}

    {\sffamily
      {\large\bfseries\scshape AGH University of Krakow}\par
      \vspace*{1mm}
      {\small\bfseries \MakeUppercase{\@faculty}}\par
    }
    \vspace*{8mm}

    {\sffamily\Large \@thesistype}\par
    \vspace*{9mm}

    {\sffamily\Large\itshape \@titleEN}\par
    \vspace*{1mm}
    {\sffamily\Large\itshape \@titlePL}\par
    \vspace{20mm}
  \end{center}

  \noindent
  {\sffamily\thesistable}
  \vspace*{9mm}

  \centerline{\small\sffamily Krak\'{o}w, \@date}

  \clearpage
  \thispagestyle{empty}

  {~~}\vspace{20mm}

  {\large
    Uprzedzony o~odpowiedzialno\'sci karnej na~podstawie art.~115 ust.~1~i~2
    ustawy z~dnia 4~lutego 1994~r. o~prawie autorskim i~prawach pokrewnych
    (t.j. Dz.U. z~2006~r. Nr~90, poz.~631 z~p{\'o}{\'z}n.\ zm.):
    ,,Kto przyw\l{}aszcza sobie autorstwo albo~wprowadza w~b\l{}\k{a}d
    co~do~autorstwa ca\l{}o\'sci lub cz\k{e}\'sci cudzego utworu albo~artystycznego
    wykonania, podlega grzywnie, karze ograniczenia wolno\'sci albo~pozbawienia
    wolno\'sci do~lat~3. Tej~samej karze podlega, kto~rozpowszechnia bez~podania
    nazwiska lub~pseudonimu tw\'orcy cudzy utw\'or w~wersji oryginalnej albo~w~postaci
    opracowania, artystycznego wykonania albo~publicznie zniekszta\l{}ca taki utw\'or,
    artystyczne wykonanie, fonogram, wideogram lub nadanie.'',
    a~tak\.ze uprzedzony o~odpowiedzialno\'sci dyscyplinarnej na~podstawie art.~211
    ust.~1 ustawy z~dnia 27~lipca 2005~r.\ Prawo o~szkolnictwie wy\.zszym
    (t.j.\ Dz.~U. z~2012~r.\ poz.~572, z~p\'o\'zn.\ zm.):
    ,,Za~naruszenie przepis\'ow obowi\k{a}zuj\k{a}cych w~uczelni oraz~za~czyny
    uchybiaj\k{a}ce godno\'sci studenta student ponosi odpowiedzialno\'s\'c
    dyscyplinarn\k{a} przed komisj\k{a} dyscyplinarn\k{a} albo~przed
    s\k{a}dem kole\.ze\'nskim samorz\k{a}du studenckiego, zwanym dalej
    <>.'',
    o\'swiadczam, \.ze~niniejsz\k{a} prac\k{e} dyplomow\k{a} wykona\l{}em(-am)
    osobi\'scie i~samodzielnie i~\.ze~nie~korzysta\l{}em(-am) ze~\'zr\'ode\l{}
    innych ni\.z~wymienione w~pracy.
  }

  \clearpage\titlepage

  \setcounter{page}{5}

  \pagestyle{fancy}
  \renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ ##1}{}}
  \renewcommand{\sectionmark}[1]{\markright{\thesection.\ ##1}}
  \renewcommand{\headrulewidth}{0.5pt}
  \renewcommand{\footrulewidth}{0.5pt}
  \fancyhf{}
  \fancyhead[RE,LO]{\slshape{\small\rightmark}}
  \fancyhead[RO,LE]{\bfseries\thepage}

  \thesisheaders

  \frenchspacing
  \sloppy
  \flushbottom

  %% Formatowanie tytulow rozdzialow
  \titleformat{\chapter}[block]
    {\bfseries\LARGE}{\filright\LARGE\thechapter.\ }{0ex}{}
  \titlespacing{\chapter}{0pt}{17mm}{19mm}
  \titleformat{\section}[block]
    {\bfseries\Large}{\filright\Large\thesection.\ }{0ex}{}
  \titlespacing{\section}{7mm}{8mm plus 0mm minus 1mm}{4mm plus 0mm minus 1mm}
  \titleformat{\subsection}[block]
    {\bfseries\large}{\filright\large\thesubsection.\ }{0ex}{}
  \titlespacing{\subsection}{7mm}{6mm}{2.5mm}
  \titleformat{\subsubsection}[block]
    {\bfseries\normalsize}{\filright\normalsize\thesubsubsection.\ }{0ex}{}
  \titlespacing{\subsubsection}{7mm}{3mm}{1.5mm}

  %% Spis tresci
  \renewcommand{\cfttoctitlefont}{\bfseries\Large}
  \renewcommand{\cftbeforetoctitleskip}{20mm}
  \renewcommand{\cftaftertoctitleskip}{19mm}
  \renewcommand{\cftchapleader}{\cftdotfill{\cftdot}}
  \renewcommand{\cftsecleader}{\cftdotfill{\cftdot}}
  \renewcommand{\cftsubsecleader}{\cftdotfill{\cftdot}}
  \renewcommand{\cftchapaftersnum}{.}
  \renewcommand{\cftsecaftersnum}{.}
  \renewcommand{\cftsubsecaftersnum}{.}
  \setlength{\cftparskip}{2pt}
  \setlength{\cftbeforechapskip}{0mm}
  \renewcommand{\cftchappagefont}{\normalsize}
}% END \titlepages

%% Ustawienia liczb
\sisetup{
  output-decimal-marker = {,},
  group-separator={~},
}

\renewcommand\UrlFont{\rmfamily\itshape}

\captionsetup[subfigure]{labelfont=md}
\captionsetup{%
  subrefformat=parens,
  margin=2cm,
  singlelinecheck=on,
}

\SetLabelAlign{parright}{\parbox[t]{\labelwidth}{\raggedleft#1}}
\setlist[description]{style=multiline, topsep=10pt, align=parright}

\newlength{\descriptiondashgap}
\setlength{\descriptiondashgap}{0.5em}

\renewcommand*\descriptionlabel[1]{%
  \hspace\labelsep\normalfont\bfseries #1\mdseries
  \hspace{\descriptiondashgap}--\hspace{\descriptiondashgap}%
}

\newlength{\eqwheremargin}
\setlength{\eqwheremargin}{2cm}

\newenvironment{eqwhere}[1][]
  {\setlength{\eqwheremargin}{#1}%
   \begin{description}[leftmargin=\eqwheremargin, itemsep=0cm, labelsep=0cm]}
  {\end{description}}
