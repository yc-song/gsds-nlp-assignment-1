%% file ETHbrief.cls

%% --- Identification part ---
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{assign_format}[2022/08/27 v1.0]

%% --- Declaration of options ---
%%\DeclareOption*{\PassOptionsToClass{\CurrentOption}{letter}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

%% --- Execution of options ---
\ProcessOptions

%% --- Package loading ---
%\RequirePackage{ETHlogo}
\RequirePackage[utf8]{inputenc}

%% --- Special commands for package and class files ---

%% --- Special class file commands ---
%%\LoadClass[a4paper]{letter}
\LoadClass[a4paper]{article}

%% --- Defining the (sub-)class ---
\setlength{\textwidth}{16.5cm}
\setlength{\marginparwidth}{1.5cm}
%\setlength{\marginparwidth}{4cm}
%\setlength{\marginparwidth}{0cm}
\setlength{\parindent}{0cm}
\setlength{\parskip}{0.15cm}
\setlength{\textheight}{25cm}
\setlength{\oddsidemargin}{0cm}
\setlength{\evensidemargin}{\oddsidemargin}
\setlength{\topmargin}{0cm}
\setlength{\headheight}{0cm}
\setlength{\headsep}{0cm}

% \renewcommand{\familydefault}{\sfdefault}

% text parameters
\newcommand{\EinheitNameText}{}
\newcommand{\EinheitInfoText}{}
\newcommand{\PersonNameText}{}
\newcommand{\PersonAdresseText}{}
\newcommand{\PersonInfoText}{}
\newcommand{\OrtText}{Z\"urich}
\newcommand{\BetreffText}{}
\newcommand{\KurstitelText}{}
\newcommand{\SemesterText}{}

\newcommand{\EinheitName}  [1]{\renewcommand{\EinheitNameText}{#1}}
\newcommand{\EinheitInfo}  [1]{\renewcommand{\EinheitInfoText}{#1}}
\newcommand{\PersonName}   [1]{\renewcommand{\PersonNameText}{#1}}
\newcommand{\PersonAdresse}[1]{\renewcommand{\PersonAdresseText}{#1}}
\newcommand{\PersonInfo}   [1]{\renewcommand{\PersonInfoText}{#1}}
\newcommand{\Ort}          [1]{\renewcommand{\OrtText}{#1}}
\newcommand{\Betreff}      [1]{\renewcommand{\BetreffText}{#1}}
\newcommand{\Kurstitel}    [1]{\renewcommand{\KurstitelText}{#1}}
\newcommand{\Semester}    [1]{\renewcommand{\SemesterText}{#1}}

% English commands as synonyms (Fritz Zaucker, 2001-01-29)
\let\UnitName\EinheitName
\let\UnitInfo\EinheitInfo
\let\PersonAddress\PersonAdresse
\let\City\Ort
\let\Subject\Betreff
\let\subject\Betreff

\newcommand{\makeheader}[2]{%
  \thispagestyle{empty} % no headers and footers (i.e. page numbers)
                        % on the first page
  \setlength{\unitlength}{1mm}
  \begin{picture}(165,40)
    \put(98,40){%
      \begin{minipage}[t]{90mm}
        \rule{90mm}{0mm}\\
        \textbf{\EinheitNameText}\vspace{1mm}\\
        \EinheitInfoText
        \vspace{1mm} \\
        \textbf{\small\PersonNameText}
        \vspace{1mm} \\
        {\small\PersonInfoText}
      \end{minipage}
      }
    \put(0,40){%
      \begin{minipage}[t]{98mm}
        \rule{98mm}{0mm}\\
        \raggedright {Assignments} \\ \textbf{\large\KurstitelText}\\ { \SemesterText} \par
      \end{minipage}
      }
    \put(0,16.5){%
      \begin{minipage}[t]{250mm}
        \textbf{\Large #1\\[1.0ex] (#2)}
      \end{minipage}
    }
  \end{picture}
  \par
}

\newcommand{\Problem}[2]{%
  \textbf{Problem #1:}
%  \vspace{5mm}
  \par
  #2
  \par
  \vspace{5mm}
}

\newcommand{\ProblemV}[3]{%
  \textbf{Problem #1 (#2):}
%  \vspace{5mm}
  \par
  #3
  \par
  \vspace{5mm}
}

\newcommand{\Solution}[2]{%
  \textbf{Solution #1:}
  \vspace{5mm}
  \par
  #2
  \par
  \vspace{5mm}
}

\newcommand{\SolutionV}[3]{%
  \textbf{Solution #1 (#2):}
  \vspace{5mm}
  \par
  #3
  \par
  \vspace{5mm}
}