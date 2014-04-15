#!/bin/bash
pdflatex -synctex=1 report.tex 
bibtex report 
pdflatex -synctex=1 report.tex 
pdflatex -synctex=1 report.tex 
texcount -v -html -inc report.tex > count.html
