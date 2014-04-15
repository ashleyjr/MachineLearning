pdflatex report.tex
bibtex report
pdflatex report.tex 
pdflatex report.tex 
texcount -v -html -inc report.tex > count.html
del .pdf
del *.aux
del *.bbl
del *.blg
del *.log
del *.synctex.gz
del *.toc

