#texコンパイル用Makefile
FILENAME = svm
CNT := 2

.SUFFIXES: .pdf .tex .dvi

$(FILENAME).pdf: $(FILENAME).dvi
	dvipdfmx $(FILENAME)

$(FILENAME).dvi: $(FILENAME).tex
	for i in `seq 1 $(CNT)`; do platex -kanji=utf-8 $(FILENAME).tex; done



clean:
	rm *.dvi *.log *.aux
