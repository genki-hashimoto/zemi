#texコンパイル用Makefile
FILENAME = clustering
CNT := 2

.SUFFIXES: .pdf .tex .dvi

$(FILENAME).pdf: $(FILENAME).dvi
	dvipdfmx $(FILENAME)

$(FILENAME).dvi: $(FILENAME).tex
	platex -kanji=utf-8 $(FILENAME).tex

$(FILENAME).dvi: $(FILENAME).tex
	platex -kanji=utf-8 $(FILENAME).tex


clean:
	rm *.dvi *.log *.aux
