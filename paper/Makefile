.PHONY: all main clean
all:
	$(MAKE) main
main:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -c main.tex
