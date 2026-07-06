MAIN = main
TEX_CHECK_FILES = $(wildcard *.tex) $(wildcard ../figures/*.tex) $(wildcard figures/*.tex)

.PHONY: all clean check

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib appendix.tex diagnostic_branches_table.tex algorithm_dagig_grpo.tex ../main_results_table.tex ../node_credit_diagnostic_table.tex ../figures/dagig_method_diagram.tex ../figures/dagig_reward_equations.tex
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

check:
	@test -s $(MAIN).tex
	@test -s references.bib
	@test -s appendix.tex
	@test -s diagnostic_branches_table.tex
	@test -s algorithm_dagig_grpo.tex
	@test -s ../main_results_table.tex
	@test -s ../node_credit_diagnostic_table.tex
	@test -s ../figures/dagig_method_diagram.tex
	@test -s ../figures/dagig_reward_equations.tex
	@! grep -R -n -E '(/root/|outputs/|scripts/)' $(TEX_CHECK_FILES)
	@! grep -R -n -E '\\\\[A-Za-z]' $(TEX_CHECK_FILES)
	@echo "latex source check passed"

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).pdf
