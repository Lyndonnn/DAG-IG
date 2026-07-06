# Venue Template Parts

These files are generated from `latex/main.tex` and `latex/appendix.tex` to make venue-template conversion less error-prone.

Use these when moving the paper into a conference/journal template:

- `title.tex`: title text only.
- `abstract.tex`: abstract body only.
- `abstract_environment.tex`: abstract wrapped in a generic `abstract` environment.
- `body_sections.tex`: main sections from Introduction through Conclusion, without documentclass, preamble, bibliography, or appendix marker.
- `appendix_sections.tex`: appendix sections.

The body uses bundle-root relative paths such as `figures/...` and `tables/...`. Put `figures/`, `tables/`, `algorithm_dagig_grpo.tex`, `diagnostic_branches_table.tex`, and `references.bib` at the venue project root, or adjust paths once in the venue template.

Do not change experimental numbers in these snippets. If a number must change, regenerate the audited tables and rerun the paper asset audit.
