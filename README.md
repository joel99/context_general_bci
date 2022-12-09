# context_general_bci
Towards large neural data models.

The entire project is structured with the assumption that Transformers will be the backbone.
This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.
- we can potentially play with task tokens and task-specific heads for joint training, but there won’t be any mixed-task batches; only text 2 text supports that.
  - [ ]  Check out if any unified vision works manage mixed-task batches?


- For example, to get online decoding; we would need a fine-tuned decoder interface that could run causally.


## Admin
- note that we installed NLB tools via pip and that this constrained our pandas to be <1.34 (whereas it was originally ~1.5). A bit annoying - we should go back and re-add NLB tools dependency at some point.
