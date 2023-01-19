# context_general_bci
Towards large neural data models.

The entire project is structured with the assumption that Transformers will be the backbone.
This is a BERT era effort to get large base models on which various BCI tasks will be solved. Being BERT era, the default task strategy will be fine-tuning.
- we can potentially play with task tokens and task-specific heads for joint training, but there won’t be any mixed-task batches; only text 2 text supports that.
  - [ ]  Check out if any unified vision works manage mixed-task batches?


- For example, to get online decoding; we would need a fine-tuned decoder interface that could run causally.

## Codebase design
This codebase mixes many different heterogenuous datasets in an attempt to make more general neural data models. To keep track of the mess of interfaces, design tends to be strongly typed.

## Admin
- note that we installed NLB tools via pip and that this constrained our pandas to be <1.34 (whereas it was originally ~1.5). A bit annoying - we should go back and re-add NLB tools dependency at some point.


## Pitt notes
- Sessions and notes (with dates) (CRS02bHome sessions)
- types
  - fbc-stitch: using and updating decoder from previous session
- All tasks are 2D center out long-term study (Angelica)


9/26/18:
- 329: obs
- 330: ortho
- 332: fbc-src
9/28/18:
- 333: fbc-stitch
10/01/18:
- 335: fbc-stitch
- 336: obs
- 337: ortho
10/02/18:
- 338: fbc-stitch
- 339: obs
- 340: ortho
- 341: fbc
10/08/18:
- 343: fbc-stitch
10/10/18:
- 344: fbc-stitch
- 345: obs
- 346: ortho/fbc
10/19/18:
- 355: fbc-stitch
10/22/18:
- 358: fbc-stitch
10/23/18:
- 359: fbc-stitch
- 360: obs
- 361: ortho/fbc
10/29/18:
- 363: fbc-stitch
10/30/18:
- 364: fbc-stitch
- 365: obs
- 366: ortho/fbc
11/13/18:
- 370: fbc-stitch
- 371: obs
- 372: ortho/fbc
11/19:
- 386: fbc-stitch
- 387: obs
- 388: ortho
11/27:
- 401: fbc-stitch
- 402: obs
- 403: ortho/fbc
12/05:
- 411: fbc-stitch
- 412/413: ortho/fbc
12/10:
- 421: fbc-stitch
- 422: obs
- 423: ortho/fbc
- 424: obs
- 425: ortho/fbc
12/17:
- 436: fbc-stitch
- 437: obs
- 438: ortho/fbc
1/7/19:
- 444: fbc-stitch
- 445: obs
- 446: ortho/fbc
IDK - the majority of Angelica's data _is_ FBC, not observation.
