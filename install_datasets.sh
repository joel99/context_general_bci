# to be called from project root
# ln -s ~/share/ndt2/ data # your share dir here, i.e. call from ./data
cd data
# RTT
# zenodo_get 3854034

# NLB
# dandi download DANDI:000128/0.220113.0400
# dandi download https://dandiarchive.org/dandiset/000129/draft
# dandi download DANDI:000138/0.220113.0407
dandi download DANDI:000139/0.220113.0408
dandi download DANDI:000140/0.220113.0408

mkdir -p runs # for wandb