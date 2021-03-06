SPEAKER_DIR=../VCTK-Corpus/wav48

# size of corpus
N_TRAIN_SPEAKERS=100
N_VAL_SPEAKERS=9

# patch generation
SCH=4
SCL=16
DIM=32768
STR=8192
SR=16000

# fraction of patches to keep
SAM=1

# ----------------------------------------------------------------------------

all:
	make vctk-multispeaker-interp-train.$(SCH).$(SCL).$(SR).$(DIM).$(STR).$(SAM).h5 vctk-multispeaker-interp-val.$(SCH).$(SCL).$(SR).$(DIM).$(STR).$(SAM).h5

# ----------------------------------------------------------------------------
# create dataset for multiple speakers

vctk-multispeaker-interp-train.$(SCH).$(SCL).$(SR).$(DIM).$(STR).$(SAM).h5: train-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir `pwd` \
		--out $@.tmp \
		--scale_high $(SCH) \
		--scale_low $(SCL) \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) \
		--sam $(SAM)
	mv $@.tmp $@

vctk-multispeaker-interp-val.$(SCH).$(SCL).$(SR).$(DIM).$(STR).$(SAM).h5: val-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir `pwd` \
		--out $@.tmp \
		--scale_high $(SCH) \
		--scale_low $(SCL) \
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) \
		--sam $(SAM)
	mv $@.tmp $@

train-files.txt: train-speakers.txt
	find ../VCTK-Corpus/ | grep -P '\.wav' | grep -f $< > $@

val-files.txt: val-speakers.txt
	find ../VCTK-Corpus/ | grep -P '\.wav' | grep -f $< > $@	

train-speakers.txt:
	ls $(SPEAKER_DIR) | head -n $(N_TRAIN_SPEAKERS) > $@

val-speakers.txt:
	ls $(SPEAKER_DIR) | tail -n $(N_VAL_SPEAKERS) > $@
