SINGLE_SPEAKER_DIR=../VCTK-Corpus/wav48/p312

# patch generation
SCA=4
SCB=16
DIM=8192
STR=2048
SR=16000

# ----------------------------------------------------------------------------

patches:
	make vctk-speaker1-train.$(SCA).$(SCB).$(SR).$(DIM).$(STR).h5 vctk-speaker1-val.$(SCA).$(SCB).$(SR).$(DIM).$(STR).h5

# ----------------------------------------------------------------------------
# create dataset for one speaker

vctk-speaker1-train.$(SCA).$(SCB).$(SR).$(DIM).$(STR).h5: speaker1-train-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale_high $(SCA) \
		--scale_low $(SCB)\
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) 
	mv $@.tmp $@

vctk-speaker1-val.$(SCA).$(SCB).$(SR).$(DIM).$(STR).h5: speaker1-val-files.txt
	python ../prep_vctk.py \
		--file-list $< \
		--in-dir $(SINGLE_SPEAKER_DIR) \
		--out $@.tmp \
		--scale_high $(SCA) \
		--scale_low $(SCB)\
		--sr $(SR) \
		--dimension $(DIM) \
		--stride $(STR) 
	mv $@.tmp $@
