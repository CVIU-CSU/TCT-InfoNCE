#!/usr/bin/env bash

FEAT_NAME=biomedclip-adapter
DATASET=fnac
INPUT_DIM=512

# train TransMIL
cd mil-methods/scripts
bash transmil.sh ${FEAT_NAME} ${DATASET} ${INPUT_DIM}
# train MHIM-MIL(TransMIL)
bash mhim\(transmil\).sh ${FEAT_NAME} ${DATASET} ${INPUT_DIM}

# train ABMIL
# bash abmil.sh ${FEAT_NAME} ${DATASET} ${INPUT_DIM}
# train MHIM-MIL(ABMIL)
# bash mhim\(abmil\).sh ${FEAT_NAME} ${DATASET} ${INPUT_DIM}
