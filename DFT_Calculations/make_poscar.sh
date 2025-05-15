for i in ARUN.POCC_*; do pushd $i; ../../../../AFLOW-src/src/aflow --xposcar aflow.in > POSCAR; popd; done
