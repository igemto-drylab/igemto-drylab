The files are npy files that are the unirep vector representations (3*1900), consisting of avg_hidden, final_hidden, final_cell. The filenames are the proteinIDs (UniProtKB ID) 

link to google drive: https://drive.google.com/drive/folders/1jCpfrNbQasSjeHxNEaMxZtv0VBX690ky?usp=sharing

The .gitignore file will not add the unirep vecs to a commit if the folder is in the datasets folder.

Use meltome_full for any experiments, which consists of the xspecies (no obsolete) and humans concatenated, along with 2 errors resolved.

the errors were proteins that were in both datasets, so the melting points have been averaged out and resolved in the full dataset.