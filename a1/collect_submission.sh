conda config --add channels conda-forge
conda install zip
rm -f 2022_50000_coding.zip #change here to your student id
zip -r 2022_50000_coding.zip ./skipgram/*.py ./skipgram/*.png ./skipgram/saved_params_40000.npy saved_state_40000.pickle ./fasttext/*.py ./fasttext/utils/*.py ./fasttext/*.png ./fasttext/saved_params_20000_inside.npy ./fasttext/saved_params_20000_outside.npy ./fasttext/saved_params_20000_outside.pickle ./tfidf/*.ipynb ./sentence/*.ipynb
#change above to your student id