python3 main.py -o batchsize -d "$1" >> tests/results/"$1"/"$1"_batchsize.out &
python3 main.py -o nlayers -d "$1" >> tests/results/"$1"/"$1"_nlayers.out &
python3 main.py -o nneurons -d "$1" >> tests/results/"$1"/"$1"_nneurons.out &
python3 main.py -o regularization -d "$1" >> tests/results/"$1"/"$1"_regularization.out &
python3 main.py -o alpha -d "$1" >> tests/results/"$1"/"$1"_alpha.out &
python3 main.py -o beta -d "$1" >> tests/results/"$1"/"$1"_beta.out &
