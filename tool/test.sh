#! /bin/sh

go run rbm.go --input ../testdata/a1a --model a1a.mlf --binary_hidden=true --hidden 12 --max_iter 1000 --cd 1
go run generate_rbm_features.go --input ../testdata/a1a --use_binary=false --cd 1 --model a1a.mlf > a1a_rbm
go run generate_rbm_features.go --input ../testdata/a1a.t --use_binary=false --cd 1 --model a1a.mlf > a1a_rbm.t

go run classifier.go --input a1a_rbm --test a1a_rbm.t

# baseline error = 15.74 %
