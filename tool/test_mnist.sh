#! /bin/sh

go run rbm.go --input mnist.data --model mnist.mlf --binary_hidden=true --hidden 10 --max_iter 1000 --cd 1
go run generate_rbm_features.go --input mnist.data --use_binary=false --cd 1 --model mnist.mlf > mnist_rbm
go run generate_rbm_features.go --input mnist_test.data --use_binary=false --cd 1 --model mnist.mlf > mnist_rbm.t

go run classifier.go --input mnist_rbm --test mnist_rbm.t --max_iter 100

# baseline error = 7.58 %
