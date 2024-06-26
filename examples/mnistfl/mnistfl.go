package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"github.com/patrikeh/go-deep"
	"github.com/patrikeh/go-deep/training"
	"io"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"
)

/*
mnist classifier
mnist is a set of hand-written digits 0-9
the dataset in a sane format (as used here) can be found at:
https://pjreddie.com/projects/mnist-in-csv/
*/
func main() {
	rand.Seed(time.Now().UnixNano())

	train, err := load("./mnist_train.csv")
	if err != nil {
		panic(err)
	}
	test, err := load("./mnist_test.csv")
	if err != nil {
		panic(err)
	}

	for i := range train {
		for j := range train[i].Input {
			train[i].Input[j] = train[i].Input[j] / 255
		}
	}
	for i := range test {
		for j := range test[i].Input {
			test[i].Input[j] = test[i].Input[j] / 255
		}
	}
	test.Shuffle()
	train.Shuffle()

	N := 10
	batchSize := 64
	parallelism := 8
	epoch := 20
	batchNum := 16
	trainsets := train.SplitN(N)

	neural := deep.NewNeural(&deep.Config{
		Inputs:     len(train[0].Input),
		Layout:     []int{50, 10},
		Activation: deep.ActivationReLU,
		Mode:       deep.ModeMultiClass,
		Weight:     deep.NewNormal(0.6, 0.1), // slight positive bias helps ReLU
		Bias:       true,
	})

	localNeurals := make([]*deep.Neural, N)

	//trainer := training.NewTrainer(training.NewSGD(0.01, 0.5, 1e-6, true), 1)
	trainers := make([]*training.BatchTrainer, N)
	for i := range trainers {
		trainers[i] = training.NewBatchTrainer(training.NewAdam(0.1, 0.9, 0.999, 1e-8), 1, batchSize, parallelism)
	}

	ts := time.Now()
	mu := sync.Mutex{}
	//fmt.Printf("training: %d, val: %d, test: %d\n", len(train), len(test), len(test))
	initDump := neural.Dump()
	for e := 1; e <= epoch; e++ {
		trainTmps := make([]training.Examples, N)
		batches := make([][]training.Examples, N)
		for k := 0; k < N; k++ {
			trainTmps[k] = make(training.Examples, len(trainsets[k]))
			copy(trainTmps[k], trainsets[k])
			trainTmps[k].Shuffle()
			batches[k] = trainTmps[k].SplitSize(batchSize)
		}

		for i := 0; i < batchNum; i++ {
			wg := new(sync.WaitGroup)
			wg.Add(N)
			deep.ZeroWeights(initDump)
			for k := 0; k < N; k++ {
				go func(k int, i int) {
					defer wg.Done()
					localNeurals[k] = deep.FromDump(neural.Dump())
					dump := trainers[k].LocalTrain(localNeurals[k], e, batches[k][i])
					mu.Lock()
					initDump = deep.AddWeights(initDump, dump)
					mu.Unlock()
				}(k, i)
			}
			wg.Wait()
			deep.FractionWeights(initDump, float64(N))
			neural = deep.FromDump(initDump)
		}
		PrintProgress(neural, test, time.Since(ts), e)
	}
}

func load(path string) (training.Examples, error) {
	f, err := os.Open(path)
	defer f.Close()
	if err != nil {
		return nil, err
	}
	r := csv.NewReader(bufio.NewReader(f))

	var examples training.Examples
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		examples = append(examples, toExample(record))
	}

	return examples, nil
}

func toExample(in []string) training.Example {
	res, err := strconv.ParseFloat(in[0], 64)
	if err != nil {
		panic(err)
	}
	resEncoded := onehot(10, res)
	var features []float64
	for i := 1; i < len(in); i++ {
		res, err := strconv.ParseFloat(in[i], 64)
		if err != nil {
			panic(err)
		}
		features = append(features, res)
	}

	return training.Example{
		Response: resEncoded,
		Input:    features,
	}
}

func onehot(classes int, val float64) []float64 {
	res := make([]float64, classes)
	res[int(val)] = 1
	return res
}

func crossValidate(n *deep.Neural, validation training.Examples) float64 {
	predictions, responses := make([][]float64, len(validation)), make([][]float64, len(validation))
	for i := 0; i < len(validation); i++ {
		predictions[i] = n.Predict(validation[i].Input)
		responses[i] = validation[i].Response
	}

	return deep.GetLoss(n.Config.Loss).F(predictions, responses)
}

func formatAccuracy(n *deep.Neural, validation training.Examples) string {
	if n.Config.Mode == deep.ModeMultiClass {
		return fmt.Sprintf("%.2f\t", accuracy(n, validation))
	}
	return ""
}

func accuracy(n *deep.Neural, validation training.Examples) float64 {
	correct := 0
	for _, e := range validation {
		est := n.Predict(e.Input)
		if deep.ArgMax(e.Response) == deep.ArgMax(est) {
			correct++
		}
	}
	return float64(correct) / float64(len(validation))
}

// PrintProgress prints the current state of training
func PrintProgress(n *deep.Neural, validation training.Examples, elapsed time.Duration, iteration int) {
	fmt.Printf("%d\t%s\t%.4f\t%s\n",
		iteration,
		elapsed.String(),
		crossValidate(n, validation),
		formatAccuracy(n, validation))
}
