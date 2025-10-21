.PHONY: clean build run all

ARGS =

all: run

build:
	g++ -std=c++23 main.cpp -o main.out

run: build
	./main.out ../data/input_full.txt ../data/output.txt $(ARGS)

clean:
	rm -f *.out
