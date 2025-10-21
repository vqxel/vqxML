.PHONY: clean build run all

all: run

build:
	g++ -std=c++23 -Iinclude src/*.cpp -o build/main.out

run: build
	./build/main.out ../data/input_full.txt ../data/output.txt

clean:
	rm -f build/* 
