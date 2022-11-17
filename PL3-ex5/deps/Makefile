CC = gcc
CC2 = clang

CFLAGS = -fopenmp -lm
EXTRA = -Wall -Wno-unused 

all: deps deps_clang

deps: deps.c
	$(CC) $(CFLAGS) $(EXTRA) -o $@ $<

deps_clang: deps.c
	$(CC2) $(CFLAGS) $(EXTRA) -o $@ $<

clean:
	rm -f $(CC)_* *.o *~ deps deps_clang file.csv

run_gcc: deps
	./deps

run_clang: deps_clang
	./deps_clang