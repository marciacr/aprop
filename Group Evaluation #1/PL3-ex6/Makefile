PROGRAM=cholesky

TARGETS=$(PROGRAM)

CC = gcc
CCG = clang

CFLAGS = -fopenmp
# -I$(MKL_INC_DIR)
# -L$(MKL_LIB_DIR) 

LIBS  = -lmkl_sequential -lmkl_core -lmkl_rt -lpthread -lm
EXTRA = -std=c99 -O3 -Wall -Wno-unused 
INCS  = 

all: $(PROGRAM)_gcc $(PROGRAM)_lvm

$(PROGRAM)_gcc: $(PROGRAM).c
	$(CC) $(CFLAGS) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

$(PROGRAM)_lvm: $(PROGRAM).c
	$(CCG) $(CFLAGS) $(EXTRA) $(INCS) -o $@ $< $(LIBS)

clean:
	rm -f $(CC)_* *.o *~ $(TARGETS) 





