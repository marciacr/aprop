MSIZES="1000"
BSIZES="10"
NUMTHREADS=(2 4 6 8 10)

echo "-----------GCC----------"

for NT in ${NUMTHREADS[@]};do 
    for MS in {1..7}; do
        ./cholesky_gcc 1000 10 $NT "cholesky-gcc.csv"
    done
    
    echo "-----------LVM----------"
    for MS in {1..7}; do
        ./cholesky_lvm 1000 10 $NT "cholesky-lvm.csv"
    done

done