MSIZES="1000"
BSIZES="10"

echo "-----------GCC----------"

for MS in {1..22}; do
    ./cholesky_gcc 1000 10 4 "cholesky-gcc.csv"
done

echo "-----------LVM----------"
for MS in {1..22}; do
    ./cholesky_lvm 1000 10 4 "cholesky-lvm.csv"
done