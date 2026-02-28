set -e

# config
SIZES=(500)
RUNS=10
OUTPUT_DIR="classic_results"

# output dir
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.txt

echo "Compiling gaussian_solve.c"
g++ -std=c++11 -O2 -o gaussian_solve gaussian_solve.c
echo "Compilation successfull"
echo

for size in "${SIZES[@]}"; do
    echo "Matrix-Size: ${size}×${size}"
    
    for run in $(seq 1 $RUNS); do
        printf "  Run %d/%d..." $run $RUNS
        
        ./gaussian_solve $size > "$OUTPUT_DIR/size_${size}_classic_run_${run}.txt" 2>&1
        
        echo " Completed"
    done
    echo
done
