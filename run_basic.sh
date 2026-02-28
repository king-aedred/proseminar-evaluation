set -e

# config
SIZES=(500)
RUNS=10
OUTPUT_DIR="basic_results"

# output dir
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.txt


echo "Compiling example-marski.c"
g++ -std=c++11 -O2 -o example_marski example-marski.c
echo "Compilation successfull"
echo

for size in "${SIZES[@]}"; do
    echo "Matrix-Size: ${size}×${size}"
    
    for run in $(seq 1 $RUNS); do
        printf "  Run %d/%d..." $run $RUNS
        
        ./example_marski $size > "$OUTPUT_DIR/size_${size}_run_${run}.txt" 2>&1
        
        echo " Completed"
    done
    echo
done