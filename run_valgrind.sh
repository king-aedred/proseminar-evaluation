set -e

# config
SIZES=(1000)
RUNS=10
OUTPUT_DIR="results_valgrind"
VALGRIND_TOOL="memcheck"
VALGRIND_OPTS="--leak-check=full --error-exitcode=1"
MODES=(fp64 fp32 ir)

# output dir
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.txt


echo "Compiling example-marski.c"
g++ -std=c++11 -O2 -o example_marski_valgrind example-marski.c
echo "Compilation successfull"
echo

for size in "${SIZES[@]}"; do
    echo "Matrix-Size: ${size}×${size}"

    for mode in "${MODES[@]}"; do
        echo " Mode: ${mode}"
        for run in $(seq 1 $RUNS); do
            printf "  Run %d/%d..." $run $RUNS

            valgrind --tool="$VALGRIND_TOOL" $VALGRIND_OPTS \
                --log-file="$OUTPUT_DIR/size_${size}_${mode}_run_${run}_valgrind.log" \
                ./example_marski_valgrind $size --mode=$mode > \
                "$OUTPUT_DIR/size_${size}_${mode}_run_${run}.txt" 2>&1

            echo " Completed"
        done
    done
    echo
done