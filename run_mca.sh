set -e

# config
SIZES=(500)
RUNS_PER_CONFIG=10
OUTPUT_DIR="results_mca"
# check for docker
if ! command -v docker &> /dev/null; then
    echo "docker not found!"
    exit 1
fi

# output dir
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.txt


echo "Compiling with verificarlo-c++ (inside Docker)"
docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
    "export VFC_BACKENDS='libinterflop_mca.so' && \
    verificarlo-c++ -std=c++11 -O2 -o gaussian_solve_fp32_64-vfc gaussian_solve_fp32_64.c" > /dev/null 2>&1
echo "Compilation successful"
echo

# mca config
for size in "${SIZES[@]}"; do
    echo "Matrix ${size}×${size}:"
    
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        printf "  Run %2d/%d..." $run $RUNS_PER_CONFIG
        
        docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
            "export VFC_BACKENDS='libinterflop_mca.so' && \
            export VFC_BACKENDS_LOGGER='False' && \
            export INTERFLOP_MCA_MODE='mca' && \
            export INTERFLOP_MCA_PRECISION_BINARY64=53 && \
            export INTERFLOP_MCA_PRECISION_BINARY32=24 && \
            ./gaussian_solve_fp32_64-vfc $size $size" > \
            "$OUTPUT_DIR/size_${size}_run_${run}.txt" 2>&1
        
        echo " Done"
    done
    echo
done