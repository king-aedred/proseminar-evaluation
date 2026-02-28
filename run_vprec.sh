set -e

# config
SIZES=(500)
RUNS_PER_CONFIG=10
OUTPUT_DIR="vprec_results"
# check for docker
if ! command -v docker &> /dev/null; then
    echo "docker not found!"
    exit 1
fi

# output dir
mkdir -p $OUTPUT_DIR
rm -f $OUTPUT_DIR/*.txt

echo "Compiling with verificarlo-c++ (inside Docker) - VPREC Backend"
docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
    "export VFC_BACKENDS='libinterflop_vprec.so' && \
    verificarlo-c++ -std=c++11 -O2 -o example-marski-vprec example-marski.c" > /dev/null 2>&1
echo "Compilation successful"
echo

# vprec config - test different precisions
for size in "${SIZES[@]}"; do
    echo "Matrix ${size}×${size} - VPREC Backend:"
    
    # Test FP64 (IEEE binary64: mantissa 53, exponent 11)
    echo "  Testing FP64 (full precision):"
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        printf "    Run %2d/%d..." $run $RUNS_PER_CONFIG
        
        docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
            "export VFC_BACKENDS='libinterflop_vprec.so' && \
            export VFC_BACKENDS_LOGGER='False' && \
            export INTERFLOP_VPREC_RANGE_BINARY64=11 && \
            export INTERFLOP_VPREC_PRECISION_BINARY64=53 && \
            export INTERFLOP_VPREC_RANGE_BINARY32=8 && \
            export INTERFLOP_VPREC_PRECISION_BINARY32=24 && \
            ./example-marski-vprec $size $size" > \
            "$OUTPUT_DIR/size_${size}_fp64_run_${run}.txt" 2>&1
        
        echo " Done"
    done
    
    # Test FP32 (IEEE binary32: mantissa 24, exponent 8)
    echo "  Testing FP32 (reduced precision):"
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        printf "    Run %2d/%d..." $run $RUNS_PER_CONFIG
        
        docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
            "export VFC_BACKENDS='libinterflop_vprec.so' && \
            export VFC_BACKENDS_LOGGER='False' && \
            export INTERFLOP_VPREC_RANGE_BINARY64=8 && \
            export INTERFLOP_VPREC_PRECISION_BINARY64=24 && \
            export INTERFLOP_VPREC_RANGE_BINARY32=8 && \
            export INTERFLOP_VPREC_PRECISION_BINARY32=24 && \
            ./example-marski-vprec $size $size" > \
            "$OUTPUT_DIR/size_${size}_fp32_run_${run}.txt" 2>&1
        
        echo " Done"
    done

    # Test bfloat16 (mantissa 8, exponent 8)
    echo "  Testing bfloat16:"
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        printf "    Run %2d/%d..." $run $RUNS_PER_CONFIG

        docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
            "export VFC_BACKENDS='libinterflop_vprec.so --preset=bfloat16' && \
            export VFC_BACKENDS_LOGGER='False' && \
            ./example-marski-vprec $size $size" > \
            "$OUTPUT_DIR/size_${size}_bfloat16_run_${run}.txt" 2>&1

        echo " Done"
    done

    # Test FP16 / binary16 (mantissa 11, exponent 5)
    echo "  Testing FP16 (binary16):"
    for run in $(seq 1 $RUNS_PER_CONFIG); do
        printf "    Run %2d/%d..." $run $RUNS_PER_CONFIG

        docker run --rm -v $(pwd):/workdir -w /workdir verificarlo/verificarlo bash -c \
            "export VFC_BACKENDS='libinterflop_vprec.so --preset=binary16' && \
            export VFC_BACKENDS_LOGGER='False' && \
            ./example-marski-vprec $size $size" > \
            "$OUTPUT_DIR/size_${size}_fp16_run_${run}.txt" 2>&1

        echo " Done"
    done
    echo
done

echo "VPREC experiments completed!"
echo "Results saved to: $OUTPUT_DIR/"