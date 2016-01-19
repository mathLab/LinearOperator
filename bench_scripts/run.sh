#!/bin/bash

BUILD_DIR="build/"
OUTPUT_DIR="./output_dir/"
PLOT_CMD="/usr/bin/python ../bench_scripts/plot.py"

[ -f ${BUILD_DIR} ] || mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

[ -f ${OUTPUT_DIR} ] || mkdir ${OUTPUT_DIR}

[ -f CMakeCache.txt ] || cmake ..

make -j4
clear

function print_header
{
  local EXECUTABLE=$1

  echo ""
  echo ""
  echo "$EXECUTABLE:"
  echo ""
  printf "%10s\t %10s\t %20s\t %20s\t %20s\t %20s\n" \
         "n" "Repetitions" "Raw" "LO" "EigenSimple" "EigenSmart"
}

function step
{
  local EXECUTABLE=$1
  local SIZE=$2
  local REPS=$3
  local OUTPUT=$4

  RESULT="$(./${EXECUTABLE} ${SIZE} ${REPS})"

  if echo "$RESULT" | grep "DEBUG"; then
    echo "Refused to run in debug mode..."
    exit 1
  fi

  DOF=$(echo "$RESULT" | grep "n:    " | sed 's#n:    ##')
  LO=$(echo "$RESULT" | grep "linear_operator" | cut -d'|' -f4 | sed 's#s##')
  ST=$(echo "$RESULT" | grep "raw" | cut -d'|' -f4 | sed 's#s##')
  ESI=$(echo "$RESULT" | grep "eigen_simple" | cut -d'|' -f4 | sed 's#s##')
  ESM=$(echo "$RESULT" | grep "eigen_smart" | cut -d'|' -f4 | sed 's#s##')

  [ $ST ] || ST=0.0
  [ $LO ] || LO=0.0
  [ $ESI ] || ESI=0.0
  [ $ESM ] || ESM=0.0
    
  printf "%10d\t %10d\t %20f\t %20f\t %20f\t %20f \n" \
         $DOF $REPS $ST $LO $ESI $ESM | tee -a $OUTPUT
}

function plot
{
  gnuplot -e "
    set xlabel \"n\" ;
    set ylabel \"Time(s)\" ;
    set term png ;
    set output \"${OUTPUT_DIR}$1.png\";
    plot  \"${OUTPUT_DIR}$1.data\" using 1:( \$3 > 0 ? \$3 : NaN) with lines title 'raw',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$4 > 0 ? \$4 : NaN) with lines title 'lo',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$5 > 0 ? \$5 : NaN) with lines title 'ESi',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$6 > 0 ? \$6 : NaN) with lines title 'ESm'
"
  # ${PLOT_CMD} ${OUTPUT_DIR}/"$1.data" "$1"
}

print_header full_matrix_01
for i in `seq 1 9`; do
  step full_matrix_01 $((2**$i)) 25000 ${OUTPUT_DIR}/"full_matrix_01.data"
done
plot "full_matrix_01"

print_header full_matrix_02
for i in `seq 1 10`; do
  step full_matrix_02 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_02.data"
done
plot "full_matrix_02"

print_header sparse_matrix_01
for i in `seq 1 8`; do
  step sparse_matrix_01 $i 25000 ${OUTPUT_DIR}/"sparse_matrix_01.data"
done
plot "sparse_matrix_01"

print_header sparse_matrix_02
for i in `seq 1 8`; do
  step sparse_matrix_02 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_02.data"
done
plot "sparse_matrix_02"

# print_header sparse_matrix_inefficient
# for i in `seq 1 10`; do
#   step sparse_matrix_inefficient $i 300 ${OUTPUT_DIR}/"sparse_matrix_inefficient.data"
# done
# 
# print_header sparse_matrix_residual
# for i in `seq 1 10`; do
#   step sparse_matrix_residual $i 500 ${OUTPUT_DIR}/"sparse_matrix_residual.data"
# done

print_header triple_add_01
for i in `seq 1 10`; do
  step triple_add_01 $((2**$i)) 20000 ${OUTPUT_DIR}/"triple_add_01.data"
done
plot "triple_add_01"

print_header triple_add_02
for i in `seq 1 10`; do
  step triple_add_02 $((2**$i)) 20000 ${OUTPUT_DIR}/"triple_add_02.data"
done
plot "triple_add_02"

print_header eigen_1
n=0
for i in `seq 1 9`; do
  let n=${n}+30
  step eigen ${n} 20000 ${OUTPUT_DIR}/"eigen_01.data"
done
plot "eigen_01"

print_header eigen_2
n=300
rep=32768
for i in `seq 1 15`; do
  step eigen $n $rep ${OUTPUT_DIR}/"eigen_02.data"
  let n=${n}*2
  let rep=$rep/2
done
plot "eigen_02"
