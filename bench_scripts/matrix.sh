#!/bin/bash

cd benchmark
FILES="full_matrix_01 full_matrix_02 sparse_matrix_01 sparse_matrix_02 triple_add_01"
BUILD_DIR="build_matrix/"
OUTPUT_DIR="./output_dir/"
PLOT_CMD="/usr/bin/python ../bench_scripts/plot.py"

[ -f ${BUILD_DIR} ] || mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

[ -f ${OUTPUT_DIR} ] || mkdir ${OUTPUT_DIR}

[ -f CMakeCache.txt ] || cmake ..

for FILE in $FILES
do
  make -j4 $FILE
done
clear

function print_header
{
  local EXECUTABLE=$1

  echo ""
  echo ""
  echo "$EXECUTABLE:"
  echo ""
  printf "%10s %10s %20s %20s %20s %20s %20s %20s\n" \
         "n" "reps" "DEAL_RAW" "DEAL_LOP" "EIGE_RAW" "EIGE_LOP" "BLAZ_RAW" "BLAZ_LOP"
}

function extract_value()
{
  VAL=$(  echo "$1" |\
          grep "$2" |\
          cut -d'|' -f4 |\
          sed 's#s##')
  [ $VAL ] || VAL=0.0
  echo $VAL
}

function step()
{

  
  local EXECUTABLE=$1
  local SIZE=$2
  local REPS=$3
  local OUT_NAME=$4

  RESULT="$(./${EXECUTABLE} ${SIZE} ${REPS} 2>>$OUT_NAME.error)"

  if echo "$RESULT" | grep "DEBUG"; then
    echo "Refused to run in debug mode..."
    exit 1
  fi

  DEAL_LOP=$( extract_value "$RESULT" "dealii_lo"   )
  DEAL_RAW=$( extract_value "$RESULT" "dealii_raw"  )
  EIGE_LOP=$( extract_value "$RESULT" "eigen_lo"    )
  EIGE_RAW=$( extract_value "$RESULT" "eigen_raw"   )
  BLAZ_LOP=$( extract_value "$RESULT" "blaze_lo"    )
  BLAZ_RAW=$( extract_value "$RESULT" "blaze_raw"   )

  DOF=$(  echo "$RESULT" |\
          grep "n:    " |\
          sed 's#n:    ##')

  printf "%10d %10d %20f %20f %20f %20f %20f %20f\n" \
         $DOF $REPS $DEAL_RAW $DEAL_LOP $EIGE_RAW $EIGE_LOP $BLAZ_RAW $BLAZ_LOP | tee -a $OUT_NAME
}

function plot
{
  gnuplot -e "
    set xlabel \"n\" ;
    set ylabel \"Time(s)\" ;
    set logscale xy;
    set key outside;
    set term png ;
    set output \"${OUTPUT_DIR}$1.png\";
    plot  \"${OUTPUT_DIR}$1.data\" using 1:( \$3 > 0 ? \$3 : NaN) with lines title 'DEAL\_RAW',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$4 > 0 ? \$4 : NaN) with lines title 'DEAL\_LOP',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$5 > 0 ? \$5 : NaN) with lines title 'EIGE\_RAW',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$6 > 0 ? \$6 : NaN) with lines title 'EIGE\_LOP',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$7 > 0 ? \$7 : NaN) with lines title 'BLAZ\_RAW',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$8 > 0 ? \$8 : NaN) with lines title 'BLAZ\_LOP'
"
}

print_header full_matrix_01
for i in `seq 1 10`; do
  step full_matrix_01 $((2**$i)) 25000 ${OUTPUT_DIR}/"full_matrix_01.data"
done

print_header full_matrix_02
for i in `seq 1 10`; do
  step full_matrix_02 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_02.data"
done

print_header sparse_matrix_01
for i in `seq 1 8`; do
  step sparse_matrix_01 $i 25000 ${OUTPUT_DIR}/"sparse_matrix_01.data"
done

print_header sparse_matrix_02
for i in `seq 1 8`; do
  step sparse_matrix_02 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_02.data"
done

print_header triple_add_01
for i in `seq 1 10`; do
  step triple_add_01 $((2**$i)) 20000 ${OUTPUT_DIR}/"triple_add_01.data"
done

plot "full_matrix_01"
plot "full_matrix_02"
plot "sparse_matrix_01"
plot "sparse_matrix_02"
plot "triple_add_01"
