#!/bin/bash

cd benchmark
FILES=" full_matrix_01 full_matrix_02 full_matrix_03 full_matrix_04 \
        sparse_matrix_01 sparse_matrix_02 sparse_matrix_03 sparse_matrix_04"
BUILD_DIR="build_matrix_test/"
OUTPUT_DIR="./output_dir/"
PLOT_CMD="/usr/bin/python ../bench_scripts/plot.py"

[ ! -f ${BUILD_DIR} ] && mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

[ ! -f ${OUTPUT_DIR} ] && mkdir ${OUTPUT_DIR}

[ ! -f CMakeCache.txt ] && cmake ..

for FILE in $FILES
do
  make -j4 $FILE
done
clear

function print_header
{
  local EXECUTABLE=$1
  local OUT_NAME=$2
  
  echo ""
  echo ""
  echo "$EXECUTABLE:"
  echo ""
  printf "%10s %10s %20s %20s %20s %20s %20s %20s %20s %20s \n" \
         "n" "reps" "DEAL_RAW" "DEAL_SMA" "DEAL_LOP" "EIGE_RAW" "EIGE_SMA" "EIGE_LOP" "BLAZ_RAW" "BLAZ_LOP"
  echo "n reps DEAL_RAW DEAL_SMA DEAL_LOP EIGE_RAW EIGE_SMA EIGE_LOP BLAZ_RAW BLAZ_LOP" >> $OUTNAME
}

function extract_value()
{
  VAL=$(  echo "$1" |\
          grep "$2" |\
          cut -d'|' -f4 |\
          sed 's#s##'| xargs)
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
  DEAL_SMA=$( extract_value "$RESULT" "dealii_smart"  )
  EIGE_LOP=$( extract_value "$RESULT" "eigen_lo"    )
  EIGE_RAW=$( extract_value "$RESULT" "eigen_raw"   )
  EIGE_RAWW=$( extract_value "$RESULT" "eigen_aaa"   )
  EIGE_SMA=$( extract_value "$RESULT" "eigen_smart" )
  BLAZ_LOP=$( extract_value "$RESULT" "blaze_lo"    )
  BLAZ_RAW=$( extract_value "$RESULT" "blaze_raw"   )

  DOF=$(  echo "$RESULT" |\
          grep "n:    " |\
          sed 's#n:    ##')

  EIGE_RAWW=$( echo "$EIGE_RAWW*100" | sed 's/[eE]+\{0,1\}/*10^/g' | bc -l )
  EIGE_RAW=$( echo "$EIGE_RAW+$EIGE_RAWW*" | sed 's/[eE]+\{0,1\}/*10^/g' | bc -l )
  
  printf "%10d %10d %20f %20f %20f %20f %20f %20f %20f %20f\n" \
         $DOF $REPS $DEAL_RAW $DEAL_SMA $DEAL_LOP $EIGE_RAW $EIGE_SMA $EIGE_LOP $BLAZ_RAW $BLAZ_LOP
  
  echo $DOF" "$REPS" "$DEAL_RAW" "$DEAL_SMA" "$DEAL_LOP" "$EIGE_RAW" "$EIGE_SMA" "$EIGE_LOP" "$BLAZ_RAW" "$BLAZ_LOP >> $OUT_NAME
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
          \"${OUTPUT_DIR}$1.data\" using 1:( \$4 > 0 ? \$4 : NaN) with lines title 'DEAL\_SMA',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$5 > 0 ? \$5 : NaN) with lines title 'DEAL\_LOP',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$6 > 0 ? \$6 : NaN) with lines title 'EIGE\_RAW',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$7 > 0 ? \$7 : NaN) with lines title 'EIGE\_SMA',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$8 > 0 ? \$8 : NaN) with lines title 'EIGE\_LOP',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$9 > 0 ? \$9 : NaN) with lines title 'BLAZ\_RAW',
          \"${OUTPUT_DIR}$1.data\" using 1:( \$10 > 0 ? \$10 : NaN) with lines title 'BLAZ\_LOP'
"
}

# Warm up
for FILE in $FILES
do
  echo "Warm up using "$FILE
  ./$FILE 4 4 2>&1 >/dev/null 
done

# # Full Matrices
# 
# print_header full_matrix_01 ${OUTPUT_DIR}/"full_matrix_01.data"
# for i in `seq 1 10`; do
#   step full_matrix_01 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_01.data"
# done
# 
print_header full_matrix_02 ${OUTPUT_DIR}/"full_matrix_02.data"
for i in `seq 1 10`; do
  step full_matrix_02 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_02.data"
done
# 
# print_header full_matrix_03 ${OUTPUT_DIR}/"full_matrix_03.data"
# for i in `seq 1 10`; do
#   step full_matrix_03 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_03.data"
# done
# 
# print_header full_matrix_04 ${OUTPUT_DIR}/"full_matrix_04.data"
# for i in `seq 1 10`; do
#   step full_matrix_04 $((2**$i)) 10000 ${OUTPUT_DIR}/"full_matrix_04.data"
# done
# 
# # Sparse Matrices
# 
# print_header sparse_matrix_01 ${OUTPUT_DIR}/"sparse_matrix_01.data"
# for i in `seq 1 8`; do
#   step sparse_matrix_01 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_01.data"
# done

print_header sparse_matrix_02 ${OUTPUT_DIR}/"sparse_matrix_02.data"
for i in `seq 1 8`; do
  step sparse_matrix_02 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_02.data"
done

# print_header sparse_matrix_03 ${OUTPUT_DIR}/"sparse_matrix_03.data"
# for i in `seq 1 8`; do
#   step sparse_matrix_03 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_03.data"
# done
# 
# print_header sparse_matrix_04 ${OUTPUT_DIR}/"sparse_matrix_04.data"
# for i in `seq 1 8`; do
#   step sparse_matrix_04 $i 10000 ${OUTPUT_DIR}/"sparse_matrix_04.data"
# done

plot "full_matrix_01"
plot "full_matrix_02"
plot "full_matrix_03"
plot "full_matrix_04"
plot "sparse_matrix_01"
plot "sparse_matrix_02"
plot "sparse_matrix_03"
plot "sparse_matrix_04"