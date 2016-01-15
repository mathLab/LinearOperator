#!/bin/bash

WORK_DIR="./build/"
BENCH_DIR="./bench_scripts/"
PYTHON="/usr/bin/python"

FILE_00="output_text_00.out"
FILE_01="output_text_01.out"

[ -d ${WORK_DIR} ] || mkdir ${WORK_DIR}
cd ${WORK_DIR}

cmake ..
make -j4 
clear

function run {
  SIZE=$1
  REPS=$2

  FILE=$3
  
  RESULT="$(./matrix ${SIZE} ${REPS})"

  DOF=$(  echo "$RESULT" | \
          grep "Number of degrees of freedom:" | \
          sed 's#Number of degrees of freedom:##' )

  OPERATOR_LO=$(  echo "$RESULT" | \
                  grep "operator - LO" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')
  OPERATOR_ST=$(  echo "$RESULT" | \
                  grep "operator - STD" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')
  SIMPVMUL_LO=$(  echo "$RESULT" | \
                  grep "simple vmult - LO" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')
  SIMPVMUL_ST=$(  echo "$RESULT" | \
                  grep "simple vmult - STD" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')

  printf "%9d, %9d, %19f, %19f, %19f, %19f, \n" \
              $DOF \
              $REPS \
              $SIMPVMUL_ST \
              $SIMPVMUL_LO \
              $OPERATOR_ST \
              $OPERATOR_LO  | tee -a $FILE
}


printf "%10s %10s %20s %20s %20s %20s \n" \
                "DOFs" \
                "Repetitions" \
                "Simple vmult STD" \
                "Simple vmult LO" \
                "Operator STD" \
                "Operator LO"

for i in `seq 1 20`
do
  run $i 1000000 "small_size.tmp"
done

for i in `seq 21 40`
do
  run $i 10000 "big_size.tmp"
done

i=128
for n in 1 2 3 4 5
do
  let i=$i*2
  run $i 10 "very_big_size.tmp"
done

for n in `seq 1 25`
do
  let i=($n*2)-1
  let size=(1+2+3+4+5+6+7+8+9+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25)*20000/$n
  run $i $size "weak.tmp"
done

${PYTHON} ../${BENCH_DIR}/plot.py "small_size.tmp" "1000000 repetitions"
${PYTHON} ../${BENCH_DIR}/plot.py "big_size.tmp" "1000 repetitions"
${PYTHON} ../${BENCH_DIR}/plot.py "very_big_size.tmp" "10 repetitions"
${PYTHON} ../${BENCH_DIR}/plot.py "weak.tmp" "weak"