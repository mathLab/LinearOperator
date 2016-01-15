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

function extract
{
  let end=$1+11
  sed -n "${1},${end}p" < ${2}
}



./test_00 | tee $FILE_00
./test_01 | tee $FILE_01

for FILE in $FILE_00 $FILE_01
do
  printf " %10s, %10s, %20s, %20s \n" "n" "iter" "raw" "lo" | tee -a ${FILE}.data
  start=1
  while : ; do
    result=$(extract $start $FILE)
    let start=$start+13
    # echo "$result"
    n=$(echo "$result" | grep "n:" | sed 's#n:##')
    iter=$(echo "$result" | grep "iter:" | sed 's#iter:##')
    raw=$(  echo "$result" | \
                  grep "raw" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')
    lo=$(  echo  "$result" | \
                  grep "linear_operator" | \
                  cut -d'|' -f4 |\
                  sed 's#s##')
    printf " %10d, %10d, %20f, %20f \n" $n $iter $raw $lo | tee -a ${FILE}.data
    [[ "$result" ]] || break
  done
done

${PYTHON} ../${BENCH_DIR}/plot_2.py ${FILE_00}.data "simple multiplication" 2
${PYTHON} ../${BENCH_DIR}/plot_2.py ${FILE_01}.data "operator" 4