#!/bin/bash

[ -f CMakeCache.txt ] || cninja ..
ninja -j4
clear

function print_header
{
  local EXECUTABLE=$1

  echo ""
  echo ""
  echo "$EXECUTABLE:"
  echo ""
  printf "%10s %10s %20s %20s\n" \
         "DOFs" "Repetitions" "ST" "LO"
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

  printf "%10d %10d %20f %20f \n" \
         $DOF $REPS $ST $LO | tee -a $OUTPUT
}

print_header full_matrix_01
for i in `seq 1 10`; do
  step full_matrix_01 $((2**$i)) 25000 "full_matrix_01.tmp"
done

print_header full_matrix_02
for i in `seq 1 10`; do
  step full_matrix_02 $((2**$i)) 10000 "full_matrix_02.tmp"
done

print_header sparse_matrix_01
for i in `seq 1 7`; do
  step sparse_matrix_01 $i 25000 "sparse_matrix_01.tmp"
done

print_header sparse_matrix_02
for i in `seq 1 7`; do
  step sparse_matrix_02 $i 10000 "sparse_matrix_02.tmp"
done

print_header sparse_matrix_inefficient
for i in `seq 1 10`; do
  step sparse_matrix_inefficient $i 300 "sparse_matrix_inefficient.tmp"
done

print_header sparse_matrix_residual
for i in `seq 1 10`; do
  step sparse_matrix_residual $i 500 "sparse_matrix_residual.tmp"
done

print_header triple_add_01
for i in `seq 1 10`; do
  step triple_add_01 $((2**$i)) 20000 "triple_add_01.tmp"
done

print_header triple_add_02
for i in `seq 1 10`; do
  step triple_add_02 $((2**$i)) 20000 "triple_add_02.tmp"
done
