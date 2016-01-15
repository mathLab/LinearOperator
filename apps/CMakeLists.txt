SET(TARGET "step")
SET(TARGET_SRC ${TARGET}.cc)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(benchmark)

ADD_EXECUTABLE(full_matrix_01 full_matrix_01.cc)
DEAL_II_SETUP_TARGET(full_matrix_01)

ADD_EXECUTABLE(full_matrix_02 full_matrix_02.cc)
DEAL_II_SETUP_TARGET(full_matrix_02)

ADD_EXECUTABLE(sparse_matrix_01 sparse_matrix_01.cc)
DEAL_II_SETUP_TARGET(sparse_matrix_01)

ADD_EXECUTABLE(sparse_matrix_02 sparse_matrix_02.cc)
DEAL_II_SETUP_TARGET(sparse_matrix_02)

ADD_EXECUTABLE(sparse_matrix_inefficient sparse_matrix_inefficient.cc)
DEAL_II_SETUP_TARGET(sparse_matrix_inefficient)

ADD_EXECUTABLE(sparse_matrix_residual sparse_matrix_residual.cc)
DEAL_II_SETUP_TARGET(sparse_matrix_residual)

ADD_EXECUTABLE(triple_add_01 triple_add_01.cc)
DEAL_II_SETUP_TARGET(triple_add_01)

ADD_EXECUTABLE(triple_add_02 triple_add_02.cc)
DEAL_II_SETUP_TARGET(triple_add_02)
