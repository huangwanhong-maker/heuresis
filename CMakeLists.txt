# heuresis: search and optimisation over spaces too large to enumerate.
#
# Every module is a header-only target heuresis::<module>. heuresis uses the targets of calx, morphe, physis
# and tyche: inside The Science Infrastructures, the siblings under foundations/ (each added here when no
# larger project has added it); built alone elsewhere, they would be fetched at pinned versions (not yet
# written). Built on its own, heuresis builds its checks (HEURESIS_CHECKS); added to a larger project, only
# when asked.

cmake_minimum_required(VERSION 3.20)
project(heuresis VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(HEURESIS_TOP_LEVEL ON)
else()
  set(HEURESIS_TOP_LEVEL OFF)
endif()
option(HEURESIS_CHECKS "Build heuresis's checks" ${HEURESIS_TOP_LEVEL})

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Threads REQUIRED)

# heuresis_sibling(<directory> <a target it defines>): the sibling library, added once.
function(heuresis_sibling directory target)
  if(NOT TARGET ${target})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../${directory}/CMakeLists.txt)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../${directory} ${CMAKE_BINARY_DIR}/${directory})
    else()
      message(FATAL_ERROR "heuresis needs ${directory}: build it inside The Science Infrastructures (foundations/${directory}).")
    endif()
  endif()
endfunction()

# The siblings in their order: calx and morphe, then physis, then tyche.
heuresis_sibling(calx calx::linalg)
heuresis_sibling(morphe morphe::topology)
heuresis_sibling(physis physis::units)
heuresis_sibling(tyche tyche::probability)
if(HEURESIS_CHECKS)
  enable_testing()
endif()

# heuresis_module(<name> [LIBRARIES ...] [CHECK_LIBRARIES ...]): the target heuresis::<name> and, with the
# checks, check_heuresis_<name>, which may draw its problems from the benchmarks.
function(heuresis_module name)
  cmake_parse_arguments(ARG "" "" "LIBRARIES;CHECK_LIBRARIES" ${ARGN})
  add_library(heuresis_${name} INTERFACE)
  add_library(heuresis::${name} ALIAS heuresis_${name})
  target_include_directories(heuresis_${name} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/${name}/include)
  target_compile_features(heuresis_${name} INTERFACE cxx_std_20)
  target_link_libraries(heuresis_${name} INTERFACE ${ARG_LIBRARIES} Threads::Threads)
  if(HEURESIS_CHECKS)
    set(check_libraries heuresis::${name} calx::random ${ARG_CHECK_LIBRARIES})
    if(NOT name STREQUAL "search" AND NOT name STREQUAL "benchmarks")
      list(APPEND check_libraries heuresis::benchmarks)
    endif()
    calx_add_check(check_heuresis_${name} SOURCES ${name}/tests/check_${name}.cpp LIBRARIES ${check_libraries})
  endif()
endfunction()

# The modules in their order (decision 0004).
heuresis_module(search LIBRARIES calx::random calx::scheduler)
heuresis_module(variation LIBRARIES heuresis::search)
heuresis_module(selection LIBRARIES heuresis::search)
heuresis_module(benchmarks LIBRARIES heuresis::variation calx::linalg)
heuresis_module(analysis LIBRARIES heuresis::search tyche::descriptive tyche::probability tyche::testing)
heuresis_module(genetic LIBRARIES heuresis::variation heuresis::selection)
heuresis_module(local LIBRARIES heuresis::variation calx::optimisation)
heuresis_module(strategies LIBRARIES heuresis::variation heuresis::selection calx::linalg)
heuresis_module(swarms LIBRARIES heuresis::variation heuresis::selection)
heuresis_module(distributions LIBRARIES heuresis::variation heuresis::selection calx::linalg)
heuresis_module(quantum LIBRARIES heuresis::distributions)
heuresis_module(annealing LIBRARIES heuresis::variation heuresis::selection)
heuresis_module(multiobjective LIBRARIES heuresis::variation heuresis::selection calx::linalg)
heuresis_module(diversity LIBRARIES heuresis::variation heuresis::selection)
heuresis_module(programs LIBRARIES heuresis::variation heuresis::selection morphe::formal physis::units)
heuresis_module(populations LIBRARIES heuresis::variation heuresis::selection CHECK_LIBRARIES heuresis::genetic)
heuresis_module(models LIBRARIES heuresis::variation heuresis::selection tyche::spatial tyche::probability calx::optimisation)
heuresis_module(landscapes LIBRARIES heuresis::variation tyche::descriptive)
