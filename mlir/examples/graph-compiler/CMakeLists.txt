# This chapter depends on JIT support enabled.
if(NOT MLIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

# For a better template to copy, see examples/standalone
include_directories(include)
#add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )

#set(LLVM_TARGET_DEFINITIONS mlir/ToyCombine.td)
#mlir_tablegen(ToyCombine.inc -gen-rewriters)
#add_public_tablegen_target(ToyCh6CombineIncGen)

add_llvm_executable(graph-compiler
    graph-compiler.cpp)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(graph-compiler
    PRIVATE
        ${dialect_libs}
        ${conversion_libs}
        ${extension_libs}
        MLIRIR)
