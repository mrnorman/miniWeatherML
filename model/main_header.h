
#pragma once

#include "mpi.h"
#include "YAKL.h"
#include "yaml-cpp/yaml.h"
#include <stdexcept>


using yakl::memHost;
using yakl::memDevice;
using yakl::styleC;
using yakl::Array;
using yakl::SArray;

inline void debug_print( char const * file , int line ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << std::endl;
}

template <class T> inline void debug_print_sum( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": sum(" << varname << ")  -->  " << yakl::intrinsics::sum( var ) << std::endl;
}

template <class T> inline void debug_print_min( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": minval(" << varname << ")  -->  " << yakl::intrinsics::minval( var ) << std::endl;
}

template <class T> inline void debug_print_max( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": maxval(" << varname << ")  -->  " << yakl::intrinsics::maxval( var ) << std::endl;
}

template <class T> inline void debug_print_val( T var , char const * file , int line , char const * varname ) {
  MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (rank == 0) std::cout << "*** DEBUG: " << file << ": " << line << ": " << varname << "  -->  " << var << std::endl;
}

#define DEBUG_PRINT_MAIN() { debug_print(__FILE__,__LINE__); }
#define DEBUG_PRINT_MAIN_SUM(var) { debug_print_sum((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_MIN(var) { debug_print_min((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_MAX(var) { debug_print_max((var),__FILE__,__LINE__,#var); }
#define DEBUG_PRINT_MAIN_VAL(var) { debug_print_val((var),__FILE__,__LINE__,#var); }

int constexpr max_fields = 50;

typedef double real;

YAKL_INLINE real constexpr operator"" _fp( long double x ) {
  return static_cast<real>(x);
}


YAKL_INLINE void endrun(char const * msg) {
  yakl::yakl_throw(msg);
};


typedef Array<real,1,memDevice,styleC> real1d;
typedef Array<real,2,memDevice,styleC> real2d;
typedef Array<real,3,memDevice,styleC> real3d;
typedef Array<real,4,memDevice,styleC> real4d;
typedef Array<real,5,memDevice,styleC> real5d;
typedef Array<real,6,memDevice,styleC> real6d;
typedef Array<real,7,memDevice,styleC> real7d;

typedef Array<real const,1,memDevice,styleC> realConst1d;
typedef Array<real const,2,memDevice,styleC> realConst2d;
typedef Array<real const,3,memDevice,styleC> realConst3d;
typedef Array<real const,4,memDevice,styleC> realConst4d;
typedef Array<real const,5,memDevice,styleC> realConst5d;
typedef Array<real const,6,memDevice,styleC> realConst6d;
typedef Array<real const,7,memDevice,styleC> realConst7d;

typedef Array<real,1,memHost,styleC> realHost1d;
typedef Array<real,2,memHost,styleC> realHost2d;
typedef Array<real,3,memHost,styleC> realHost3d;
typedef Array<real,4,memHost,styleC> realHost4d;
typedef Array<real,5,memHost,styleC> realHost5d;
typedef Array<real,6,memHost,styleC> realHost6d;
typedef Array<real,7,memHost,styleC> realHost7d;

typedef Array<real const,1,memHost,styleC> realConstHost1d;
typedef Array<real const,2,memHost,styleC> realConstHost2d;
typedef Array<real const,3,memHost,styleC> realConstHost3d;
typedef Array<real const,4,memHost,styleC> realConstHost4d;
typedef Array<real const,5,memHost,styleC> realConstHost5d;
typedef Array<real const,6,memHost,styleC> realConstHost6d;
typedef Array<real const,7,memHost,styleC> realConstHost7d;



typedef Array<int,1,memDevice,styleC> int1d;
typedef Array<int,2,memDevice,styleC> int2d;
typedef Array<int,3,memDevice,styleC> int3d;
typedef Array<int,4,memDevice,styleC> int4d;
typedef Array<int,5,memDevice,styleC> int5d;
typedef Array<int,6,memDevice,styleC> int6d;
typedef Array<int,7,memDevice,styleC> int7d;

typedef Array<int const,1,memDevice,styleC> intConst1d;
typedef Array<int const,2,memDevice,styleC> intConst2d;
typedef Array<int const,3,memDevice,styleC> intConst3d;
typedef Array<int const,4,memDevice,styleC> intConst4d;
typedef Array<int const,5,memDevice,styleC> intConst5d;
typedef Array<int const,6,memDevice,styleC> intConst6d;
typedef Array<int const,7,memDevice,styleC> intConst7d;

typedef Array<int,1,memHost,styleC> intHost1d;
typedef Array<int,2,memHost,styleC> intHost2d;
typedef Array<int,3,memHost,styleC> intHost3d;
typedef Array<int,4,memHost,styleC> intHost4d;
typedef Array<int,5,memHost,styleC> intHost5d;
typedef Array<int,6,memHost,styleC> intHost6d;
typedef Array<int,7,memHost,styleC> intHost7d;

typedef Array<int const,1,memHost,styleC> intConstHost1d;
typedef Array<int const,2,memHost,styleC> intConstHost2d;
typedef Array<int const,3,memHost,styleC> intConstHost3d;
typedef Array<int const,4,memHost,styleC> intConstHost4d;
typedef Array<int const,5,memHost,styleC> intConstHost5d;
typedef Array<int const,6,memHost,styleC> intConstHost6d;
typedef Array<int const,7,memHost,styleC> intConstHost7d;



typedef Array<bool,1,memDevice,styleC> bool1d;
typedef Array<bool,2,memDevice,styleC> bool2d;
typedef Array<bool,3,memDevice,styleC> bool3d;
typedef Array<bool,4,memDevice,styleC> bool4d;
typedef Array<bool,5,memDevice,styleC> bool5d;
typedef Array<bool,6,memDevice,styleC> bool6d;
typedef Array<bool,7,memDevice,styleC> bool7d;

typedef Array<bool const,1,memDevice,styleC> boolConst1d;
typedef Array<bool const,2,memDevice,styleC> boolConst2d;
typedef Array<bool const,3,memDevice,styleC> boolConst3d;
typedef Array<bool const,4,memDevice,styleC> boolConst4d;
typedef Array<bool const,5,memDevice,styleC> boolConst5d;
typedef Array<bool const,6,memDevice,styleC> boolConst6d;
typedef Array<bool const,7,memDevice,styleC> boolConst7d;

typedef Array<bool,1,memHost,styleC> boolHost1d;
typedef Array<bool,2,memHost,styleC> boolHost2d;
typedef Array<bool,3,memHost,styleC> boolHost3d;
typedef Array<bool,4,memHost,styleC> boolHost4d;
typedef Array<bool,5,memHost,styleC> boolHost5d;
typedef Array<bool,6,memHost,styleC> boolHost6d;
typedef Array<bool,7,memHost,styleC> boolHost7d;

typedef Array<bool const,1,memHost,styleC> boolConstHost1d;
typedef Array<bool const,2,memHost,styleC> boolConstHost2d;
typedef Array<bool const,3,memHost,styleC> boolConstHost3d;
typedef Array<bool const,4,memHost,styleC> boolConstHost4d;
typedef Array<bool const,5,memHost,styleC> boolConstHost5d;
typedef Array<bool const,6,memHost,styleC> boolConstHost6d;
typedef Array<bool const,7,memHost,styleC> boolConstHost7d;



typedef Array<float,1,memDevice,styleC> float1d;
typedef Array<float,2,memDevice,styleC> float2d;
typedef Array<float,3,memDevice,styleC> float3d;
typedef Array<float,4,memDevice,styleC> float4d;
typedef Array<float,5,memDevice,styleC> float5d;
typedef Array<float,6,memDevice,styleC> float6d;
typedef Array<float,7,memDevice,styleC> float7d;

typedef Array<float const,1,memDevice,styleC> floatConst1d;
typedef Array<float const,2,memDevice,styleC> floatConst2d;
typedef Array<float const,3,memDevice,styleC> floatConst3d;
typedef Array<float const,4,memDevice,styleC> floatConst4d;
typedef Array<float const,5,memDevice,styleC> floatConst5d;
typedef Array<float const,6,memDevice,styleC> floatConst6d;
typedef Array<float const,7,memDevice,styleC> floatConst7d;

typedef Array<float,1,memHost,styleC> floatHost1d;
typedef Array<float,2,memHost,styleC> floatHost2d;
typedef Array<float,3,memHost,styleC> floatHost3d;
typedef Array<float,4,memHost,styleC> floatHost4d;
typedef Array<float,5,memHost,styleC> floatHost5d;
typedef Array<float,6,memHost,styleC> floatHost6d;
typedef Array<float,7,memHost,styleC> floatHost7d;

typedef Array<float const,1,memHost,styleC> floatConstHost1d;
typedef Array<float const,2,memHost,styleC> floatConstHost2d;
typedef Array<float const,3,memHost,styleC> floatConstHost3d;
typedef Array<float const,4,memHost,styleC> floatConstHost4d;
typedef Array<float const,5,memHost,styleC> floatConstHost5d;
typedef Array<float const,6,memHost,styleC> floatConstHost6d;
typedef Array<float const,7,memHost,styleC> floatConstHost7d;



typedef Array<double,1,memDevice,styleC> double1d;
typedef Array<double,2,memDevice,styleC> double2d;
typedef Array<double,3,memDevice,styleC> double3d;
typedef Array<double,4,memDevice,styleC> double4d;
typedef Array<double,5,memDevice,styleC> double5d;
typedef Array<double,6,memDevice,styleC> double6d;
typedef Array<double,7,memDevice,styleC> double7d;

typedef Array<double const,1,memDevice,styleC> doubleConst1d;
typedef Array<double const,2,memDevice,styleC> doubleConst2d;
typedef Array<double const,3,memDevice,styleC> doubleConst3d;
typedef Array<double const,4,memDevice,styleC> doubleConst4d;
typedef Array<double const,5,memDevice,styleC> doubleConst5d;
typedef Array<double const,6,memDevice,styleC> doubleConst6d;
typedef Array<double const,7,memDevice,styleC> doubleConst7d;

typedef Array<double,1,memHost,styleC> doubleHost1d;
typedef Array<double,2,memHost,styleC> doubleHost2d;
typedef Array<double,3,memHost,styleC> doubleHost3d;
typedef Array<double,4,memHost,styleC> doubleHost4d;
typedef Array<double,5,memHost,styleC> doubleHost5d;
typedef Array<double,6,memHost,styleC> doubleHost6d;
typedef Array<double,7,memHost,styleC> doubleHost7d;

typedef Array<double const,1,memHost,styleC> doubleConstHost1d;
typedef Array<double const,2,memHost,styleC> doubleConstHost2d;
typedef Array<double const,3,memHost,styleC> doubleConstHost3d;
typedef Array<double const,4,memHost,styleC> doubleConstHost4d;
typedef Array<double const,5,memHost,styleC> doubleConstHost5d;
typedef Array<double const,6,memHost,styleC> doubleConstHost6d;
typedef Array<double const,7,memHost,styleC> doubleConstHost7d;



