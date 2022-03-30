
#pragma once

#include "YAKL.h"
#include "yaml-cpp/yaml.h"


using yakl::memHost;
using yakl::memDevice;
using yakl::styleC;
using yakl::Array;
using yakl::SArray;

typedef double real;

inline real operator"" _fp( long double x ) {
  return static_cast<real>(x);
}


inline void endrun(std::string msg) {
  std::cerr << msg << std::endl << std::endl;
  throw msg;
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



