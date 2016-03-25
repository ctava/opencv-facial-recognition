%module facerecognizer
%{
#include "opencv2/core/core.hpp"
#include "facerecognizer.h"
%}

%include <typemaps.i>
%include "std_string.i"
%include "std_vector.i"

%include "facerecognizer.h"