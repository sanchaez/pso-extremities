/* PSO DLL interface */

//  This header is intended to be used 
//  for importing and exporting DLL symbols.


#pragma once
#ifdef PSOEXTREMITIESWIN32_EXPORTS  
#define PSOEXTREMITIESWIN32_API __declspec(dllexport)   
#else  
#define PSOEXTREMITIESWIN32_API __declspec(dllimport)   
#endif  

#include <cmath>
#include <valarray>
#include <random>

namespace pso {

  //TODO: Add exported types and classes
}