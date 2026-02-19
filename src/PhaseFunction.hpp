#pragma once

#include "DisortTypes.hpp"
#include <vector>

namespace disortpp {
namespace phase_function {

/// Fill Henyey-Greenstein moments: phaseFunctionMoments(k) = g^k
void fillHenyeyGreenstein(std::vector<std::vector<double>>& moments,
                          int nmom_nstr, int num_layers, double g, int lc = -1);

/// Fill isotropic phase function: phaseFunctionMoments(0)=1, rest=0
void fillIsotropic(std::vector<std::vector<double>>& moments,
                   int nmom_nstr, int num_layers, int lc = -1);

/// Fill Rayleigh phase function: phaseFunctionMoments(0)=1, phaseFunctionMoments(2)=0.1, rest=0
void fillRayleigh(std::vector<std::vector<double>>& moments,
                  int nmom_nstr, int num_layers, int lc = -1);

/// Fill Haze-L phase function (Garcia & Siewert 1985, Table 12-16)
void fillHazeGarciaSiewert(std::vector<std::vector<double>>& moments,
                           int nmom_nstr, int num_layers, int lc = -1);

/// Fill Cloud C.1 phase function (Garcia & Siewert 1985, Table 19-20)
void fillCloudGarciaSiewert(std::vector<std::vector<double>>& moments,
                            int nmom_nstr, int num_layers, int lc = -1);

/// Dispatch by PhaseFunction enum
void fillPhaseFunction(std::vector<std::vector<double>>& moments,
                       int nmom_nstr, int num_layers,
                       PhaseFunction type, double g = 0.0, int lc = -1);

} // namespace phase_function
} // namespace disortpp
