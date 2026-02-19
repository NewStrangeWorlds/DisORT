# DisORT++

A modern C++17 implementation of the DISORT (Discrete Ordinates Radiative
Transfer) algorithm. DisORT++ solves the radiative transfer equation in a
plane-parallel or pseudo-spherical atmosphere, computing fluxes, mean
intensities, and angular intensities at arbitrary optical depths and directions.

## Features

- Discrete-ordinate method with arbitrary even stream count
- Henyey-Greenstein, Rayleigh, and tabulated phase functions
- Delta-M and Delta-M+ scaling for forward-peaked scattering
- Thermal emission with Planck-function integration
- Lambertian and BRDF surface reflection (RPV, Cox-Munk, Ambrals, Hapke)
- Plane-parallel and pseudo-spherical beam geometry
- Nakajima-Tanaka and Buras-Emde intensity corrections
- OpenMP-parallelised spectral loop functions for multi-wavenumber calculations
- Python bindings via pybind11

## Two solver interfaces

| | DisortSolver | DisortFluxSolver |
|---|---|---|
| Output | Fluxes + intensities | Fluxes only |
| Surface | Lambertian + BRDF | Lambertian |
| Streams | Any even number >= 4 | 4, 6, 8, 10, 12, 14, 16, 32, 64 |
| Performance | Baseline | 30-50% faster |

## Installation

### From PyPI

```bash
pip install disortpp
```

### From source (Python module)

```bash
git clone https://github.com/NewStrangeWorlds/DisORT.git
cd DisORT
pip install .
```

### From source (C++ library only)

```bash
cd DisORT
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Dependencies (Eigen 3.4, pybind11) are fetched automatically via CMake
FetchContent.

## Quick start (Python)

```python
import disortpp
import numpy as np

nlyr = 4
nstr = 16

cfg = disortpp.DisortConfig(nlyr, nstr)
cfg.flags.use_lambertian_surface = True
cfg.flags.comp_only_fluxes = True
cfg.allocate()

cfg.delta_tau = [0.1, 0.5, 1.0, 2.0]
cfg.single_scat_albedo = [0.9, 0.9, 0.9, 0.9]
cfg.set_henyey_greenstein(g=0.8)

cfg.bc.direct_beam_flux = np.pi
cfg.bc.direct_beam_mu = 0.5
cfg.bc.surface_albedo = 0.1

solver = disortpp.DisortSolver()
result = solver.solve(cfg)

flux_up = np.array(result.flux_up)
print(f"Upward flux at TOA: {flux_up[0]:.6f}")
```

### Spectral loops

Solve many wavenumber bands efficiently in a single C++ call with OpenMP
parallelisation:

```python
bands = [(100.0, 200.0), (200.0, 300.0), (300.0, 400.0)]
results = disortpp.solve_spectral_bands(cfg, bands)
flux_spectrum = np.array([r.flux_up[0] for r in results])
```

## Quick start (C++)

```cpp
#include "DisortConfig.hpp"
#include "DisortSolver.hpp"

using namespace disortpp;

int main() {
    DisortConfig cfg(4, 16);
    cfg.flags.use_lambertian_surface = true;
    cfg.flags.comp_only_fluxes = true;
    cfg.allocate();

    cfg.delta_tau = {0.1, 0.5, 1.0, 2.0};
    cfg.single_scat_albedo = {0.9, 0.9, 0.9, 0.9};
    cfg.setHenyeyGreenstein(0.8);

    cfg.bc.direct_beam_flux = 3.14159;
    cfg.bc.direct_beam_mu = 0.5;
    cfg.bc.surface_albedo = 0.1;

    DisortSolver solver;
    DisortResult result = solver.solve(cfg);

    double flux_up_toa = result.flux_up[0];
}
```

## CMake build options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Release` (`-O3`) or `Debug` (`-g -Wall`) |
| `BUILD_PYTHON_BINDINGS` | `OFF` | Build the `disortpp` Python module |
| `DISORTPP_MARCH_NATIVE` | `ON` | Use `-march=native` (disable for portable binaries) |

## Running the tests

```bash
cd build
ctest --output-on-failure
```

## License

GNU General Public License v3 (GPLv3). See [LICENSE](LICENSE) for details.
