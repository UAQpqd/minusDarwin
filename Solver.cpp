//
// Created by dev on 1/13/2018.
//

#include "Solver.hpp"

bool MinusDarwin::Solver::useOpenCL() {
    return device.id() != 0;
}
