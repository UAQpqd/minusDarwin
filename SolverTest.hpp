//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_SOLVERTEST_HPP
#define MINUSDARWIN_SOLVERTEST_HPP

#include <gtest/gtest.h>
#include "Solver.hpp"

namespace MinusDarwinTest {
    class SolverWithSumFunction : public ::testing::Test {
    public:
        MinusDarwin::Solver *solver;
        void SetUp() override {
            std::function<float (std::vector<float>)> sum = [](std::vector<float> v) { return v.at(0)+v.at(1); };
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    2,20,4,MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Best,1,
                    0.05f,0.7f,0.7f
            };
            solver = new MinusDarwin::Solver(
                    solverParameterSet,sum
            );
        }
        void TearDown() override {
            delete solver;
        }
    };
}

#endif //MINUSDARWIN_SOLVERTEST_HPP
