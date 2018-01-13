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
        MinusDarwin::Solver<float, float> *solver;
        void SetUp() override {
            std::function<float (float,float)> sum = [](float a,float b) { return a+b; };
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    10,4,MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Best,1,
                    0.05f,0.7f,0.7f
            };
            solver = new MinusDarwin::Solver<float,float>(
                    solverParameterSet,sum
            );
        }
        void TearDown() override {
            delete solver;
        }
    };
}

#endif //MINUSDARWIN_SOLVERTEST_HPP
