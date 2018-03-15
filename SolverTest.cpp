//
// Created by dev on 1/13/2018.
//

#include "SolverTest.hpp"

namespace MinusDarwinTest {

    TEST_F(SolverWithSumFunction, TestNeighbourCreation) {
        auto neighbours =
                MinusDarwin::Neighbours(
                        solver->sParams.popSize,
                        std::vector<size_t>(kNeighsPerAgent(solver->sParams.modeDepth),0));
        solver->createNeighbours(neighbours,1);
        for(auto &a : neighbours) {
            ASSERT_EQ(a.at(0),1);
            std::sort(a.begin(),a.end());
            std::unique(a.begin(),a.end());
            ASSERT_EQ(a.size(),kNeighsPerAgent(solver->sParams.modeDepth));
        }
    }
    TEST_F(SolverWithSumFunction, TestFitting) {
        auto bestAgent = solver->run(false);
        GTEST_ASSERT_LE(bestAgent.at(0)+bestAgent.at(1), 0.05f);
    }
    TEST_F(SolverWithSinewaveFitFunction, TestFitting) {
        auto bestAgent = solver->run(true);
        auto bestAgentScore = solver->tracer.bestAgentScore;
        GTEST_ASSERT_LE(bestAgentScore, 0.005f);
    }
    TEST_F(SolverWithSinewaveFitFunctionOpenCL, TestFitting) {
        auto bestAgent = solver->run(true);
        auto bestAgentScore = solver->tracer.bestAgentScore;
        GTEST_ASSERT_LE(bestAgentScore, 0.005f);
    }
}
