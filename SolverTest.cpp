//
// Created by dev on 1/13/2018.
//

#include "SolverTest.hpp"


namespace MinusDarwinTest {
    TEST_F(SolverWithSumFunction, TestScoreFunction) {
        ASSERT_FLOAT_EQ(solver->evaluateAgent({2.0f, 3.0f}), 5.0f);
    }

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
    TEST_F(SolverWithSinewaveFitFunction, TestFitting) {
        MinusDarwin::Agent result = solver->run(false);
        ASSERT_LE(solver->evaluateAgent(result),0.0005f);
    }
}
