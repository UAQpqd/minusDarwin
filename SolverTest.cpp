//
// Created by dev on 1/13/2018.
//

#include "SolverTest.hpp"


namespace MinusDarwinTest {
    TEST_F(SolverWithSumFunction, TestScoreFunction) {
        ASSERT_FLOAT_EQ(solver->evaluateAgent({2.0f, 3.0f}), 5.0f);
    }

    TEST_F(SolverWithSumFunction, TestPopulationScoreFunction) {
        MinusDarwin::Population X(solver->sParams.popSize, std::vector<float>(solver->sParams.dims, 0.0f));
        std::vector<float> Xscores(solver->sParams.popSize);
        solver->initPopulation(X);
        solver->evaluatePopulation(Xscores, X);
        boost::accumulators::accumulator_set<
                float,
                ba::stats<ba::tag::mean, ba::tag::variance> > acc;
        for (auto &s:Xscores) acc(s);
        ASSERT_GT(ba::variance(acc), 0.0f);
    }
    TEST_F(SolverWithSumFunction, TestNeighbourCreation) {
        auto neighbours =
                MinusDarwin::Neighbours(sParams.popSize, std::vector<size_t>(kNeighsPerAgent,0));
        createNeighbours(neighbours,1);

    }
}
