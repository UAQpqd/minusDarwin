//
// Created by dev on 3/15/2018.
//

#include "SinewaveFittingTuningTest.hpp"

namespace MinusDarwinTest {
    TEST_F(SinewaveFittingTuningTest, TestFitting) {
        const size_t reps = 4;
        const size_t maxGens = 100;
        const std::vector<size_t> popSizeVector = { 25, 50, 100, 200, 400 };
        const std::vector<MinusDarwin::CrossoverMode> modeVector = {
                MinusDarwin::CrossoverMode::Best,
                MinusDarwin::CrossoverMode::Random
        };
        const std::vector<size_t> modeDepthVector = { 1, 2, 3 };
        const std::vector<float> coProbVector = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
        const std::vector<float> diffFactorVector = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
        const std::vector<bool> useUniformVector = { false, true };
        for (int rep = 0; rep < reps; ++rep) {
            for(auto popSize : popSizeVector)
                for(auto mode : modeVector)
                    for(auto modeDepth : modeDepthVector)
                        for(auto coProb : coProbVector)
                            for(auto diffFactor : diffFactorVector)
                                for(auto useUniform : useUniformVector)
                                    runFittingTest(popSize,maxGens,mode,modeDepth,coProb,diffFactor,useUniform);
        }
        auto bestAgent = solver->run(true);
        auto bestAgentScore = solver->tracer.bestAgentScore;
        GTEST_ASSERT_LE(bestAgentScore, 0.005f);
    }

    void SinewaveFittingTuningTest::runFittingTest(size_t popSize, size_t maxGens,
                                                   MinusDarwin::CrossoverMode mode, size_t modeDepth, float coProb,
                                                   float diffFactor, bool useUniformFactor) {
        solverParameterSetSinewaveFittingTuning.popSize = popSize;
        solverParameterSetSinewaveFittingTuning.maxGens = maxGens;
        solverParameterSetSinewaveFittingTuning.mode = mode;
        solverParameterSetSinewaveFittingTuning.modeDepth = modeDepth;
        solverParameterSetSinewaveFittingTuning.coProb = coProb;
        solverParameterSetSinewaveFittingTuning.diffFactor = diffFactor;
        solverParameterSetSinewaveFittingTuning.useUniformFactor = useUniformFactor;
        auto bestAgent = solver->run(false);
        auto trace = solver->tracer;
        for (size_t gen = 0; gen <= maxGens; ++gen) {
            
        }
    }
}