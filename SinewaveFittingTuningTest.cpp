//
// Created by dev on 3/15/2018.
//

#include "SinewaveFittingTuningTest.hpp"

namespace MinusDarwinTest {
    TEST_F(SinewaveFittingTuningTest, TestFitting) {
        const size_t reps = 10;
        const size_t maxGens = 80;
        const std::vector<size_t> popSizeVector = { 25, 50, 100, 200, 400 };
        const std::vector<MinusDarwin::CrossoverMode> modeVector = {
                MinusDarwin::CrossoverMode::Best,
                MinusDarwin::CrossoverMode::Random
        };
        const std::vector<size_t> modeDepthVector = { 1, 2, 3 /* */};
        const std::vector<float> coProbVector = { /**/0.0f, 0.25f, 0.5f, 0.75f, /*1.0f*/ };
        const std::vector<float> diffFactorVector = { /**/0.0f, 0.25f, 0.5f, 0.75f, /*1.0f*/ };
        const std::vector<bool> useUniformVector = { /**/false, true };
        csv << "popSize,mode,modeDepth,coProb,diffFactor,useUniform,gen,mean,sd" << std::endl;
        for(auto popSize : popSizeVector)
            for(auto mode : modeVector)
                for(auto modeDepth : modeDepthVector)
                    for(auto coProb : coProbVector)
                        for(auto diffFactor : diffFactorVector)
                            for(auto useUniform : useUniformVector) {
                                std::vector<std::vector<float> > genBestScoresByGen(maxGens+1,std::vector<float>(reps,0.0f));
                                for (int rep = 0; rep < reps; ++rep) {
                                    runFittingTest(popSize, maxGens, mode, modeDepth, coProb, diffFactor, useUniform);
                                    auto trace = solver->tracer;
                                    for(size_t gen = 0; gen<=maxGens; gen++) {
                                        auto bestScoreOfgen = *std::min_element(
                                                trace.generationsScores.at(gen).begin(),
                                                trace.generationsScores.at(gen).end()
                                        );
                                        genBestScoresByGen.at(gen).at(rep) = bestScoreOfgen;
                                    }
                                }
                                //Once all the best scores by gen are calculate obtain quartiles to report
                                std::vector<float> meanByGeneration(maxGens+1,0.0f);
                                std::vector<float> sdByGeneration(maxGens+1,0.0f);
                                std::transform(genBestScoresByGen.begin(),genBestScoresByGen.end(),
                                               meanByGeneration.begin(),[](const std::vector<float> &values) -> float {
                                            return 1.0f/((float)values.size())*std::accumulate(values.begin(),values.end(),0.0f);
                                        });
                                std::transform(genBestScoresByGen.begin(),genBestScoresByGen.end(),meanByGeneration.begin(),
                                               sdByGeneration.begin(),[](const std::vector<float> &values, const float &mean) -> float {
                                            return sqrt(
                                                    1.0f/((float)values.size()-1.0f)*
                                                            std::accumulate(values.begin(),values.end(),0.0f,
                                                                            [&mean](float accum, const float val) {
                                                                                return accum + (val-mean)*(val-mean);}));
                                        });

                                for(size_t gen = 0; gen<=maxGens; gen++) {
                                        csv << popSize <<","<<
                                            (mode==MinusDarwin::CrossoverMode::Best?"Best":"Random") <<","<<
                                            modeDepth <<","<<coProb <<","<<diffFactor <<","<<useUniform<<","<<gen<<","<<
                                                meanByGeneration.at(gen) << "," << sdByGeneration.at(gen) <<
                                            std::endl;
                                    }

                                }

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
    }
}