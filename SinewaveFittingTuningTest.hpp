//
// Created by dev on 3/15/2018.
//

#ifndef MINUSDARWIN_SINEWAVEFITTINGTUNINGTEST_HPP
#define MINUSDARWIN_SINEWAVEFITTINGTUNINGTEST_HPP


#include <gtest/gtest.h>
#include <boost/compute.hpp>
#include "Solver.hpp"

namespace bc = boost::compute;
MinusDarwin::SolverParameterSet solverParameterSetSinewaveFittingTuning = {
        2, 200, 100, MinusDarwin::GoalFunction::MaxGenerations,
        MinusDarwin::CrossoverMode::Random, 1,
        0.005f, 0.7f, 0.8f, true
};

namespace MinusDarwinTest {
    class SinewaveFittingTuningTest : public ::testing::Test {
    public:
        MinusDarwin::Solver *solver;
        std::ofstream csv;
        const std::string csvFilename = "SinewaveFittingTuningTest.csv";
        BOOST_COMPUTE_FUNCTION(float,accumSquared,(float accum,float val), {
            return accum + val*val;
        });

        void runFittingTest(
                size_t popSize,
        size_t maxGens,
        MinusDarwin::CrossoverMode mode,
        size_t modeDepth,
        float coProb,
        float diffFactor,
        bool useUniformFactor);
        void SetUp() override {
            csv.open(csvFilename);
            const char sinewaveFitSource[] =
                    BOOST_COMPUTE_STRINGIZE_SOURCE(
                            __kernel void calculateScoresOfPopulation(
                                    __global const float *signalData,
                            __global const float2 *population,
                            __global float *scores,
                            const size_t populationSize,
                            const size_t signalDataSize,
                            const float sumOfSquares,
                            const float omegaMin,
                            const float omegaMax,
                            const float phiMax,
                            const size_t sps,
                            const float a)
                    {
                            const uint aId = get_global_id(0);
                            const float2 agent = population[aId];
                            float error = 0.0f;
                            for (size_t p = 0; p < signalDataSize; p++) {
                            float t = (float)p/(float)sps;
                            float realOmega = omegaMin+agent[0]*(omegaMax-omegaMin);
                            float realPhi = agent[1]*phiMax;
                            float estimated =
                            a*sin(realOmega*t+realPhi);
                            error += (estimated-signalData[p])*(estimated-signalData[p]);
                    }
                            scores[aId] = error/sumOfSquares;
                    });
            const float time = 10.0f;
            const size_t sps = 8000;
            const float a = 100.0f;
            const float omega = 2.0f * (float) M_PI * 60.0f;
            const float omegaMin = omega * 0.95f;
            const float omegaMax = omega * 1.05f;
            const float phiMax = 2.0f * (float) M_PI;
            const float phi = 0.0f;
            std::vector<float> signalData(
                    (size_t) floor((float) sps * time), 0.0f);
            for (size_t p = 0; p < signalData.size(); p++) {
                float t = (float) p / (float) sps;
                signalData.at(p) =
                        a * sin(omega * t + phi);
            }
            auto dev = bc::system::default_device();
            auto ctx = new bc::context(dev);
            auto queue = new bc::command_queue(*ctx,dev);
            auto dSignalData = new bc::vector<float>(
                    signalData.begin(),signalData.end(),
                    *queue);
            auto dScores = new bc::vector<float>(solverParameterSetSinewaveFittingTuning.popSize,*ctx);
            auto dPopulation = new bc::vector<bc::float2_>(solverParameterSetSinewaveFittingTuning.popSize,*ctx);
            const float sumOfSquares =
                    bc::accumulate(dSignalData->begin(),dSignalData->end(),0.0f,
                                   accumSquared
                            ,*queue);

            bc::program program =
                    bc::program::create_with_source(sinewaveFitSource, *ctx);
            // compile the program
            program.build();
            auto kernel = bc::kernel(program,"calculateScoresOfPopulation");
            kernel.set_arg(0,*dSignalData);
            kernel.set_arg(1,*dPopulation);
            kernel.set_arg(2,*dScores);
            kernel.set_arg(3,dPopulation->size());
            kernel.set_arg(4,dSignalData->size());
            kernel.set_arg(5,sumOfSquares);
            kernel.set_arg(6,omegaMin);
            kernel.set_arg(7,omegaMax);
            kernel.set_arg(8,phiMax);
            kernel.set_arg(9,sps);
            kernel.set_arg(10,a);

            auto evaluatePopulationFitError =
                    [queue, kernel, dSignalData, dScores, dPopulation, sumOfSquares, sps, a,
                            omegaMin, omegaMax, phiMax](
                            MinusDarwin::Scores &scores,
                            const MinusDarwin::Population &population) {
                        //Population to Device
                        std::vector<bc::float2_> hPopulation(population.size());
                        std::transform(population.begin(),population.end(),
                                       hPopulation.begin(),[](const MinusDarwin::Agent &a){
                                    bc::float2_ b;
                                    b[0] = a.at(0);
                                    b[1] = a.at(1);
                                    return b;
                                });
                        bc::copy(hPopulation.begin(),hPopulation.end(),dPopulation->begin(),*queue);
                        //Once population has been copied to the device
                        //a parallel calculation of scores must be done
                        queue->enqueue_1d_range_kernel(kernel,0,hPopulation.size(),0);
                        bc::copy(dScores->begin(),dScores->end(),scores.begin(),*queue);
                    };
            solver = new MinusDarwin::Solver(
                    solverParameterSetSinewaveFittingTuning, evaluatePopulationFitError
            );
        }
        void TearDown() override {
            csv.close();
            delete solver;
        }
    };
}


#endif //MINUSDARWIN_SINEWAVEFITTINGTUNINGTEST_HPP
