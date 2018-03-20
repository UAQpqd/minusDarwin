//
// Created by dev on 1/13/2018.
//

#ifndef MINUSDARWIN_SOLVERTEST_HPP
#define MINUSDARWIN_SOLVERTEST_HPP

#include <gtest/gtest.h>
#include <boost/compute.hpp>
#include "Solver.hpp"

namespace bc = boost::compute;
MinusDarwin::SolverParameterSet solverParameterSetSinewaveFitting1 = {
        2, 200, 100, MinusDarwin::GoalFunction::EpsilonReached,
        MinusDarwin::CrossoverMode::Best, 1,
        0.0005f, 0.7f, 0.8f, false
};
namespace MinusDarwinTest {
    class SolverWithSumFunction : public ::testing::Test {
    public:
        MinusDarwin::Solver *solver;

        void SetUp() override {
            auto evaluatePopulationSum =
                    [](MinusDarwin::Scores &scores,
                    const MinusDarwin::Population &population) {
                        for (size_t aId = 0; aId < population.size(); ++aId) {
                            auto &agent = population.at(aId);
                            scores.at(aId) = agent.at(0) + agent.at(1);
                        }
                    };
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    2, 20, 4, MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Best, 1,
                    0.05f, 0.7f, 0.7f, false
            };
            solver = new MinusDarwin::Solver(
                    solverParameterSet, evaluatePopulationSum
            );
        }

        void TearDown() override {
            delete solver;
        }
    };

    class SolverWithSinewaveFitFunction : public ::testing::Test {
    public:
        MinusDarwin::Solver *solver;

        void SetUp() override {

            const float time = 4.0f;
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
            const float sumOfSquares = std::accumulate(
                    signalData.begin(),signalData.end(),0.0f,
                    [](float accum, float val) { return accum+val*val; });
            auto evaluatePopulationFitError =
                    [signalData, sumOfSquares, sps, a,
                            omegaMin, omegaMax, phiMax](
                            MinusDarwin::Scores &scores,
                            const MinusDarwin::Population &population) {
                        for (size_t aId = 0; aId < population.size(); ++aId) {
                            auto &agent = population.at(aId);
                            float error = 0.0f;
                            for (size_t p = 0; p < signalData.size(); p++) {
                                float t = (float)p/(float)sps;
                                float realOmega = omegaMin+agent.at(0)*(omegaMax-omegaMin);
                                float realPhi = agent.at(1)*phiMax;
                                float estimated =
                                        a*sin(realOmega*t+realPhi);
                                error += pow(estimated-signalData.at(p),2.0f);
                            }
                            scores.at(aId) = error/sumOfSquares;
                        }
                    };
            solver = new MinusDarwin::Solver(
                    solverParameterSetSinewaveFitting1, evaluatePopulationFitError
            );
        }
        void TearDown() override {
            delete solver;
        }
    };


    class SolverWithSinewaveFitFunctionOpenCL : public ::testing::Test {
    public:
        MinusDarwin::Solver *solver;
        BOOST_COMPUTE_FUNCTION(float,accumSquared,(float accum,float val), {
            return accum + val*val;
        });


        void SetUp() override {
            const char sinewaveFitSource[] =
                    BOOST_COMPUTE_STRINGIZE_SOURCE(
                            __kernel void calculateScoresOfPopulation(
                                    __global const float *signalData,
                            __global const float2 *population,
                            __global float *scores,
                            const unsigned int populationSize,
                            const unsigned int signalDataSize,
                            const float sumOfSquares,
                            const float omegaMin,
                            const float omegaMax,
                            const float phiMax,
                            const unsigned int sps,
                            const float a)
                    {
                            const uint aId = get_global_id(0);
                            const float2 agent = population[aId];
                            float error = 0.0f;
                            for (size_t p = 0; p < signalDataSize; p++) {
                            float t = (float)p/(float)sps;
                            float realOmega = omegaMin+agent.x*(omegaMax-omegaMin);
                            float realPhi = agent.y*phiMax;
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
            auto dScores = new bc::vector<float>(solverParameterSetSinewaveFitting1.popSize,*ctx);
            auto dPopulation = new bc::vector<bc::float2_>(solverParameterSetSinewaveFitting1.popSize,*ctx);
            const float sumOfSquares =
                    bc::accumulate(dSignalData->begin(),dSignalData->end(),0.0f,
                                   accumSquared
                            ,*queue);

            bc::program program =
                    bc::program::create_with_source(sinewaveFitSource, *ctx);
            // compile the program
            try {
                program.build();
            } catch(bc::opencl_error &e) {
                std::cout << program.build_log() << std::endl;
            }
            auto kernel = bc::kernel(program,"calculateScoresOfPopulation");
            kernel.set_arg(0,*dSignalData);
            kernel.set_arg(1,*dPopulation);
            kernel.set_arg(2,*dScores);
            kernel.set_arg(3,(unsigned int)dPopulation->size());
            kernel.set_arg(4,(unsigned int)dSignalData->size());
            kernel.set_arg(5,sumOfSquares);
            kernel.set_arg(6,omegaMin);
            kernel.set_arg(7,omegaMax);
            kernel.set_arg(8,phiMax);
            kernel.set_arg(9,(unsigned int)sps);
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
                    solverParameterSetSinewaveFitting1, evaluatePopulationFitError
            );
        }
        void TearDown() override {
            delete solver;
        }
    };
}

#endif //MINUSDARWIN_SOLVERTEST_HPP
