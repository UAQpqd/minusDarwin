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
            std::function<float(std::vector<float>)> sum = [](std::vector<float> v) { return v.at(0) + v.at(1); };
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    2, 20, 4, MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Best, 1,
                    0.05f, 0.7f, 0.7f
            };
            solver = new MinusDarwin::Solver(
                    solverParameterSet, sum
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
            const float time = 1.0f;
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
            std::function<float(std::vector<float>)> fitError =
                    [signalData, sumOfSquares, sps, a,
                            omegaMin, omegaMax, phiMax](std::vector<float> v) -> float {
                        float error = 0.0f;
                        for (size_t p = 0; p < signalData.size(); p++) {
                            float t = (float)p/(float)sps;
                            float realOmega = omegaMin+v.at(0)*(omegaMax-omegaMin);
                            float realPhi = v.at(1)*phiMax;
                            float estimated =
                                    a*sin(realOmega*t+realPhi);
                            error += pow(estimated-signalData.at(p),2.0f);
                        }
                        return error/sumOfSquares;
                    };
            MinusDarwin::SolverParameterSet solverParameterSet = {
                    2, 1200, 20, MinusDarwin::GoalFunction::EpsilonReached,
                    MinusDarwin::CrossoverMode::Best, 1,
                    0.0005f, 0.7f, 0.7f
            };
            solver = new MinusDarwin::Solver(
                    solverParameterSet, fitError, bc::system::default_device()
            );
        }
        void TearDown() override {
            delete solver;
        }
    };
}

#endif //MINUSDARWIN_SOLVERTEST_HPP
