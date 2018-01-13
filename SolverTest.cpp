//
// Created by dev on 1/13/2018.
//

#include "SolverTest.hpp"


namespace MinusDarwinTest {
    TEST_F(SolverWithSumFunction, TestScoreFunction) {
        ASSERT_FLOAT_EQ(solver->scoreFunction(2.0f, 3.0f), 5.0f);
    }
}
