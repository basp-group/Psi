#include <catch.hpp>
#include <Eigen/Core>
#include "psi/sort_utils.h"

TEST_CASE("Test bound functions defined in sort_utils.h "){
    Eigen::VectorXd x(4); // to be replaced by psi typedefs?
    x << 0., 1., 2., 3.;  //!\\ 0-indexing

    SECTION("Test lower_bound_index", "[sort_utils][lower_bound_index]") {
        auto lm1 = psi::lower_bound_index(x.data(),x.data() + x.size(), -1.); // index of the first element >= -1.
        CHECK(lm1 == 0);

        auto l0 = psi::lower_bound_index(x.data(),x.data() + x.size(), 0.); // index of the first element >= 0.
        CHECK(l0 == 0);

        auto l2 = psi::lower_bound_index(x.data(),x.data() + x.size(), 2.); // index of the last element >= 2.
        CHECK(l2 == 2);

        auto l4 = psi::lower_bound_index(x.data(),x.data() + x.size(), 4.); // index of the first element >= 4.
        CHECK(l4 == -1); // no element >= 4, should return -1
    }

    SECTION("Test upper_bound_index", "[sort_utils][upper_bound_index]") {
        auto um1 = psi::upper_bound_index(x.data(),x.data() + x.size(), -1.); // index of the last element <= -1.
        CHECK(um1 == -1); // no element <= -1, should return -1

        auto u0 = psi::upper_bound_index(x.data(),x.data() + x.size(), 0.); // index of the last element <= 0.
        CHECK(u0 == 0);

        auto u2 = psi::upper_bound_index(x.data(),x.data() + x.size(), 2.); // index of the last element <= 2.
        CHECK(u2 == 2);
        
        auto u4 = psi::upper_bound_index(x.data(),x.data() + x.size(), 4.); //  index of the last element <= 4.
        CHECK(u4 == 3);
    }

    SECTION("Test strict_upper_bound_index", "[sort_utils][strict_upper_bound_index]") {
        auto um1 = psi::strict_upper_bound_index(x.data(),x.data() + x.size(), -1.); // index of the last element < -1.
        CHECK(um1 == -1); // no element < -1, should return -1

        auto u0 = psi::strict_upper_bound_index(x.data(),x.data() + x.size(), 0.); // index of the last element < 0.
        CHECK(u0 == -1); // no element < 0, should return -1

        auto u2 = psi::strict_upper_bound_index(x.data(),x.data() + x.size(), 2.); // index of the last element < 2.
        CHECK(u2 == 1);
        
        auto u4 = psi::strict_upper_bound_index(x.data(),x.data() + x.size(), 4.); //  index of the last element < 4.
        CHECK(u4 == 3);
    }
}

TEST_CASE("Test sort_indices function", "[sort_utils][sort_indices]"){
    Eigen::VectorXd x(4); // to be replaced by psi typedefs?
    x << 3., 2., 1., 0.;  //!\\ 0-indexing

    Eigen::Matrix<size_t, Eigen::Dynamic, 1> idx(x.size());
    psi::sort_indices(x, idx);

    for(int k = 0; k < x.size(); ++k){
        CHECK(idx(k) == (x.size() - 1 - k));
    }
}

TEST_CASE("Test find_if_indices function", "[sort_utils][find_if_indices]"){
    Eigen::VectorXd x(4); // to be replaced by psi typedefs?
    x << 0., 1., 2., 3.;  //!\\ 0-indexing

    auto idx = psi::find_if_index(x.data(), x.data() + x.size(), [](double y) {return (y > 2.);});

    CHECK(idx == 3);
}
