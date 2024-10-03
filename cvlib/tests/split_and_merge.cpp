/* Split and merge segmentation algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("constant image", "[split_and_merge]")
{
    const cv::Mat image(100, 100, CV_8UC1, cv::Scalar{15});

    const auto res = split_and_merge(image, 1);
    REQUIRE(image.size() == res.size());
    REQUIRE(image.type() == res.type());
    REQUIRE(cv::Scalar(15) == cv::mean(res));
}

TEST_CASE("simple regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 2, 2, 
														   2, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 
												 2, 3);
        auto res = split_and_merge(image, 10);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));

        res = split_and_merge(image, 1);
        REQUIRE(0 == cv::countNonZero(image - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 0, 10, 10, 
														   4, 2, 2, 
														   4, 2, 2);
        cv::Mat image = (cv::Mat_<char>(3, 3) << 0, 10, 10, 
												 4, 1, 2, 
												 5, 2, 1);
        auto res = split_and_merge(image, 1);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));
    }
}

TEST_CASE("compex regions", "[split_and_merge]")
{
    SECTION("2x2")
    {
        const cv::Mat reference = (cv::Mat_<char>(2, 2) << 0, 0, 
														   5, 2);
        cv::Mat image = (cv::Mat_<char>(2, 2) << 0, 1, 
											     5, 2);
        auto res = split_and_merge(image, 1);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));
    }

    SECTION("3x3")
    {
        const cv::Mat reference = (cv::Mat_<char>(3, 3) << 0, 2, 2, 
														   0, 0, 4, 
														   4, 7, 7);
        cv::Mat image = (cv::Mat_<char>(3, 3) << 0, 1, 2, 
												 1, 0, 4, 
												 5, 6, 7);
        auto res = split_and_merge(image, 1);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));
    }

    SECTION("4x4")
    {
        const cv::Mat reference = (cv::Mat_<char>(4, 4) << 0, 0, 8, 13, 
														   0, 0, 8, 13, 
														   0, 0, 100, 100, 
														   0, 0, 100, 94);
        cv::Mat image = (cv::Mat_<char>(4, 4) << 0, 1, 8, 13, 
												 1, 0, 10, 15, 
												 2, 2, 100, 98, 
												 0, 0, 97, 94);
        auto res = split_and_merge(image, 2);
        REQUIRE(image.size() == res.size());
        REQUIRE(image.type() == res.type());
        REQUIRE(0 == cv::countNonZero(reference - res));
    }
}
