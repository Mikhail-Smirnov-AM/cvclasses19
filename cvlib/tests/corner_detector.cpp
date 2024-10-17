/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    
    SECTION("flat image")
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar{127});
        std::vector<cv::KeyPoint> out;
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("edge image")
    {
        cv::Mat image_left(10, 5, CV_8UC1, cv::Scalar{50});
        cv::Mat image_right(10, 5, CV_8UC1, cv::Scalar{80});
        cv::Mat image_edge;
        cv::hconcat(image_left, image_right, image_edge);
        
		std::vector<cv::KeyPoint> out;
        fast->detect(image_edge, out);
        REQUIRE(out.empty());
    }

	SECTION("flat L-corner image")
    {
        cv::Mat image_lt(5, 5, CV_8UC1, cv::Scalar{50});
        cv::Mat image_rt(5, 5, CV_8UC1, cv::Scalar{120});
        cv::Mat image_lb(5, 5, CV_8UC1, cv::Scalar{120});
        cv::Mat image_rb(5, 5, CV_8UC1, cv::Scalar{120});
        
        cv::Mat image_top;
        cv::Mat image_bottom;
        cv::Mat image_L;
        cv::hconcat(image_lt, image_rt, image_top);
        cv::hconcat(image_lb, image_rb, image_bottom);
        cv::vconcat(image_top, image_bottom, image_L);

        std::vector<cv::KeyPoint> out;
        fast->detect(image_L, out);
        REQUIRE(out.size() == 0);
    }

	SECTION("L-corner image")
    {
        cv::Mat image_lt(5, 5, CV_8UC1, cv::Scalar{50});
        cv::Mat image_rt(5, 5, CV_8UC1, cv::Scalar{120});
        cv::Mat image_lb(5, 5, CV_8UC1, cv::Scalar{120});
        cv::Mat image_rb(5, 5, CV_8UC1, cv::Scalar{120});
        image_lt.at<unsigned char>(4, 4) = 90;

		cv::Mat image_top;
        cv::Mat image_bottom;
        cv::Mat image_L;
		cv::hconcat(image_lt, image_rt, image_top);
        cv::hconcat(image_lb, image_rb, image_bottom);
        cv::vconcat(image_top, image_bottom, image_L);

        std::vector<cv::KeyPoint> out;
        fast->detect(image_L, out);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0].pt.x == 4);
		REQUIRE(out[0].pt.y == 4);
    }

	SECTION("2 L-corner image")
    {
        cv::Mat image_lt(10, 10, CV_8UC1, cv::Scalar{50});
        cv::Mat image_rt(10, 10, CV_8UC1, cv::Scalar{120});
        cv::Mat image_lb(10, 10, CV_8UC1, cv::Scalar{120});
        cv::Mat image_rb(10, 10, CV_8UC1, cv::Scalar{120});
        image_lt.at<unsigned char>(9, 9) = 90;

        cv::Mat image_top;
        cv::Mat image_bottom;
        cv::Mat image_L1;
        cv::hconcat(image_lt, image_rt, image_top);
        cv::hconcat(image_lb, image_rb, image_bottom);
        cv::vconcat(image_top, image_bottom, image_L1);

		cv::Mat image_L2;
        cv::hconcat(image_L1, image_L1, image_L2);
        
        std::vector<cv::KeyPoint> out;
        fast->detect(image_L2, out);

        REQUIRE(out.size() == 2);
        REQUIRE(out[0].pt.x == 9);
        REQUIRE(out[0].pt.y == 9);
        REQUIRE(out[1].pt.x == 29);
        REQUIRE(out[1].pt.y == 9);

    }

	SECTION("T-corner image")
    {
        cv::Mat image_lt(10, 10, CV_8UC1, cv::Scalar{120});
        cv::Mat image_rt(10, 10, CV_8UC1, cv::Scalar{120});
        cv::Mat image_lb(10, 10, CV_8UC1, cv::Scalar{50});
        cv::Mat image_rb(10, 10, CV_8UC1, cv::Scalar{90});
        image_lb.at<unsigned char>(0, 9) = 70;
        image_rb.at<unsigned char>(0, 0) = 70;

        cv::Mat image_top;
        cv::Mat image_bottom;
        cv::Mat image_T;
        cv::hconcat(image_lt, image_rt, image_top);
        cv::hconcat(image_lb, image_rb, image_bottom);
        cv::vconcat(image_top, image_bottom, image_T);

        std::vector<cv::KeyPoint> out;
        fast->detect(image_T, out);

        REQUIRE(out.size() == 2);
        REQUIRE(out[0].pt.x == 9);
        REQUIRE(out[0].pt.y == 10);
        REQUIRE(out[1].pt.x == 10);
        REQUIRE(out[1].pt.y == 10);
    }

}
