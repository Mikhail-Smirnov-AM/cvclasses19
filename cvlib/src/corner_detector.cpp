/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{

// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

void corner_detector_fast::detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray /*mask = cv::noArray()*/)
{
    
    cv::Mat image_mat = image.getMat();
    if (image_mat.channels() > 1)
		cv::cvtColor(image, image_mat, cv::COLOR_BGR2GRAY);

    int t = 10;
    int N_t = 12;
    keypoints.clear();

    int rows = image_mat.rows;
    int cols = image_mat.cols;
    
	for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int i_p = image_mat.at<unsigned char>(i, j);
            int n = 0;

            // neighbour pixels
            std::pair<int, int> circle[16];
            circle[0] = std::pair<int, int>(std::max(i - 3, 0), j);
            circle[1] = std::pair<int, int>(std::max(i - 3, 0), std::min(j + 1, cols-1));
            circle[2] = std::pair<int, int>(std::max(i - 2, 0), std::min(j + 2, cols-1));
            circle[3] = std::pair<int, int>(std::max(i - 1, 0), std::min(j + 3, cols-1));
            circle[4] = std::pair<int, int>(i, std::min(j + 3, cols-1));
            circle[5] = std::pair<int, int>(std::min(i + 1, rows-1), std::min(j + 3, cols-1));
            circle[6] = std::pair<int, int>(std::min(i + 2, rows-1), std::min(j + 2, cols-1));
            circle[7] = std::pair<int, int>(std::min(i + 3, rows-1), std::min(j + 1, cols-1));
            circle[8] = std::pair<int, int>(std::min(i + 3, rows-1), j);
            circle[9] = std::pair<int, int>(std::min(i + 3, rows-1), std::max(j - 1, 0));
            circle[10] = std::pair<int, int>(std::min(i + 2, rows-1), std::max(j - 2, 0));
            circle[11] = std::pair<int, int>(std::min(i + 1, rows-1), std::max(j - 3, 0));
            circle[12] = std::pair<int, int>(i, std::max(j - 3, 0));
            circle[13] = std::pair<int, int>(std::max(i - 1, 0), std::max(j - 3, 0));
            circle[14] = std::pair<int, int>(std::max(i - 2, 0), std::max(j - 2, 0));
            circle[15] = std::pair<int, int>(std::max(i - 3, 0), std::max(j - 1, 0));

            // check pixels on horizontal and vertical directions
            for (int k = 0; k < 16; k += 4)
                if (abs(image_mat.at<unsigned char>(circle[k].first, circle[k].second) - i_p) > t)
                    n++;

            if (n >= 3)
            {
                
                // check all pixels
                int N = 0;
                for (int k = 0; k < 16; k++)
                {
                    if (abs(image_mat.at<unsigned char>(circle[k].first, circle[k].second) - i_p) > t)
                        N++;
                    else
                        N = 0;

					if (N == N_t || ((16-k) + N) < N_t)
                        break;
                }

                // save key point
                if (N >= N_t)
                    keypoints.push_back(cv::KeyPoint((float)j, (float)i, 1));
            }
        }
    }
    
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
    {
        // \todo implement me
    }
} // namespace cvlib
