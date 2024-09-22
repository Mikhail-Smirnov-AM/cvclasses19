/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>

namespace
{
void split_image(cv::Mat image, std::vector<cv::Mat>& regions, double stddev)
{
	cv::Mat mean;
    cv::Mat dev;
    cv::meanStdDev(image, mean, dev);

    if (dev.at<double>(0) <= stddev)
    {
        image.setTo(mean);
        regions.push_back(image); // save new region in vector
        return;
    }

    const auto width = image.cols;
    const auto height = image.rows;

    if (width > 1 && height > 1)
    {
        split_image(image(cv::Range(0, height / 2), cv::Range(0, width / 2)), regions, stddev);
        split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), regions, stddev);
        split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), regions, stddev);
        split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), regions, stddev);
    }

	// split row
	if (width > 1 && height == 1)
	{
        split_image(image(cv::Range(height / 2, height), cv::Range(0, width / 2)), regions, stddev);
        split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), regions, stddev);
	}

	// split column
	if (width == 1 && height > 1)
	{
        split_image(image(cv::Range(0, height / 2), cv::Range(width / 2, width)), regions, stddev);
        split_image(image(cv::Range(height / 2, height), cv::Range(width / 2, width)), regions, stddev);
	}
}

void merge_regions(std::vector<cv::Mat>& regions, double stddev)
{
    size_t L = regions.size();
	// boolean array that shows whether regions R_i and R_j has been merged
	bool** merged = new bool*[L];
	for (int i = 0; i < L; i++)
	{
        merged[i] = new bool[L];
        for (int j = 0; j < L; j++)
            merged[i][j] = false;
	}

	bool while_flag = true;
	// merge regions while there are regions that can be merged
    while (while_flag)
    {
        while_flag = false;
        for (int i = 0; i < L; i++)
        {
            for (int j = i+1; j < L; j++)
            {
				// if R_i and R_j have been merged, then skip them
                if (merged[i][j])
                    continue;
				// check predicate: mean values differ from each other less then stddev
                cv::Mat mean, mean_i, mean_j;
                cv::Mat dev;
                cv::Mat concat;
                cv::meanStdDev(regions[i], mean_i, dev);
                cv::meanStdDev(regions[j], mean_j, dev);
                cv::hconcat(mean_i, mean_j, concat);
                cv::meanStdDev(concat, mean, dev);
                // merging of regions
				if (dev.at<double>(0) < stddev)
                {
                    regions[i].setTo(mean_i);
                    regions[j].setTo(mean_i);
                    merged[i][j] = true;
					// if at least two regions have been merged, then it is necessary to check all regions again for the possibility of merging
                    while_flag = true;
                }
            }
        }
    }
    for (int i = 0; i < L; i++)
        delete[] merged[i];
    delete[] merged;
}

} // namespace

namespace cvlib
{
cv::Mat split_and_merge(const cv::Mat& image, double stddev)
{
    // split part
    std::vector<cv::Mat> regions;
    cv::Mat res = image;
    split_image(res, regions, stddev);
    
	// merge part
    merge_regions(regions, stddev);
    return res;
}
} // namespace cvlib
