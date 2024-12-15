/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace cvlib
{
void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    for (int i = 0; i < q_desc.rows; ++i)
    {
        int min_dist_j = 0;
        float min_dist = FLT_MAX;
        float dist = 0;
      
		for (int j = 0; j < t_desc.rows; j++)
        {
            dist = 0;
            for (int k = 0; k < q_desc.cols; k++)
                dist += (float)pow(t_desc.at<int>(j,k)-q_desc.at<int>(i,k),2);
            if (dist/min_dist < ratio_)
            {
				min_dist = dist;
                min_dist_j = j;
            }
        }
        matches[i].emplace_back(i, min_dist_j, min_dist);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float /*maxDistance*/,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib
