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

    int t = 25;
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
			 
            // neighbour pixels on horizontal and vertical directions
            bool circle[16];
            circle[0]  = abs(image_mat.at<unsigned char>(std::max(i - 3, 0), j) - i_p) > t;
            circle[4]  = abs(image_mat.at<unsigned char>(i, std::min(j + 3, cols-1)) - i_p) > t;
            circle[8]  = abs(image_mat.at<unsigned char>(std::min(i + 3, rows-1), j) - i_p) > t;
            circle[12] = abs(image_mat.at<unsigned char>(i, std::max(j - 3, 0)) - i_p) > t;
            
            // check pixels on horizontal and vertical directions
            for (int k = 0; k < 16; k += 4)
				n += circle[k];

            if (n >= 3)
            {
				// all neighbour pixel
                circle[1]  = abs(image_mat.at<unsigned char>(std::max(i - 3, 0), std::min(j + 1, cols - 1)) - i_p) > t;
                circle[2]  = abs(image_mat.at<unsigned char>(std::max(i - 2, 0), std::min(j + 2, cols - 1)) - i_p) > t;
                circle[3]  = abs(image_mat.at<unsigned char>(std::max(i - 1, 0), std::min(j + 3, cols - 1)) - i_p) > t;
                circle[5]  = abs(image_mat.at<unsigned char>(std::min(i + 1, rows - 1), std::min(j + 3, cols - 1)) - i_p) > t;
                circle[6]  = abs(image_mat.at<unsigned char>(std::min(i + 2, rows - 1), std::min(j + 2, cols - 1)) - i_p) > t;
                circle[7]  = abs(image_mat.at<unsigned char>(std::min(i + 3, rows - 1), std::min(j + 1, cols - 1)) - i_p) > t;
                circle[9]  = abs(image_mat.at<unsigned char>(std::min(i + 3, rows - 1), std::max(j - 1, 0)) - i_p) > t;
                circle[10] = abs(image_mat.at<unsigned char>(std::min(i + 2, rows - 1), std::max(j - 2, 0)) - i_p) > t;
                circle[11] = abs(image_mat.at<unsigned char>(std::min(i + 1, rows - 1), std::max(j - 3, 0)) - i_p) > t;
                circle[13] = abs(image_mat.at<unsigned char>(std::max(i - 1, 0), std::max(j - 3, 0)) - i_p) > t;
                circle[14] = abs(image_mat.at<unsigned char>(std::max(i - 2, 0), std::max(j - 2, 0)) - i_p) > t;
                circle[15] = abs(image_mat.at<unsigned char>(std::max(i - 3, 0), std::max(j - 1, 0)) - i_p) > t;
                
                // check all pixels
                int N = 0;
                int k;
                int k_max = 16;
				
				// if 16 and 0 are pixels that satisfy condition then we should check if there is a sequence of N pixels in counterclockwise directions
				if (circle[0])
				{
                    k = 15;
                    while (circle[k] && k >= 0)
                    {
                        N++;
                        k--;
					}
                    k_max = k+1;
				}
				
				// check pixels in clockwise direction
				for (k = 0; k < k_max; k++)
				{
					// if circle[k] == 0 then N = 0 else N = N + 1
                    N = circle[k] * N + circle[k];
                    if (N == N_t || ((k_max - k) + N) < N_t)
                        break;
				}
                 
                // save key point
                if (N >= N_t)
                    keypoints.push_back(cv::KeyPoint((float)j, (float)i, 1, -1, 0, 0, 0));
            }
        }
    }
    
}

void corner_detector_fast::compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat image_mat = image.getMat();
    if (image_mat.channels() > 1)
        cv::cvtColor(image, image_mat, cv::COLOR_BGR2GRAY);
	
	std::srand(unsigned(std::time(0)));
    const int desc_length = 128;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

	int rows = image_mat.rows;
    int cols = image_mat.cols;

    int width = 20;
    int height = 20;

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        int pt_x = static_cast<int>(pt.pt.x);
        int pt_y = static_cast<int>(pt.pt.y);

        int tl_x = std::max(pt_x - width / 2, 0);
        int tl_y = std::max(pt_y - height / 2, 0);
        int roi_w = std::min(cols - pt_x, width);
        int roi_h = std::min(rows - pt_y, height);
		
		
		cv::Rect roi(tl_x, tl_y, roi_w, roi_h);
        cv::Mat img_roi = image_mat(roi);

        for (int i = 0; i < desc_length; ++i)
        {
            // first point
            int x1 = std::rand() % roi_w;
            int y1 = std::rand() % roi_h;
			
            // second point
            int x2 = std::rand() % roi_w;
            int y2 = std::rand() % roi_h;
    		
			*ptr = img_roi.at<unsigned short>(x1,y1) < img_roi.at<unsigned short>(x2,y2);
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
                                            cv::OutputArray descriptors, bool useProvidedKeypoints)
{
	if (!useProvidedKeypoints)
		detect(image, keypoints, mask);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
