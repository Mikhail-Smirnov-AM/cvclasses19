/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (size_t i = 0; i < temp.size(); ++i)
        {
            temp[i] -= right[i];
        }
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

	double norm_l2() const
	{
        double res = 0.0;
		for (auto v : *this)
		{
            res = +std::pow(v,2);
		}
        return std::sqrt(res);
	}
};

void calculateDescriptor(const cv::Mat& image, int kernel_size, descriptor& descr)
{
    descr.clear();
    cv::Mat response;
    cv::Mat mean;
    cv::Mat dev;

    // create and use Gabor's filters
	for (auto gm = 0.25; gm <= 1.0; gm += 0.25)
	{
        for (auto lm = 1.0; lm <= 9.0; lm += 4.0)
        {
            for (auto th = 0.0; th <= 2*CV_PI; th += CV_PI / 4)
            {
                for (auto sig = 1; sig <= 11; sig += 5)
                {
                    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm);
                    cv::filter2D(image, response, CV_32F, kernel);
                    cv::meanStdDev(response, mean, dev);
                    descr.emplace_back(mean.at<double>(0));
                    descr.emplace_back(dev.at<double>(0));
                }
            }
        }
	}
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    cv::Mat imROI = image(roi);

	// round to nearest odd
	int size = std::min(roi.height, roi.width);
    const int kernel_size = (size % 2) ? size/2+1 : size/2+2; 
	
    descriptor reference;
    calculateDescriptor(imROI, kernel_size, reference);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);
    
    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    // move ROI smoothly pixel-by-pixel
    for (int i = 0; i < image.size().width-roi.width; ++i)
    {
        for (int j = 0; j < image.size().height - roi.height; ++j)
        {
            auto curROI = baseROI + cv::Point(i, j);
            calculateDescriptor(image(curROI), kernel_size, test);

            // norm L2 to compare test and reference
            res(curROI) = 255 * ((test - reference).norm_l2() <= eps);
        }
    }
    return res;
}
} // namespace cvlib
