/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <vector>
#include <iostream>

namespace cvlib
{
void motion_segmentation::apply(cv::InputArray _image, cv::OutputArray _fgmask, double alpha)
{
	// Mean algorithm imlementation
   
	
	// convert to grayscale
	cv::Mat I;
    cv::cvtColor(_image, I, cv::COLOR_BGR2GRAY);

	if (bg_model_.empty()) // first frame
	{
        I.copyTo(bg_model_);
        _fgmask.assign(cv::Mat::zeros(I.size(),I.type()));
    }
	else
	{
		// difference between frame and bg_model
        cv::Mat I_minus_bg;
        cv::absdiff(I, bg_model_, I_minus_bg);

        cv::Mat res = I_minus_bg > varThreshold_*cv::Mat::ones(I_minus_bg.size(), I_minus_bg.type());
        
        _fgmask.assign(res);

		// update model
        bg_model_ = (1 - alpha) * bg_model_ + alpha * I;
	}
}
} // namespace cvlib
