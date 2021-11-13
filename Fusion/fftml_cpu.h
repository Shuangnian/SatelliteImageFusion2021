#pragma once
#ifndef fftml_HPP_
#define fftml_HPP_
#ifdef __cplusplus
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
//using namespace cv;
class FFTML
{

public:
	FFTML();
	~FFTML();
	int FFT_Registration(cv::Mat mul_src_8, cv::Mat pan_src_8, cv::Mat &result, vector<cv::Point2f> &_all_p01, vector<cv::Point2f> &_all_p02,cv::Mat &homo);
	bool MatchPointShwo(cv::Mat A, cv::Mat B, vector<cv::Point2f> one, vector<cv::Point2f> two, string name);
	cv::RotatedRect LogPolarFFTTemplateMatch(cv::Mat& im0, cv::Mat& im1, cv::Point2d &tr, double &response, double canny_threshold1, double canny_threshold2);
private:
	void Recomb(cv::Mat &src, cv::Mat &dst);
	void ForwardFFT(cv::Mat &Src, cv::Mat *FImg, bool do_recomb = true);
	void InverseFFT(cv::Mat *FImg, cv::Mat &Dst, bool do_recomb = true);
	void highpass(cv::Size & sz, cv::Mat& dst);
	float logpolar(cv::Mat& src, cv::Mat& dst);
	//void highpass(cv::Size sz, cv::Mat& dst)
};

#endif
#endif