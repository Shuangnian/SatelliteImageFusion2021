//#include "stdafx.h"
#include "fftml_cpu.h"

FFTML::FFTML()
{
}

FFTML::~FFTML()
{
}
//----------------------------------------------------------
// Recombinate image quaters
//----------------------------------------------------------
void FFTML::Recomb(cv::Mat &src, cv::Mat &dst)
{
	int cx = src.cols >> 1;
	int cy = src.rows >> 1;
	cv::Mat tmp;
	tmp.create(src.size(), src.type());
	src(cv::Rect(0, 0, cx, cy)).copyTo(tmp(cv::Rect(cx, cy, cx, cy)));
	src(cv::Rect(cx, cy, cx, cy)).copyTo(tmp(cv::Rect(0, 0, cx, cy)));
	src(cv::Rect(cx, 0, cx, cy)).copyTo(tmp(cv::Rect(0, cy, cx, cy)));
	src(cv::Rect(0, cy, cx, cy)).copyTo(tmp(cv::Rect(cx, 0, cx, cy)));
	dst = tmp;
}
//----------------------------------------------------------
// 2D Forward FFT
//----------------------------------------------------------
void FFTML::ForwardFFT(cv::Mat &Src, cv::Mat *FImg, bool do_recomb )
{
	int M = cv::getOptimalDFTSize(Src.rows);
	int N = cv::getOptimalDFTSize(Src.cols);
	cv::Mat padded;
	copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
	cv::Mat complexImg;
	merge(planes, 2, complexImg);
	dft(complexImg, complexImg);
	split(complexImg, planes);
	planes[0] = planes[0](cv::Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
	planes[1] = planes[1](cv::Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));
	if (do_recomb)
	{
		Recomb(planes[0], planes[0]);
		Recomb(planes[1], planes[1]);
	}
	planes[0] /= float(M*N);
	planes[1] /= float(M*N);
	FImg[0] = planes[0].clone();
	FImg[1] = planes[1].clone();
}
//----------------------------------------------------------
// 2D inverse FFT
//----------------------------------------------------------
void FFTML::InverseFFT(cv::Mat *FImg, cv::Mat &Dst, bool do_recomb )
{
	if (do_recomb)
	{
		Recomb(FImg[0], FImg[0]);
		Recomb(FImg[1], FImg[1]);
	}
	cv::Mat complexImg;
	merge(FImg, 2, complexImg);
	idft(complexImg, complexImg);
	split(complexImg, FImg);
	Dst = FImg[0].clone();
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
//void highpass(cv::Size & sz, cv::Mat& dst);
void FFTML::highpass(cv::Size & sz, cv::Mat& dst)
{
	cv::Mat a = cv::Mat(sz.height, 1, CV_32FC1);
	cv::Mat b = cv::Mat(1, sz.width, CV_32FC1);

	float step_y = CV_PI / sz.height;
	float val = -CV_PI*0.5;

	for (int i = 0; i < sz.height; ++i)
	{
		a.at<float>(i) = cos(val);
		val += step_y;
	}

	val = -CV_PI*0.5;
	float step_x = CV_PI / sz.width;
	for (int i = 0; i < sz.width; ++i)
	{
		b.at<float>(i) = cos(val);
		val += step_x;
	}

	cv::Mat tmp = a*b;
	dst = (1.0 - tmp).mul(2.0 - tmp);
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
float FFTML::logpolar(cv::Mat& src, cv::Mat& dst)
{
	float radii = src.cols;
	float angles = src.rows;
	cv::Point2f center(src.cols / 2, src.rows / 2);
	float d = norm(cv::Vec2f(src.cols - center.x, src.rows - center.y));
	float log_base = pow(10.0, log10(d) / radii);
	float d_theta = CV_PI / (float)angles;
	float theta = CV_PI / 2.0;
	float radius = 0;
	cv::Mat map_x(src.size(), CV_32FC1);
	cv::Mat map_y(src.size(), CV_32FC1);
	for (int i = 0; i < angles; ++i)
	{
		for (int j = 0; j < radii; ++j)
		{
			radius = pow(log_base, float(j));
			float x = radius * sin(theta) + center.x;
			float y = radius * cos(theta) + center.y;
			map_x.at<float>(i, j) = x;
			map_y.at<float>(i, j) = y;
		}
		theta += d_theta;
	}
	remap(src, dst, map_x, map_y, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	return log_base;
}
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
cv::RotatedRect FFTML::LogPolarFFTTemplateMatch(cv::Mat& im0, cv::Mat& im1, cv::Point2d &tr, double &response, double canny_threshold1, double canny_threshold2)
{
	// Accept 1 or 3 channel CV_8U, CV_32F or CV_64F images.
	CV_Assert((im0.type() == CV_8UC1) || (im0.type() == CV_8UC3) ||
		(im0.type() == CV_32FC1) || (im0.type() == CV_32FC3) ||
		(im0.type() == CV_64FC1) || (im0.type() == CV_64FC3));

	CV_Assert(im0.rows == im1.rows && im0.cols == im1.cols);

	CV_Assert(im0.channels() == 1 || im0.channels() == 3 || im0.channels() == 4);

	CV_Assert(im1.channels() == 1 || im1.channels() == 3 || im1.channels() == 4);

	cv::Mat im0_tmp = im0.clone();
	cv::Mat im1_tmp = im1.clone();
	if (im0.channels() == 3)
	{
		cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
	}

	if (im0.channels() == 4)
	{
		cvtColor(im0, im0, cv::COLOR_BGRA2GRAY);
	}

	if (im1.channels() == 3)
	{
		cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
	}

	if (im1.channels() == 4)
	{
		cvtColor(im1, im1, cv::COLOR_BGRA2GRAY);
	}

	if (im0.type() == CV_32FC1)
	{
		im0.convertTo(im0, CV_8UC1, 255.0);
	}

	if (im1.type() == CV_32FC1)
	{
		im1.convertTo(im1, CV_8UC1, 255.0);
	}

	if (im0.type() == CV_64FC1)
	{
		im0.convertTo(im0, CV_8UC1, 255.0);
	}

	if (im1.type() == CV_64FC1)
	{
		im1.convertTo(im1, CV_8UC1, 255.0);
	}


	// Canny(im0, im0, canny_threshold1, canny_threshold2); // you can change this
	// Canny(im1, im1, canny_threshold1, canny_threshold2);

	// Ensure both images are of CV_32FC1 type
	im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
	im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

	cv::Mat F0[2], F1[2];
	cv::Mat f0, f1;
	ForwardFFT(im0, F0);
	ForwardFFT(im1, F1);
	magnitude(F0[0], F0[1], f0);
	magnitude(F1[0], F1[1], f1);

	// Create filter 
	cv::Mat h;
	highpass(f0.size(), h);

	// Apply it in freq domain
	f0 = f0.mul(h);
	f1 = f1.mul(h);

	float log_base;
	cv::Mat f0lp, f1lp;

	log_base = logpolar(f0, f0lp);
	log_base = logpolar(f1, f1lp);

	// Find rotation and scale
	cv::Point2d rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);


	float angle = 180.0 * rotation_and_scale.y / f0lp.rows;
	float scale = pow(log_base, rotation_and_scale.x);
	// --------------
	if (scale > 1.8)
	{
		rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
		angle = -180.0 * rotation_and_scale.y / f0lp.rows;
		scale = 1.0 / pow(log_base, rotation_and_scale.x);
		if (scale > 1.8)
		{
			cout << "Images are not compatible. Scale change > 1.8" << endl;
			return cv::RotatedRect();
		}
	}
	// --------------
	if (angle < -90.0)
	{
		angle += 180.0;
	}
	else if (angle > 90.0)
	{
		angle -= 180.0;
	}
	//cout << "angle:"<<angle << endl;
	//cout << "scale:"<<scale << endl;
	// Now rotate and scale fragment back, then find translation
	cv::Mat rot_mat = getRotationMatrix2D(cv::Point(im1.cols / 2, im1.rows / 2), angle, 1.0 / scale);

	// rotate and scale
	cv::Mat im1_rs;
	warpAffine(im1, im1_rs, rot_mat, im1.size());
	//imshow("im1_rs", im1_rs);
	// find translation
	// cv::Point2d 
	//tr = cv::phaseCorrelate(im1_rs, im0);//原来的代码改为下面三行
	cv::InputArray window = cv::noArray();
	//double response = 0;
	tr = cv::phaseCorrelateRes(im1_rs, im0, window, &response);
	//cout << "response:" << response << endl;
	// compute rotated rectangle parameters
	cv::RotatedRect rr;
	//rr.size
	rr.center = tr + cv::Point2d(im0.cols / 2, im0.rows / 2);
	rr.angle = -angle;
	rr.size.width = im1.cols / scale;
	rr.size.height = im1.rows / scale;

	im0 = im0_tmp.clone();
	im1 = im1_tmp.clone();

	return rr;
}
bool FFTML::MatchPointShwo(cv::Mat A, cv::Mat B, vector<cv::Point2f> one, vector<cv::Point2f> two, string name)
{
	IplImage *dst_big;
	CvRect rect1 = cvRect(0, 0, A.cols, A.rows);//两个ROI区域
	CvRect rect2 = cvRect(A.cols, 0, B.cols, B.rows);
	IplImage *img1 = cvCloneImage(&(IplImage)A);
	IplImage *img2 = cvCloneImage(&(IplImage)B);
	int cols_big, rows_big;
	rows_big = A.rows > B.rows ? A.rows : B.rows;
	cols_big = A.cols + B.cols;
	dst_big = cvCreateImage(cvSize(cols_big, rows_big), IPL_DEPTH_8U, 3);
	cvSetImageROI(dst_big, rect1);//设置ROI
	cvCopy(img1, dst_big);
	cvSetImageROI(dst_big, rect2);
	cvCopy(img2, dst_big);
	cvResetImageROI(dst_big);//释放ROI
	cv::Mat showimg = dst_big;
	int line_num = one.size();
	cv::RNG rng(0xFFFFFFFF);
	for (int i = 0; i < line_num; i++)
	{
		cvLine(dst_big, cvPoint(one[i].x, one[i].y), cvPoint(two[i].x + A.cols, two[i].y), CV_RGB(255, 0, 0), 3, 3);//只能画整数坐标的直线，像素精度
	}
	IplImage *temp_img = cvCreateImage(cvSize(cols_big / 2, rows_big / 2), dst_big->depth, 3);
	cvResize(dst_big, temp_img);
	IplImage *dst = cvCreateImage(cvSize(cols_big / 2, rows_big / 2), temp_img->depth, 3);
	cvCopy(temp_img, dst);
	cvNamedWindow(name.c_str(), 0);
	//cvResizeWindow("Display", 500, 500); //创建一个500*500大小的窗口
	cvShowImage(name.c_str(), dst);//显示合并后的大图
	cvWaitKey();

	cvReleaseImage(&img1);//释放图像空间
	cvReleaseImage(&img2);
	cvReleaseImage(&temp_img);
	cvReleaseImage(&dst);
	cvReleaseImage(&dst_big);

	return true;
}
int FFTML::FFT_Registration(cv::Mat mul_src_8, cv::Mat pan_src_8, cv::Mat &result, vector<cv::Point2f> &_all_p01, vector<cv::Point2f> &_all_p02,cv::Mat &homo)
{
	if (mul_src_8.empty() || pan_src_8.empty())
		return 0;
	int nBlockSize = 200;
	int img_height = pan_src_8.rows;
	int img_width = pan_src_8.cols;
	cv::Mat mul_src_8_gray, pan_src_8_gray;
	//GpuMat gpu_mul_src_8(mul_src_8_3);
	//GpuMat gpu_mul_gray_resize_8;
	//GpuMat gpu_mul_src_8_3_channel;
	cvtColor(mul_src_8, mul_src_8_gray, CV_BGR2GRAY);
	cvtColor(pan_src_8, pan_src_8_gray, CV_BGR2GRAY);
	//vector<cv::Point2f> _all_p01, _all_p02;
	vector<double> response_static;
	//map
	for (int y = 0; y < pan_src_8.rows; y += nBlockSize)
	{
		for (int x = 0; x < pan_src_8.cols; x += nBlockSize)
		{
			//int nXBK = img_width;//nBlockSize;
			//int nYBK = img_height;// nBlockSize;
			int nXBK = nBlockSize;
			int nYBK =  nBlockSize;
			if (y + nBlockSize > img_height)//最下面的剩余块
				nYBK = img_height - y;
			if (x + nBlockSize > img_width)//最右侧的剩余块
				nXBK = img_width - x;

			cv::Mat block_Mat1_cpu = mul_src_8_gray(cv::Rect(x, y, nXBK, nYBK)).clone();
			cv::Mat block_Mat2_cpu = pan_src_8_gray(cv::Rect(x, y, nXBK, nYBK)).clone();
			cv::Point2d tr;
			//cout << "行列为：" << x << "  , " << y << endl;
			double response;
			cv::RotatedRect rr = LogPolarFFTTemplateMatch(block_Mat1_cpu, block_Mat2_cpu, tr, response, 100, 200);
			if (response >= 0.2)
			{
				cv::Point2d p(block_Mat2_cpu.cols / 2 + x, block_Mat2_cpu.rows / 2 + y);
				rr.center.x += x;
				rr.center.y += y;
				//rr.center.x *= pan_src_8.cols / mul_src_8.cols;
				//rr.center.y *= pan_src_8.rows / mul_src_8.rows;
				_all_p01.push_back(rr.center);
				_all_p02.push_back(p);
				response_static.push_back(response);
			}
		}
	}
	sort(response_static.begin(), response_static.end());
	int numofpoint = response_static.size();
	//for (int numofpoint = 0;i < 3;i--)
	//{

	//}
	//MatchPointShwo(mul_src_8, pan_src_8, _all_p01, _all_p02, "fft_try");

	try
	{

		//能否替换为 透视变换？LYH 20210327
		homo = findHomography(_all_p01, _all_p02, CV_RANSAC);

		//homo = cv::getPerspectiveTransform(_all_p01, _all_p02);

		cv::warpPerspective(mul_src_8, result, homo, mul_src_8.size());
		//////将中间的密集匹配结果保存，用于分析   20181022   yh
		//cv::Mat test_srcImg1 = mul_src_8;
		//cv::Mat test_srcImg2 = pan_src_8;

		//cv::Mat img_merge(test_srcImg1.rows, test_srcImg1.cols + test_srcImg2.cols , test_srcImg1.type());
		//test_srcImg1.colRange(0, test_srcImg1.cols).copyTo(img_merge.colRange(0, test_srcImg1.cols));
		//test_srcImg2.colRange(0, test_srcImg2.cols).copyTo(img_merge.colRange(test_srcImg1.cols , img_merge.cols));
		//for (int i = 0; i < _all_p01.size(); i++)
		//{
		//	//cv::line(img_merge, _match_p01[i], cv::Point2f(_match_p02[i].x + test_srcImg1.cols, _match_p02[i].y), cv::Scalar(0, 255, 255));
		//	circle(img_merge, _all_p01[i], 2, cv::Scalar(0, 255, 0));
		//	circle(img_merge, cv::Point2f(_all_p02[i].x + test_srcImg1.cols, _all_p02[i].y), 2, cv::Scalar(0, 255, 0));
		//	putText(img_merge, to_string(i), _all_p01[i], cv::FONT_HERSHEY_COMPLEX,2,cv::Scalar(0, 255, 255) );
		//	putText(img_merge, to_string(i), cv::Point2f(_all_p02[i].x + test_srcImg1.cols, _all_p02[i].y), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 255, 255));
		//}
		//string sub_img_merge = "data/img_merge.bmp";
		//imwrite(sub_img_merge, img_merge);
		//////以上
		return 1;
	}
	catch (const std::exception&)
	{
		return 0;
	}

}