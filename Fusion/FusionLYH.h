
#pragma once

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <windows.h> 
#include <GL/gl.h>
#include <GL/glu.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "gdal.h"
#include "gdalwarper.h"
#include "ogrsf_frmts.h"

#include "Fit.h"

class FusionLYH
{
public:
	FusionLYH(std::string sMSSImgPath, std::string sPANImgPath, std::string sPANAdjImgPath, 
		std::string sBuildingWeightPath, std::string sBuildingWeightDownSampPath,
		std::string sTxtReportPath,
		std::string sFusionResultImgPath,
		std::string sGDALDataFolder);
	void Run();

private:
	std::string m_sMSSImgPath;
	std::string m_sPANImgPath;
	std::string m_sPANAdjImgPath;
	std::string m_sBuildingWeightPath;
	std::string m_sBuildingWeightDownSampPath;
	std::string m_sTxtReportPath;

	std::string m_sFusionResultImgPath;
	std::string m_sGDALDataFolder;


	//geo ref
	double GEOpara[6];
	const char * PROdata = NULL;
	int valid_values_amounts=0;

	double GEOpara_Pan[6];
	const char * PROdata_Pan = NULL;


	//FFTML matching threshold
	double m_dResponseThNorm = 0.08;
	double m_dResponseThLow = 0.02;

	//surf matching threshold
	double m_dErrorThresholdPix = 50;

	int CPU_Surf_Registration_Poly(const char * Mss_path, const char *Pan_path, 
		std::vector<cv::Mat> &mul_src_16, std::vector<cv::Mat> &mul_regist_16, std::vector<cv::Mat> &pan_src_16);
	
	//image registration
	int CPU_FFTML_Registration(const char * Mss_path, const char *Pan_path,
		std::vector<cv::Mat> &mul_src_16, std::vector<cv::Mat> &mul_regist_16, std::vector<cv::Mat> &pan_src_16);

	int CheckMatchPointValid(cv::Mat mul_gray_resize_8, int block_x, int block_y, int x_img, int y_img);

	int ImgWarpPoly(std::vector<cv::Point2f> MSSHomo_points, std::vector<cv::Point2f> PANHomo_points,
		std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, std::vector<cv::Mat> &mul_regist_16);

	int ImgWarpPolySurf(std::vector<cv::Point2f> MSSHomo_points, std::vector<cv::Point2f> PANHomo_points,
		std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, std::vector<cv::Mat> &mul_regist_16);

	int ImgWarpHomo(cv::Mat homo,
		std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, std::vector<cv::Mat> &mul_regist_16);


	void FilterBadSurfMatches(std::vector<cv::Point2f> &p01, std::vector<cv::Point2f> &p02);


	void FilterLowResponseMatches(std::vector<cv::Point2f> &MSSHomo_points, std::vector<cv::Point2f> &PANHomo_points, 
		std::vector<double> response_static2);


	void ComputeBOMeanByBuildingWeight(GUInt16 * pBuffer, std::string sBuildingWeightPath,int nWidth, int nHeight, double & BMean, double & OMean);
	void ComputeBOAGByBuildingWeight(GUInt16 * pBuffer, std::string sBuildingWeightPath, int nWidth, int nHeight, double & BAG, double & OAG);
	cv::Mat CombinePANByBuildingWeight(cv::Mat PanB, cv::Mat PanO, std::string sBuildingWeightPath);


	//////////////////////////////////

	//fusion method mainly use this
	int AGSFIM(std::vector<cv::Mat> Mss, std::vector<cv::Mat> &Mss_registed, cv::Mat Pan);

	bool Mat2File_depth16(std::vector<cv::Mat> imgMat, char * fileName);

	char* findImageTypeGDAL(char *pDstImgFileName);

	int stretch_percent_16to8(const char *inFilename, std::vector<cv::Mat> &src_all_channels, cv::Mat &src8);

	void get_Mask(cv::Mat Mss_Mat, cv::Mat &Mask);

	bool refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints,
		const std::vector<cv::KeyPoint>& trainKeypoints, float reprojectionThreshold,
		std::vector<cv::DMatch>& matches, cv::Mat& homography);

	//fusion method mainly use this
	cv::Mat Get_Pan_ds(std::vector<cv::Mat> Mss_split, cv::Mat Pan);
	double* cal_mean_gradient(cv::Mat img);
	double cal_gradient(cv::Mat img);


	bool FFT_LYH(double RESPONSE_VALUE, cv::Mat block_Mat1_cpu, cv::Mat block_Mat2_cpu,
		cv::Point2f &p01, cv::Point2f &p02, double &response_static,
		int start_x1, int start_y1, int start_x2, int start_y2);


	bool MatchPointShwo(cv::Mat A, cv::Mat B, std::vector<cv::Point2f> one, std::vector<cv::Point2f> two,
		std::string name);
};

