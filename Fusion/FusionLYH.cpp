
#define _CRT_SECURE_NO_WARNINGS


#include "FusionLYH.h"
#include "fftml_cpu.h"


FusionLYH::FusionLYH(std::string sMSSImgPath, std::string sPANImgPath, std::string sPANAdjImgPath, 
	std::string sBuildingWeightPath, std::string sBuildingWeightDownSampPath,
	std::string sTxtReportPath,
	std::string sFusionResultImgPath,
	std::string sGDALDataFolder)
{
	m_sMSSImgPath = sMSSImgPath;
	m_sPANImgPath = sPANImgPath;
	m_sPANAdjImgPath = sPANAdjImgPath;
	m_sBuildingWeightPath = sBuildingWeightPath;
	m_sBuildingWeightDownSampPath = sBuildingWeightDownSampPath;
	m_sTxtReportPath = sTxtReportPath;

	m_sFusionResultImgPath = sFusionResultImgPath;
	m_sGDALDataFolder = sGDALDataFolder;
	
}

void FusionLYH::Run()
{
	std::vector<cv::Mat> regis_Mss_Mat;
	std::vector<cv::Mat> mul_src_16;
	std::vector<cv::Mat> pan_src_16;


	clock_t startTime, endTime;
	startTime = clock();

	int regis_result = CPU_FFTML_Registration(m_sMSSImgPath.c_str(), m_sPANImgPath.c_str(), mul_src_16, regis_Mss_Mat, pan_src_16);
	if (regis_result != 1)
	{
		if (regis_result==2)
		{
			pan_src_16.clear();
			mul_src_16.clear();
			regis_Mss_Mat.clear();

			regis_result = CPU_Surf_Registration_Poly(m_sMSSImgPath.c_str(), m_sPANImgPath.c_str(), mul_src_16, regis_Mss_Mat, pan_src_16);
			if (regis_result!=1)
			{
				std::cout << "surf also fail" << std::endl;
				return;
			}
		}
		else
		{
			std::cout << "regis_result error" << std::endl;
			return ;
		}

	}

	endTime = clock();
	double timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << " regis time: " << timeCost << " s" << std::endl;




	clock_t startTime2, endTime2;
	startTime2 = clock();

	int fusion_result_signal = AGSFIM(mul_src_16, regis_Mss_Mat, pan_src_16[0]);
	if (fusion_result_signal != 1)
	{
		std::cout << "fusion_result_signal error" << std::endl;
		return;
	}

	endTime2 = clock();
	double timeCost2 = (double)(endTime2 - startTime2) / CLOCKS_PER_SEC;
	std::cout << " fusion time: " << timeCost2 << " s" << std::endl;




	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	CPLSetConfigOption("GDAL_DATA", m_sGDALDataFolder.c_str());
	GDALAllRegister();

	GDALDataset *poIn = (GDALDataset *)GDALOpen(m_sPANImgPath.c_str(), GA_ReadOnly);
	PROdata_Pan = NULL;
	poIn->GetGeoTransform(GEOpara_Pan);
	PROdata_Pan = poIn->GetProjectionRef();

	char * pcTempOutputFilePath = new char [m_sFusionResultImgPath.length()];
	strcpy(pcTempOutputFilePath, m_sFusionResultImgPath.c_str());

	Mat2File_depth16(regis_Mss_Mat, pcTempOutputFilePath);//已进行过并行优化了

	std::cout << "融合完成" << std::endl;

	delete[] pcTempOutputFilePath;
}

//FFTML matching 
int FusionLYH::CPU_FFTML_Registration(const char * Mss_path, const char *Pan_path,
	std::vector<cv::Mat> &mul_src_16, std::vector<cv::Mat> &mul_regist_16, std::vector<cv::Mat> &pan_src_16)
{
	cv::Mat mul_src_8, pan_src_8;
	std::cout << "FFTML matching" << std::endl;

	clock_t startTime, endTime;
	startTime = clock();

	stretch_percent_16to8(Mss_path, mul_src_16, mul_src_8);
	stretch_percent_16to8(Pan_path, pan_src_16, pan_src_8);


	endTime = clock();
	double timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	cout << " stretch 16 to 8 time: " << timeCost << " s" << endl;

	cv::Mat mask;
	get_Mask(pan_src_16[0], mask);
	if (mul_src_16[0].empty() || mul_src_8.empty() || pan_src_16[0].empty() || pan_src_8.empty())
	{
		std::cout << "empty" << std::endl;
		return 0;
	}

	//convert to gray
	cv::Mat mul_src_8_gray;
	cv::Mat mul_gray_resize_8;
	cvtColor(mul_src_8, mul_src_8_gray, CV_BGR2GRAY);
	mul_src_8.release();

	//upsamp to the same size with pan image
	cv::resize(mul_src_8_gray, mul_gray_resize_8, pan_src_16[0].size());
	mul_src_8_gray.release();

	CV_Assert(!mul_gray_resize_8.empty());

	//size after upsamp
	int nImgwidth = mul_gray_resize_8.cols;
	int nImgheight = mul_gray_resize_8.rows;

	//matching point number
	int rownum = 10;
	int colnum = 10;
	std::vector<cv::Point2f> Match_points;
	for (int i = 0; i < rownum; i++)
	{
		for (int j = 0; j < colnum; j++)
		{
			cv::Point2f p_herer;
			p_herer.x = nImgwidth*(j + 1) / (colnum + 1);
			p_herer.y = nImgheight*(i + 1) / (rownum + 1);
			Match_points.push_back(p_herer);
		}
	}

	//image block size
	int block_x = mul_gray_resize_8.cols / 9;
	int block_y = mul_gray_resize_8.rows / 9;

	std::vector<cv::Point2f> MSSHomo_points;
	std::vector<cv::Point2f> PANHomo_points;

	std::vector<cv::Point2f> MSSHomo_points2;
	std::vector<cv::Point2f> PANHomo_points2;
	std::vector<double> response_static2;

	std::vector<double> response_static;
	std::vector<double> response_static_Para(Match_points.size());
	std::vector<cv::Point2f> img_points(Match_points.size());
	std::vector<cv::Point2f> imggoogle_points(Match_points.size());

	std::vector<bool> vIsMatched(Match_points.size());
	std::vector<bool> vDoMatchOrNot(Match_points.size());
	std::vector<cv::Mat> vRPCOrthoResampImgBlock(Match_points.size());
	std::vector<cv::Mat> vRefImgBlock(Match_points.size());
	std::vector<int> vx_start_fft(Match_points.size());
	std::vector<int> vy_start_fft(Match_points.size());
	std::vector<int> vx_start_fft_gool(Match_points.size());
	std::vector<int> vy_start_fft_gool(Match_points.size());

	for (int epi_point_num = 0; epi_point_num < Match_points.size(); epi_point_num++)
	{
		vDoMatchOrNot[epi_point_num] = true;
		vIsMatched[epi_point_num] = false;

		double x_f, y_f, x_img, y_img, x_gool, y_gool;
		double z_f = 0;

		x_img = Match_points[epi_point_num].x; 
		y_img = Match_points[epi_point_num].y; 

		x_gool = Match_points[epi_point_num].x;
		y_gool = Match_points[epi_point_num].y;

		int block_x_this = block_x;
		int block_y_this = block_y;
		if (x_img + block_x_this / 2 > mul_gray_resize_8.cols)
			block_x_this = 2 * (mul_gray_resize_8.cols - x_img - 1);
		if (x_img - block_x_this / 2 < 0)
			block_x_this = 2 * (x_img - 1);

		if (y_img + block_y_this / 2 > mul_gray_resize_8.rows)
			block_y_this = 2 * (mul_gray_resize_8.rows - y_img - 1);

		if (y_img - block_y_this / 2 < 0)
			block_y_this = 2 * (y_img - 1);

		int nMatchPointValidLvl = CheckMatchPointValid(mul_gray_resize_8, block_x, block_y, x_img, y_img);
		if (nMatchPointValidLvl<2)
		{
			vDoMatchOrNot[epi_point_num] = false;
			continue;
		}

		int x_gool_start = x_gool;
		int y_gool_start = y_gool;


		int x_start_fft = x_img - block_x_this / 2;
		int y_start_fft = y_img - block_y_this / 2;
		int x_start_fft_gool = x_gool_start - block_x_this / 2;
		int y_start_fft_gool = y_gool_start - block_y_this / 2;

		cv::Mat imgmat_block = mul_gray_resize_8(cv::Rect(x_start_fft, y_start_fft, block_x_this, block_y_this));//.clone();
		cv::Mat googlemat_block = pan_src_8(cv::Rect(x_start_fft_gool, y_start_fft_gool, block_x_this, block_y_this));//.clone();

		vRPCOrthoResampImgBlock[epi_point_num] = imgmat_block;
		vRefImgBlock[epi_point_num] = googlemat_block;

		vx_start_fft[epi_point_num] = x_start_fft;
		vy_start_fft[epi_point_num] = y_start_fft;
		vx_start_fft_gool[epi_point_num] = x_start_fft_gool;
		vy_start_fft_gool[epi_point_num] = y_start_fft_gool;

	}


#pragma omp parallel for
	for (int epi_point_num = 0; epi_point_num < Match_points.size(); epi_point_num++)
	{
		if (!vDoMatchOrNot[epi_point_num])
		{
			continue;
		}
		else
		{
			double yuzhi = m_dResponseThNorm;//0.08 threshold for FFTML matching
			if (FFT_LYH(yuzhi, vRefImgBlock[epi_point_num], vRPCOrthoResampImgBlock[epi_point_num],
				imggoogle_points[epi_point_num], img_points[epi_point_num], response_static_Para[epi_point_num],
				vx_start_fft_gool[epi_point_num], vy_start_fft_gool[epi_point_num],
				vx_start_fft[epi_point_num], vy_start_fft[epi_point_num]))
			{
				vIsMatched[epi_point_num] = true;
			}
		}
	}

	for (int i = 0; i<vIsMatched.size(); i++)
	{
		if (vIsMatched[i])
		{
			cv::Point2f mss_pt;
			mss_pt.x = img_points[i].x;
			mss_pt.y = img_points[i].y;

			MSSHomo_points.push_back(mss_pt);
			MSSHomo_points2.push_back(mss_pt);

			cv::Point2f pan_pt;
			pan_pt.x = imggoogle_points[i].x;
			pan_pt.y = imggoogle_points[i].y;

			PANHomo_points.push_back(pan_pt);
			PANHomo_points2.push_back(pan_pt);

			response_static.push_back(response_static_Para[i]);
			response_static2.push_back(response_static_Para[i]);

		}
		else if(response_static_Para[i] > 0.02)
		{
			cv::Point2f mss_pt;
			mss_pt.x = img_points[i].x;
			mss_pt.y = img_points[i].y;

			MSSHomo_points2.push_back(mss_pt);

			cv::Point2f pan_pt;
			pan_pt.x = imggoogle_points[i].x;
			pan_pt.y = imggoogle_points[i].y;

			PANHomo_points2.push_back(pan_pt);

			response_static2.push_back(response_static_Para[i]);
		}

	}

	mul_gray_resize_8.release();
	pan_src_8.release();


	//use surf instead
	bool bUseLowerResponse = false;
	if (MSSHomo_points.size()<10)
	{
		if (MSSHomo_points2.size()<10)
		{
			std::cout << "SURF" << std::endl;
			return 2;
		}
		else
		{
			std::cout << "SURF" << std::endl;
			return 2;
		}
	}

	try
	{
		if (!bUseLowerResponse)
		{
			if (ImgWarpPoly(MSSHomo_points, PANHomo_points, mul_src_16, pan_src_16[0].size(), mul_regist_16) != 0)
			{
				std::cout << "error" << std::endl;
				return 0;
			}
		}
		else
		{
			std::cout << "use bad matching points" << std::endl;

			//filter out some points
			FilterLowResponseMatches(MSSHomo_points2, PANHomo_points2,response_static2);

			if (ImgWarpPoly(MSSHomo_points2, PANHomo_points2, mul_src_16, pan_src_16[0].size(), mul_regist_16) != 0)
			{
				std::cout << "error" << std::endl;
				return 0;
			}
		}

	}
	catch (...)
	{
		std::cout << "error" << std::endl;
		return 0;
	}

	std::cout << "regis finished!" << std::endl;
	

	return 1;

}


void FusionLYH::FilterLowResponseMatches(std::vector<cv::Point2f> &MSSHomo_points, std::vector<cv::Point2f> &PANHomo_points,
	std::vector<double> response_static2)
{
	if (response_static2.size()<=16)
	{
		return;
	}
	else
	{
		std::vector<cv::Point2f> MSSHomo_pointsCopy(MSSHomo_points);
		std::vector<cv::Point2f> PANHomo_pointsCopy(PANHomo_points);

		MSSHomo_points.clear();
		PANHomo_points.clear();

		std::vector<double> response_static2_Sort(response_static2);
		std::sort(response_static2_Sort.begin(), response_static2_Sort.end());

		int nCounter = 0;

		for (int i = response_static2_Sort.size()-1; i >=0; i--)
		{
			if (nCounter>=16)
			{
				break;
			}
			std::vector<double>::iterator findResult = std::find(response_static2.begin(), response_static2.end(), response_static2_Sort[i]);
			int nIndex = std::distance(response_static2.begin(), findResult);
			MSSHomo_points.push_back(MSSHomo_pointsCopy[nIndex]);
			PANHomo_points.push_back(PANHomo_pointsCopy[nIndex]);

			nCounter++;
		}

	}

}

int FusionLYH::ImgWarpPoly(std::vector<cv::Point2f> MSSHomo_points, std::vector<cv::Point2f> PANHomo_points,
	std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, std::vector<cv::Mat> &mul_regist_16)
{
	GDALAllRegister();

	GDAL_GCP *gcp_here = new GDAL_GCP[MSSHomo_points.size()];
	std::string *strs = new std::string[MSSHomo_points.size()];

	for (int i=0;i<MSSHomo_points.size();i++)
	{
		char ch[64];
		_itoa(i, ch, 10);
		gcp_here[i].dfGCPPixel = MSSHomo_points[i].x;
		gcp_here[i].dfGCPLine = MSSHomo_points[i].y;
		gcp_here[i].dfGCPX = PANHomo_points[i].x;
		gcp_here[i].dfGCPY = PANHomo_points[i].y;
		gcp_here[i].dfGCPZ = 0;
		
		strs[i] = ch;
		gcp_here[i].pszId = const_cast<char *>(strs[i].c_str());
		gcp_here[i].pszInfo = NULL;
	}

	int iOrder = 2;
	void *hTransform = GDALCreateGCPTransformer(MSSHomo_points.size(), gcp_here, iOrder, FALSE);
	if (NULL == hTransform)
	{
		return 1;
	}

	int nDstLines = warpedSize.height;
	int nDstPixels = warpedSize.width;

	int nSrcLines = warpedSize.height;
	int nSrcPixels = warpedSize.width;

	for (int i = 0; i < mul_src_16.size(); i++)
	{
		cv::Mat Mat_mul_16_resize;
		cv::resize(mul_src_16[i], Mat_mul_16_resize, warpedSize);//上采样

		cv::Mat Mat_mul_16_dst(nSrcLines, nSrcPixels, CV_16UC1, cv::Scalar(0));

#pragma omp parallel for
		for (int nRow = 0; nRow < nDstLines; nRow++)
		{
			unsigned short* pdata = Mat_mul_16_dst.ptr<unsigned short>(nRow);

			for (int nCol = 0; nCol < nDstPixels; nCol++)
			{
				double dbX = nCol;
				double dbY = nRow;

				int nFlag;
				GDALGCPTransform(hTransform, TRUE, 1, &dbX, &dbY, NULL, &nFlag);

				unsigned short nValue = 0;
				if (dbX < 0 || dbX >= nSrcPixels || dbY < 0 || dbY >= nSrcLines)
				{
					nValue = 0;
				}
				else
				{
					int nLUPixX = floor(dbX);
					int nLUPixY = floor(dbY);
					int nIsLUValid = 1;
					float fLUPixVal = 0;

					if (nLUPixX < 0 || nLUPixX >= nSrcPixels || nLUPixY < 0 || nLUPixY >= nSrcLines)
					{
						nIsLUValid = 0;
					}
					else
					{
	
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLUPixY);  
						fLUPixVal = pRowTemp[nLUPixX];

					}

					int nRUPixX = ceil(dbX);
					int nRUPixY = floor(dbY);
					int nIsRUValid = 1;
					float fRUPixVal = 0;

					if (nRUPixX < 0 || nRUPixX >= nSrcPixels || nRUPixY < 0 || nRUPixY >= nSrcLines)
					{
						nIsRUValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRUPixY);  
						fRUPixVal = pRowTemp[nRUPixX];
					}

					int nLDPixX = floor(dbX);
					int nLDPixY = ceil(dbY);
					int nIsLDValid = 1;
					float fLDPixVal = 0;

					if (nLDPixX < 0 || nLDPixX >= nSrcPixels || nLDPixY < 0 || nLDPixY >= nSrcLines)
					{
						nIsLDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLDPixY); 
						fLDPixVal = pRowTemp[nLDPixX];
					}

					int nRDPixX = ceil(dbX);
					int nRDPixY = ceil(dbY);
					int nIsRDValid = 1;
					float fRDPixVal = 0;

					if (nRDPixX < 0 || nRDPixX >= nSrcPixels || nRDPixY < 0 || nRDPixY >= nSrcLines)
					{
						nIsRDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRDPixY); 
						fRDPixVal = pRowTemp[nRDPixX];
					}

					if ((nIsLUValid + nIsRUValid + nIsLDValid + nIsRDValid) == 4)
					{
						float fWeightR = dbX - nLUPixX;
						float fWeightL = nRUPixX - dbX;
						float fWeightD = dbY - nLUPixY;
						float fWeightU = nLDPixY - dbY;

						float fUPixValWeighted = fWeightL*fLUPixVal + fWeightR*fRUPixVal;
						float fDPixValWeighted = fWeightL*fLDPixVal + fWeightR*fRDPixVal;

						nValue = unsigned short(fWeightU * fUPixValWeighted + fWeightD * fDPixValWeighted);
					}
					else
					{
						int nXCol = int(dbX + 0.5);
						int nYRow = int(dbY + 0.5);

						if (nXCol < 0 || nXCol >= nSrcPixels || nYRow < 0 || nYRow >= nSrcLines)
						{
							nIsLDValid = 0;
						}
						else
						{
							unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nYRow); 
							nValue = pRowTemp[nXCol];
						}
					}

				}
				pdata[nCol] = nValue;
			}
		}

		mul_regist_16.push_back(Mat_mul_16_dst);
	}

	if (hTransform != NULL)
	{
		GDALDestroyGCPTransformer(hTransform);
		hTransform = NULL;
	}
	delete[] gcp_here;
	delete[] strs;
	return 0;
}

int FusionLYH::ImgWarpPolySurf(std::vector<cv::Point2f> MSSHomo_points, std::vector<cv::Point2f> PANHomo_points,
	std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, std::vector<cv::Mat> &mul_regist_16)
{
	GDALAllRegister();

	GDAL_GCP *gcp_here = new GDAL_GCP[MSSHomo_points.size()];
	std::string *strs = new std::string[MSSHomo_points.size()];

	for (int i = 0; i<MSSHomo_points.size(); i++)
	{
		char ch[64];
		_itoa(i, ch, 10);
		gcp_here[i].dfGCPPixel = MSSHomo_points[i].x;
		gcp_here[i].dfGCPLine = MSSHomo_points[i].y;
		gcp_here[i].dfGCPX = PANHomo_points[i].x;
		gcp_here[i].dfGCPY = PANHomo_points[i].y;
		gcp_here[i].dfGCPZ = 0;

		strs[i] = ch;
		gcp_here[i].pszId = const_cast<char *>(strs[i].c_str());
		gcp_here[i].pszInfo = NULL;
	}

	int iOrder = 2;

	void *hTransform = GDALCreateGCPTransformer(MSSHomo_points.size(), gcp_here, iOrder, FALSE);
	
	if (NULL == hTransform)
	{
		return 1;
	}


	int nDstLines = warpedSize.height;
	int nDstPixels = warpedSize.width;

	int nSrcLines = warpedSize.height;
	int nSrcPixels = warpedSize.width;

	for (int i = 0; i < mul_src_16.size(); i++)
	{
		cv::Mat Mat_mul_16_resize;
		cv::resize(mul_src_16[i], Mat_mul_16_resize, warpedSize);//上采样

		cv::Mat Mat_mul_16_dst(nSrcLines, nSrcPixels, CV_16UC1, cv::Scalar(0));

#pragma omp parallel for
		for (int nRow = 0; nRow < nDstLines; nRow++)
		{
			unsigned short* pdata = Mat_mul_16_dst.ptr<unsigned short>(nRow);

			for (int nCol = 0; nCol < nDstPixels; nCol++)
			{
				double dbX = nCol;
				double dbY = nRow;

				int nFlag;
				GDALGCPTransform(hTransform, TRUE, 1, &dbX, &dbY, NULL, &nFlag);

				unsigned short nValue = 0;

				if (dbX < 0 || dbX >= nSrcPixels || dbY < 0 || dbY >= nSrcLines)
				{
					nValue = 0;
				}
				else
				{
					int nLUPixX = floor(dbX);
					int nLUPixY = floor(dbY);
					int nIsLUValid = 1;
					float fLUPixVal = 0;

					if (nLUPixX < 0 || nLUPixX >= nSrcPixels || nLUPixY < 0 || nLUPixY >= nSrcLines)
					{
						nIsLUValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLUPixY);
						fLUPixVal = pRowTemp[nLUPixX];

					}

					int nRUPixX = ceil(dbX);
					int nRUPixY = floor(dbY);
					int nIsRUValid = 1;
					float fRUPixVal = 0;

					if (nRUPixX < 0 || nRUPixX >= nSrcPixels || nRUPixY < 0 || nRUPixY >= nSrcLines)
					{
						nIsRUValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRUPixY); 
						fRUPixVal = pRowTemp[nRUPixX];
					}

					int nLDPixX = floor(dbX);
					int nLDPixY = ceil(dbY);
					int nIsLDValid = 1;
					float fLDPixVal = 0;

					if (nLDPixX < 0 || nLDPixX >= nSrcPixels || nLDPixY < 0 || nLDPixY >= nSrcLines)
					{
						nIsLDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLDPixY);
						fLDPixVal = pRowTemp[nLDPixX];
					}

					int nRDPixX = ceil(dbX);
					int nRDPixY = ceil(dbY);
					int nIsRDValid = 1;
					float fRDPixVal = 0;

					if (nRDPixX < 0 || nRDPixX >= nSrcPixels || nRDPixY < 0 || nRDPixY >= nSrcLines)
					{
						nIsRDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRDPixY);
						fRDPixVal = pRowTemp[nRDPixX];
					}

					if ((nIsLUValid + nIsRUValid + nIsLDValid + nIsRDValid) == 4)
					{
						float fWeightR = dbX - nLUPixX;
						float fWeightL = nRUPixX - dbX;
						float fWeightD = dbY - nLUPixY;
						float fWeightU = nLDPixY - dbY;

						float fUPixValWeighted = fWeightL*fLUPixVal + fWeightR*fRUPixVal;
						float fDPixValWeighted = fWeightL*fLDPixVal + fWeightR*fRDPixVal;

						nValue = unsigned short(fWeightU * fUPixValWeighted + fWeightD * fDPixValWeighted);
					}
					else
					{
						int nXCol = int(dbX + 0.5);
						int nYRow = int(dbY + 0.5);

						if (nXCol < 0 || nXCol >= nSrcPixels || nYRow < 0 || nYRow >= nSrcLines)
						{
							nIsLDValid = 0;
						}
						else
						{
							unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nYRow);
							nValue = pRowTemp[nXCol];
						}
					}

				}
				pdata[nCol] = nValue;
			}
		}

		mul_regist_16.push_back(Mat_mul_16_dst);
	}

	if (hTransform != NULL)
	{
		GDALDestroyGCPTransformer(hTransform);
		hTransform = NULL;
	}
	delete[] gcp_here;
	delete[] strs;
	return 0;
}


int FusionLYH::ImgWarpHomo(cv::Mat homo, std::vector<cv::Mat> &mul_src_16, cv::Size warpedSize, 
	std::vector<cv::Mat> &mul_regist_16)
{
	GDALAllRegister();

	int nDstLines = warpedSize.height;
	int nDstPixels = warpedSize.width;

	int nSrcLines = warpedSize.height;
	int nSrcPixels = warpedSize.width;

	double m11 = homo.at<float>(0, 0);
	double m12 = homo.at<float>(0, 1);
	double m13 = homo.at<float>(0, 2);
	double m21 = homo.at<float>(1, 0);
	double m22 = homo.at<float>(1, 1);
	double m23 = homo.at<float>(1, 2);
	double m31 = homo.at<float>(2, 0);
	double m32 = homo.at<float>(2, 1);
	double m33 = homo.at<float>(2, 2);

	for (int i = 0; i < mul_src_16.size(); i++)
	{
		cv::Mat Mat_mul_16_resize;
		cv::resize(mul_src_16[i], Mat_mul_16_resize, warpedSize);//上采样

		cv::Mat Mat_mul_16_dst(nSrcLines, nSrcPixels, CV_16UC1, cv::Scalar(0));

		for (int nRow = 0; nRow < nDstLines; nRow++)
		{
			unsigned short* pdata = Mat_mul_16_dst.ptr<unsigned short>(nRow);

			for (int nCol = 0; nCol < nDstPixels; nCol++)
			{
				double dbX = nCol;
				double dbY = nRow;

				dbX = (m11*nCol + m12*nRow + m13) / (m31*nCol + m32*nRow + m33);
				dbY = (m21*nCol + m22*nRow + m23) / (m31*nCol + m32*nRow + m33);


				unsigned short nValue = 0;

				if (dbX < 0 || dbX >= nSrcPixels || dbY < 0 || dbY >= nSrcLines)
				{
					nValue = 0;
				}
				else
				{
					int nLUPixX = floor(dbX);
					int nLUPixY = floor(dbY);
					int nIsLUValid = 1;
					float fLUPixVal = 0;

					if (nLUPixX < 0 || nLUPixX >= nSrcPixels || nLUPixY < 0 || nLUPixY >= nSrcLines)
					{
						nIsLUValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLUPixY);
						fLUPixVal = pRowTemp[nLUPixX];

					}

					int nRUPixX = ceil(dbX);
					int nRUPixY = floor(dbY);
					int nIsRUValid = 1;
					float fRUPixVal = 0;

					if (nRUPixX < 0 || nRUPixX >= nSrcPixels || nRUPixY < 0 || nRUPixY >= nSrcLines)
					{
						nIsRUValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRUPixY);
						fRUPixVal = pRowTemp[nRUPixX];
					}

					int nLDPixX = floor(dbX);
					int nLDPixY = ceil(dbY);
					int nIsLDValid = 1;
					float fLDPixVal = 0;

					if (nLDPixX < 0 || nLDPixX >= nSrcPixels || nLDPixY < 0 || nLDPixY >= nSrcLines)
					{
						nIsLDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nLDPixY);
						fLDPixVal = pRowTemp[nLDPixX];
					}

					int nRDPixX = ceil(dbX);
					int nRDPixY = ceil(dbY);
					int nIsRDValid = 1;
					float fRDPixVal = 0;

					if (nRDPixX < 0 || nRDPixX >= nSrcPixels || nRDPixY < 0 || nRDPixY >= nSrcLines)
					{
						nIsRDValid = 0;
					}
					else
					{
						unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nRDPixY);
						fRDPixVal = pRowTemp[nRDPixX];
					}

					if ((nIsLUValid + nIsRUValid + nIsLDValid + nIsRDValid) == 4)
					{
						float fWeightR = dbX - nLUPixX;
						float fWeightL = nRUPixX - dbX;
						float fWeightD = dbY - nLUPixY;
						float fWeightU = nLDPixY - dbY;

						float fUPixValWeighted = fWeightL*fLUPixVal + fWeightR*fRUPixVal;
						float fDPixValWeighted = fWeightL*fLDPixVal + fWeightR*fRDPixVal;

						nValue = unsigned short(fWeightU * fUPixValWeighted + fWeightD * fDPixValWeighted);
					}
					else
					{
						int nXCol = int(dbX + 0.5);
						int nYRow = int(dbY + 0.5);

						if (nXCol < 0 || nXCol >= nSrcPixels || nYRow < 0 || nYRow >= nSrcLines)
						{
							nIsLDValid = 0;
						}
						else
						{
							unsigned short *pRowTemp = Mat_mul_16_resize.ptr<unsigned short>(nYRow);
							nValue = pRowTemp[nXCol];
						}
					}

				}
				pdata[nCol] = nValue;
			}
		}

		mul_regist_16.push_back(Mat_mul_16_dst);
	}


	return 0;

}

//regis MS and PAN image
int FusionLYH::CPU_Surf_Registration_Poly(const char * Mss_path, const char *Pan_path,
	std::vector<cv::Mat> &mul_src_16, std::vector<cv::Mat> &mul_regist_16, std::vector<cv::Mat> &pan_src_16)
{
	cv::Mat mul_src_8, pan_src_8;
	std::string temp1 = "start";
	std::cout << temp1 << std::endl;

	stretch_percent_16to8(Mss_path, mul_src_16, mul_src_8);
	stretch_percent_16to8(Pan_path, pan_src_16, pan_src_8);

	cv::Mat mask;
	get_Mask(pan_src_16[0], mask);
	if (mul_src_16[0].empty() || mul_src_8.empty() || pan_src_16[0].empty() || pan_src_8.empty())
		return 0;
	cv::Mat mul_src_8_gray;
	cv::Mat mul_gray_resize_8;
	cvtColor(mul_src_8, mul_src_8_gray, CV_BGR2GRAY);
	mul_src_8.release();

	cv::resize(mul_src_8_gray, mul_gray_resize_8, pan_src_16[0].size());
	mul_src_8_gray.release();

	int nBlockSize = 1000;
	CV_Assert(!mul_gray_resize_8.empty());
	int img_height = pan_src_16[0].rows;
	int img_width = pan_src_16[0].cols;

	std::vector<cv::Point2f> _all_p01, _all_p02;
	int number = 0;
	temp1 = "registration start";
	std::cout << temp1 << std::endl;

	try
	{
		for (int y = 0; y < pan_src_16[0].rows; y += 3 * nBlockSize)
		{
			for (int x = 0; x < pan_src_16[0].cols; x += 3 * nBlockSize)
			{
				int nXBK = nBlockSize;
				int nYBK = nBlockSize;
				if (y + nBlockSize > img_height)
					nYBK = img_height - y;
				if (x + nBlockSize > img_width)
					nXBK = img_width - x;
				cv::Mat block_Mat1 = mul_gray_resize_8(cv::Rect(x, y, nXBK, nYBK)).clone();
				cv::Mat block_Mat2 = pan_src_8(cv::Rect(x, y, nXBK, nYBK)).clone();
				cv::Mat block_mask = mask(cv::Rect(x, y, nXBK, nYBK)).clone();
				cv::SurfFeatureDetector surf(200);
				//detecting keypoints & computing descriptors
				std::vector<cv::KeyPoint> keypoints1, keypoints2;
				cv::Mat descriptors1, descriptors2;
				if (block_Mat1.rows < 200 || block_Mat1.cols < 200)
					continue;
				surf(block_Mat1, block_mask, keypoints1, descriptors1);
				surf(block_Mat2, block_mask, keypoints2, descriptors2);
				std::cout << "FOUND " << keypoints1.size() << " keypoints on first image" << std::endl;
				std::cout << "FOUND " << keypoints2.size() << " keypoints on second image" << std::endl;
				if (keypoints1.size() <= 0 || keypoints2.size() <= 0)
					continue;
				//matching descriptors
				cv::BFMatcher matcher(cv::NORM_L2);
				std::vector<cv::DMatch> matches;
				matcher.match(descriptors1, descriptors2, matches);
				std::vector<cv::DMatch>good_matches;
				double max_dist = 0;
				double min_dist = 100;
				for (int i = 0; i < descriptors1.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist)
					{
						min_dist = dist;
					}
					else if (dist > max_dist)
					{
						max_dist = dist;
					}
				}
				for (int i = 0; i < descriptors1.rows; i++)
				{
					if (matches[i].distance<0.3)
					{
						good_matches.push_back(matches[i]);
					}
				}
				
				std::vector<cv::KeyPoint> goodImagePoints1, goodImagePoints2;
				std::vector<cv::Point2f>p01, p02;
				cv::Mat homo;
				float homographyReprojectionThreshold = 1.0;
				bool homographyFound = refineMatchesWithHomography(keypoints1, keypoints2, homographyReprojectionThreshold, good_matches, homo);
				sort(good_matches.begin(), good_matches.end());
				std::vector<cv::DMatch> matchesVoted;
				int sort_matchs_number = good_matches.size();
				std::cout << "匹配点数" << sort_matchs_number << std::endl;
				if (sort_matchs_number <= 4)
					continue;
				else
					sort_matchs_number = 5;
				for (int i = 0; i < sort_matchs_number; i++)
				{
					matchesVoted.push_back(good_matches[i]);
				}
				cv::Mat feature_pic1, feature_pic2;
				for (size_t i = 0; i < matchesVoted.size(); i++)
				{
					cv::Point2f temp1, temp2;
					temp1.x = keypoints1[matchesVoted[i].queryIdx].pt.x + x;
					temp1.y = keypoints1[matchesVoted[i].queryIdx].pt.y + y;
					temp2.x = keypoints2[matchesVoted[i].trainIdx].pt.x + x;
					temp2.y = keypoints2[matchesVoted[i].trainIdx].pt.y + y;
					_all_p01.push_back(temp1);
					_all_p02.push_back(temp2);
					goodImagePoints1.push_back(keypoints1[matchesVoted[i].queryIdx]);
					goodImagePoints2.push_back(keypoints2[matchesVoted[i].trainIdx]);
				}

				block_Mat1.release();
				block_Mat2.release();
			}
		}
	}
	catch (const std::exception&)
	{
		return 0;
	}

	//delete matches with large error
	FilterBadSurfMatches(_all_p01, _all_p02);

	for (int i=0;i<_all_p01.size();i++)
	{
		std::cout << _all_p01[i].x - _all_p02[i].x <<" , "<< _all_p01[i].y - _all_p02[i].y << std::endl;
	}


	try
	{

		if (ImgWarpPolySurf(_all_p01, _all_p02, mul_src_16, pan_src_16[0].size(), mul_regist_16) != 0)
		{
			std::cout << "error" << std::endl;
			return 0;
		}

	}
	catch (...)
	{
		std::cout << "error" << std::endl;
		return 0;
	}

	return 1;
}

void FusionLYH::FilterBadSurfMatches(std::vector<cv::Point2f> &p01, std::vector<cv::Point2f> &p02)
{
	double dErrorThresholdPix = m_dErrorThresholdPix;

	double dSumDiffX = 0;
	double dSumDiffY = 0;
	double dMeanDiffX = 0;
	double dMeanDiffY = 0;

	for (int i=0;i<p01.size();i++)
	{
		dSumDiffX = dSumDiffX + (p01[i].x - p02[i].x);
		dSumDiffY = dSumDiffY + (p01[i].y - p02[i].y);
	}

	dMeanDiffX = dSumDiffX / p01.size();
	dMeanDiffY = dSumDiffY / p01.size();


	std::vector<cv::Point2f> p01_new;
	std::vector<cv::Point2f> p02_new;

	for (int i = 0; i < p01.size(); i++)
	{
		double tempXDiff = (p01[i].x - p02[i].x);
		double tempYDiff = (p01[i].y - p02[i].y);

		if (abs(tempXDiff- dMeanDiffX) > dErrorThresholdPix || abs(tempYDiff - dMeanDiffY) > dErrorThresholdPix)
		{
			continue;
		}
		else
		{
			cv::Point2f tempP01;
			tempP01.x = p01[i].x; tempP01.y = p01[i].y;
			cv::Point2f tempP02;
			tempP02.x = p02[i].x; tempP02.y = p02[i].y;

			p01_new.push_back(tempP01);
			p02_new.push_back(tempP02);
		}

	}

	p01 = p01_new;
	p02 = p02_new;
}



int FusionLYH::CheckMatchPointValid(cv::Mat mul_gray_resize_8, int block_x, int block_y, int x_img, int y_img)
{
	int nCheckValidLU_x = x_img - block_x / 4;
	int nCheckValidLU_y = y_img - block_y / 4;
	int nCheckValidLU_Record = 1;
	if ((nCheckValidLU_x<0) || (nCheckValidLU_y<0))
	{
		nCheckValidLU_Record = 0;
	}

	int nCheckValidRU_x = x_img + block_x / 4;
	int nCheckValidRU_y = y_img - block_y / 4;
	int nCheckValidRU_Record = 1;
	if ((nCheckValidRU_x>mul_gray_resize_8.cols) || (nCheckValidRU_y<0))
	{
		nCheckValidRU_Record = 0;
	}

	int nCheckValidLD_x = x_img - block_x / 4;
	int nCheckValidLD_y = y_img + block_y / 4;
	int nCheckValidLD_Record = 1;
	if ((nCheckValidLD_x<0) || (nCheckValidLD_y>mul_gray_resize_8.rows))
	{
		nCheckValidLD_Record = 0;
	}

	int nCheckValidRD_x = x_img + block_x / 4;
	int nCheckValidRD_y = y_img + block_y / 4;
	int nCheckValidRD_Record = 1;
	if ((nCheckValidRD_x>mul_gray_resize_8.cols) || (nCheckValidRD_y>mul_gray_resize_8.rows))
	{
		nCheckValidRD_Record = 0;
	}

	if (nCheckValidLU_Record==1)
	{
		uchar *pRowTemp = mul_gray_resize_8.ptr<uchar>(nCheckValidLU_y); 
		uchar ucCheckPixVal = pRowTemp[nCheckValidLU_x];

		if (ucCheckPixVal==0)
		{
			nCheckValidLU_Record = 0;
		}
	}

	if (nCheckValidRU_Record == 1)
	{
		uchar *pRowTemp = mul_gray_resize_8.ptr<uchar>(nCheckValidRU_y);
		uchar ucCheckPixVal = pRowTemp[nCheckValidRU_x];

		if (ucCheckPixVal == 0)
		{
			nCheckValidRU_Record = 0;
		}
	}

	if (nCheckValidLD_Record == 1)
	{
		uchar *pRowTemp = mul_gray_resize_8.ptr<uchar>(nCheckValidLD_y);
		uchar ucCheckPixVal = pRowTemp[nCheckValidLD_x];

		if (ucCheckPixVal == 0)
		{
			nCheckValidLD_Record = 0;
		}
	}

	if (nCheckValidRD_Record == 1)
	{
		uchar *pRowTemp = mul_gray_resize_8.ptr<uchar>(nCheckValidRD_y);
		uchar ucCheckPixVal = pRowTemp[nCheckValidRD_x];

		if (ucCheckPixVal == 0)
		{
			nCheckValidRD_Record = 0;
		}
	}

	return nCheckValidLU_Record + nCheckValidRU_Record + nCheckValidLD_Record + nCheckValidRD_Record;

}

//image fusion
int FusionLYH::AGSFIM(std::vector<cv::Mat> Mss, std::vector<cv::Mat> &Mss_registed, cv::Mat Pan)
{
	try {

		cv::Mat gaussian_pan_resize = Get_Pan_ds(Mss, Pan);//Get_Pan_ds get PAN'
		int img_height = Pan.rows;
		int img_width = Pan.cols;
		int nBlockSize = 3000; // divide into blocks

		for (int y = 0; y < Pan.rows; y += nBlockSize)
		{
			for (int x = 0; x < Pan.cols; x += nBlockSize)
			{
				int nXBK = nBlockSize;
				int nYBK = nBlockSize;
				if (y + nBlockSize > img_height)
					nYBK = img_height - y;
				if (x + nBlockSize > img_width)
					nXBK = img_width - x;
				cv::Mat block_Mat1 = cv::Mat(Pan(cv::Rect(x, y, nXBK, nYBK)));
				cv::Mat block_Mat3 = gaussian_pan_resize(cv::Rect(x, y, nXBK, nYBK));
				block_Mat1.convertTo(block_Mat1, CV_32FC1);
				block_Mat3.convertTo(block_Mat3, CV_32FC1);
				for (int i = 0; i < Mss_registed.size(); i++)
				{
					cv::Mat block_Mat2 = cv::Mat(Mss_registed.at(i)(cv::Rect(x, y, nXBK, nYBK)));
					block_Mat2.convertTo(block_Mat2, CV_32FC1);
					cv::multiply(block_Mat2, block_Mat1, block_Mat2);
					cv::divide(block_Mat2, block_Mat3, block_Mat2);
					block_Mat2.copyTo(Mss_registed.at(i)(cv::Rect(x, y, nXBK, nYBK)));
				}
				block_Mat1.release();
				block_Mat3.release();
			}
		}
		return 1;

	}
	catch (...) 
	{
		return 0;
	}

}

//stretch 16 to 8
int FusionLYH::stretch_percent_16to8(const char *inFilename, std::vector<cv::Mat> &src_all_channels, cv::Mat &src8)
{
	GDALAllRegister();

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	int src_height = 0;
	int src_width = 0;
	GDALDataset *poIn = (GDALDataset *)GDALOpen(inFilename, GA_ReadOnly);			
	if (poIn == NULL)
		return 0;
	src_width = poIn->GetRasterXSize();
	src_height = poIn->GetRasterYSize();

	int InBands = poIn->GetRasterCount();
	if (InBands == 1)
		src8.create(src_height, src_width, CV_8UC1);
	else
		src8.create(src_height, src_width, CV_8UC3);

	poIn->GetGeoTransform(GEOpara);
	PROdata = poIn->GetProjectionRef();

	double src_min = 0.0, src_max = 0.0;
	double* minp = &src_min;
	double* maxp = &src_max;
	for (int iBand = 0; iBand < InBands; iBand++)
	{
		uint16_t *srcData = (uint16_t *)malloc(sizeof(uint16_t) *src_width * src_height * 1);//又申请内存
		memset(srcData, 0, sizeof(uint16_t) * 1 * src_width * src_height);

		poIn->GetRasterBand(iBand + 1)->RasterIO(GF_Read, 0, 0, src_width, src_height, srcData + 0 * src_width * src_height, src_width, src_height, GDT_UInt16, 0, 0);
		cv::Mat src_mat_16_temp = cv::Mat(src_height, src_width, CV_16UC1, srcData + 0 * src_width * src_height).clone();

		src_all_channels.push_back(src_mat_16_temp);
		if (iBand < 3)
		{	
			minMaxIdx(src_all_channels[iBand], minp, maxp);
			double *accumlt_frequency_val = new double[src_max + 1]();
			int channels = 0;
			cv::MatND dstHist;
			int histSize[] = { (int)src_max + 1 };  
			float midRanges[] = { 0, (float)src_max };
			const float *ranges[] = { midRanges };
			src_mat_16_temp.convertTo(src_mat_16_temp, CV_32FC1);
			calcHist(&src_mat_16_temp, 1, &channels, cv::Mat(), dstHist, 1, histSize, ranges, true, false);

			dstHist = dstHist / src_mat_16_temp.rows*src_mat_16_temp.cols;
			accumlt_frequency_val[0] = dstHist.at<float>(0);
			for (size_t i = 1; i < src_max + 1; i++)
			{
				accumlt_frequency_val[i] = accumlt_frequency_val[i - 1] + dstHist.at<float>(i);
			}
			int minVal = 0, maxVal = 0;
			for (int val_i = 1; val_i < src_max; val_i++)
			{
				double acc_fre_temVal0 = *(accumlt_frequency_val + 0);
				double acc_fre_temVal = *(accumlt_frequency_val + val_i);
				if ((acc_fre_temVal - acc_fre_temVal0) > 0.0015)
				{
					minVal = val_i;
					break;
				}
			}
			for (int val_i = src_max - 1; val_i > 0; val_i--)
			{
				double acc_fre_temVal0 = *(accumlt_frequency_val + (int)src_max);
				double acc_fre_temVal = *(accumlt_frequency_val + val_i);
				if (acc_fre_temVal < (acc_fre_temVal0 - 0.00012))
				{
					maxVal = val_i;
					break;
				}
			}
			for (int src_row = 0; src_row < src_height; src_row++)
			{
				uint8_t *dstData = (uint8_t*)malloc(sizeof(uint8_t)*src_width);
				memset(dstData, 0, sizeof(uint8_t)*src_width);
				for (int src_col = 0; src_col < src_width; src_col++)
				{
					uint16_t src_temVal = *(srcData + src_row * src_width + src_col);
					double stre_temVal = (src_temVal - minVal) / double(maxVal - minVal);
					if (src_temVal < minVal)
					{
						*(dstData + src_col) = (src_temVal) *(20.0 / double(minVal));
					}
					else if (src_temVal > maxVal)
					{
						stre_temVal = (src_temVal - src_min) / double(src_max - src_min);
						*(dstData + src_col) = 254;
					}
					else
						*(dstData + src_col) = pow(stre_temVal, 0.7) * 250;
					if (InBands >= 3)
						src8.at<cv::Vec3b>(src_row, src_col)[iBand] = *(dstData + src_col);
					else if (InBands == 1)
						src8.at<uchar>(src_row, src_col) = *(dstData + src_col);
				}
				free(dstData);
			}
			delete[] accumlt_frequency_val;
			free(srcData);
		}
	}
	GDALClose(poIn);
	return 1;
}

//get valid mask
void FusionLYH::get_Mask(cv::Mat Mss_Mat, cv::Mat &Mask)
{
	int R, G, B, NIR;
	Mask = cv::Mat::zeros(Mss_Mat.rows, Mss_Mat.cols, CV_8UC1);
	valid_values_amounts = 0;

	std::vector<int> valid_values_each_rows(Mss_Mat.rows, 0);

	if (Mss_Mat.channels() == 4)
	{
#pragma omp parallel for
		for (int i = 0; i < Mss_Mat.rows; i++)
		{

			unsigned short* pRowTemp = Mss_Mat.ptr<unsigned short>(i);
			uchar* pRowTempMask = Mask.ptr<uchar>(i);

			for (int j = 0; j < Mss_Mat.cols; j++)
			{
				NIR = pRowTemp[4 * j + 0];
				R = pRowTemp[4 * j + 1];
				G = pRowTemp[4 * j + 2];
				B = pRowTemp[4 * j + 3];

				if (R != 0 || G != 0 || B != 0 || NIR != 0)
				{
					pRowTempMask[j] = 255;
					valid_values_each_rows[i]++;
				}
				

			}
		}

	}
	else if (Mss_Mat.channels() == 3)
	{
#pragma omp parallel for
		for (int i = 0; i < Mss_Mat.rows; i++)
		{
			unsigned short* pRowTemp = Mss_Mat.ptr<unsigned short>(i);
			uchar* pRowTempMask = Mask.ptr<uchar>(i);

			for (int j = 0; j < Mss_Mat.cols; j++)
			{
				R = pRowTemp[3 * j + 0];
				G = pRowTemp[3 * j + 1];
				B = pRowTemp[3 * j + 2];

				if (R != 0 || G != 0 || B != 0)
				{
					pRowTempMask[j] = 255;
					valid_values_each_rows[i]++;
				}


			}
		}
	}
	else if (Mss_Mat.channels() == 1)
	{
#pragma omp parallel for
		for (int i = 0; i < Mss_Mat.rows; i++)
		{
			unsigned short* pRowTemp = Mss_Mat.ptr<unsigned short>(i);
			uchar* pRowTempMask = Mask.ptr<uchar>(i);

			for (int j = 0; j < Mss_Mat.cols; j++)
			{
				R = pRowTemp[j];

				if (R != 0)
				{
					pRowTempMask[j] = 255;
					valid_values_each_rows[i]++;
				}


			}
		}
	}
	else
	{
		std::cout << " Mss_Mat.channels number not support!" << std::endl;
		return;
	}



	for (int i=0;i<valid_values_each_rows.size();i++)
	{
		valid_values_amounts = valid_values_amounts+valid_values_each_rows[i];
	}
	
}

bool FusionLYH::refineMatchesWithHomography(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints, float reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography)
{
	const int minNumberMatchesAllowed = 4;
	if (matches.size() < minNumberMatchesAllowed)
		return false;
	// Prepare data for cv::findHomography    
	std::vector<cv::Point2f> queryPoints(matches.size());
	std::vector<cv::Point2f> trainPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		queryPoints[i] = queryKeypoints[matches[i].queryIdx].pt;
		trainPoints[i] = trainKeypoints[matches[i].trainIdx].pt;
	}
	// Find homography matrix and get inliers mask    
	std::vector<unsigned char> inliersMask(matches.size());
	homography = cv::findHomography(queryPoints,
		trainPoints,
		CV_FM_RANSAC,
		reprojectionThreshold,
		inliersMask);
	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
	return matches.size() > minNumberMatchesAllowed;
}

//input PAN is after gradient simulation   Mss_split is the original MS image
cv::Mat FusionLYH::Get_Pan_ds(std::vector<cv::Mat> Mss_split, cv::Mat Pan)
{
	const char * panAdjPath = m_sPANAdjImgPath.c_str();

	GDALAllRegister();

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	int src_height = 0;
	int src_width = 0;
	GDALDataset *poIn = (GDALDataset *)GDALOpen(panAdjPath, GA_ReadOnly);//get image size			
	src_width = poIn->GetRasterXSize();
	src_height = poIn->GetRasterYSize();

	//read image
	uint16_t *srcData = (uint16_t *)malloc(sizeof(uint16_t) *src_width * src_height * 1);
	memset(srcData, 0, sizeof(uint16_t) * 1 * src_width * src_height);

	poIn->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, src_width, src_height, srcData + 0 * src_width * src_height, src_width, src_height, GDT_UInt16, 0, 0);
	cv::Mat src_mat_16_temp = cv::Mat(src_height, src_width, CV_16UC1, srcData + 0 * src_width * src_height).clone();

	//output report
	FILE* txtfile = fopen(m_sTxtReportPath.c_str(), "wt");


	//compute mean for two types of areas
	double dPANMeanB = 0;
	double dPANMeanO = 0;
	ComputeBOMeanByBuildingWeight(srcData, m_sBuildingWeightPath,src_width, src_height, dPANMeanB, dPANMeanO);

	fprintf(txtfile, "dPANMeanB, dPANMeanO : %lf,%lf\n", dPANMeanB, dPANMeanO);


	std::vector<double> vMSSMeanB;
	std::vector<double> vMSSMeanO;
	for (int i = 0; i < Mss_split.size(); i++)
	{
		double dMSSMeanB = 0;
		double dMSSMeanO = 0;

		GUInt16 *pTempMSSBand = new GUInt16[Mss_split[i].rows*Mss_split[i].cols];

#pragma omp parallel for
		for (int r = 0; r < Mss_split[i].rows; r++)
		{
			int tmpI = r*Mss_split[i].cols;
			GUInt16 *p = Mss_split[i].ptr<GUInt16>(r);
			for (int c = 0; c < Mss_split[i].cols; c++)
			{
				pTempMSSBand[tmpI + c] = p[c];
			}
		}


		ComputeBOMeanByBuildingWeight(pTempMSSBand, m_sBuildingWeightDownSampPath, 
			Mss_split[i].cols, Mss_split[i].rows, dMSSMeanB, dMSSMeanO);

		vMSSMeanB.push_back(dMSSMeanB);
		vMSSMeanO.push_back(dMSSMeanO);

		fprintf(txtfile, "dMSSMeanB, dMSSMeanO : %lf,%lf\n", dMSSMeanB, dMSSMeanO);


		delete[] pTempMSSBand;
	}

	//compute miu
	std::vector<double> vMiuB;
	std::vector<double> vMiuO;
	for (int i = 0; i < Mss_split.size(); i++)
	{
		double dMiuB = 1;
		double dMiuO = 1;
		dMiuB = dPANMeanB / vMSSMeanB[i];
		dMiuO = dPANMeanO / vMSSMeanO[i];
		vMiuB.push_back(dMiuB);
		vMiuO.push_back(dMiuO);

		fprintf(txtfile, "dMiuB, dMiuO : %lf,%lf\n", dMiuB, dMiuO);

	}
	

	//compute AG
	double dPAN_BAG = 1;
	double dPAN_OAG = 1;
	ComputeBOAGByBuildingWeight(srcData, m_sBuildingWeightPath,src_width, src_height,dPAN_BAG, dPAN_OAG);

	fprintf(txtfile, "dPAN_BAG, dPAN_OAG : %lf,%lf\n", dPAN_BAG, dPAN_OAG);

	std::vector<double> vMSS_BAG;//MSS的AG
	std::vector<double> vMSS_OAG;
	for (int i = 0; i < Mss_split.size(); i++)
	{
		double dMSSBAG = 0;
		double dMSSOAG = 0;

		GUInt16 *pTempMSSBand = new GUInt16[Mss_split[i].rows*Mss_split[i].cols];

#pragma omp parallel for
		for (int r = 0; r < Mss_split[i].rows; r++)
		{
			int tmpI = r*Mss_split[i].cols;
			GUInt16 *p = Mss_split[i].ptr<GUInt16>(r);
			for (int c = 0; c < Mss_split[i].cols; c++)
			{
				pTempMSSBand[tmpI + c] = p[c];
			}
		}

		ComputeBOAGByBuildingWeight(pTempMSSBand, m_sBuildingWeightDownSampPath, Mss_split[i].cols, Mss_split[i].rows, dMSSBAG, dMSSOAG);
		dMSSBAG = dMSSBAG * vMiuB[i];
		dMSSOAG = dMSSOAG * vMiuO[i];
		
		vMSS_BAG.push_back(dMSSBAG);
		vMSS_OAG.push_back(dMSSOAG);

		fprintf(txtfile, "dMSSBAG, dMSSOAG : %lf,%lf\n", dMSSBAG, dMSSOAG);

		delete[] pTempMSSBand;
	}

	//the ideal AG for built-up and non-built-up
	double dIdealBAG = 0;
	double dIdealOAG = 0;
	for (int i = 0; i < Mss_split.size(); i++)
	{
		dIdealBAG += vMSS_BAG[i];
		dIdealOAG += vMSS_OAG[i];
	}
	dIdealBAG = dIdealBAG / Mss_split.size();
	dIdealOAG = dIdealOAG / Mss_split.size();

	fprintf(txtfile, "dIdealBAG, dIdealOAG : %lf,%lf\n", dIdealBAG, dIdealOAG);


	// σ 0.5 - 1.5，compute PAN' gradient 
	cv::Mat d_src_pan = src_mat_16_temp;//Pan;
	cv::Mat d_src_pan_ds;
	cv::resize(d_src_pan, d_src_pan_ds, Mss_split[0].size());
	d_src_pan.release();

	std::vector<double> fit_Sigma;
	std::vector<double> fit_AGB;
	std::vector<double> fit_AGO;
	for (double i = 1.5; i >= 0.5; i = i - 0.1)
	{
		cv::Mat gaussian_pan;
		cv::GaussianBlur(d_src_pan_ds, gaussian_pan, cv::Size(31, 31), i, i);//gaussian filter

		GUInt16 *tempBand = new GUInt16[gaussian_pan.rows*gaussian_pan.cols];

#pragma omp parallel for
		for (int r = 0; r < gaussian_pan.rows; r++)
		{
			int tmpI = r*gaussian_pan.cols;
			GUInt16 *p = gaussian_pan.ptr<GUInt16>(r);
			for (int c = 0; c < gaussian_pan.cols; c++)
			{
				tempBand[tmpI + c] = p[c];
			}
		}


		double dPAN_BAG = 1;
		double dPAN_OAG = 1;
		ComputeBOAGByBuildingWeight(tempBand, m_sBuildingWeightDownSampPath,
			gaussian_pan.cols, gaussian_pan.rows, dPAN_BAG, dPAN_OAG);

		gaussian_pan.release();
		fit_Sigma.push_back(i);
		fit_AGB.push_back(dPAN_BAG);
		fit_AGO.push_back(dPAN_OAG);


		fprintf(txtfile, "dPAN_BAG, dPAN_OAG : %lf,%lf\n", dPAN_BAG, dPAN_OAG);

		delete[] tempBand;
	}

	czy::Fit fitB;
	fitB.polyfit(fit_Sigma, fit_AGB, 2);// fitting
	std::vector<double> factorB;
	fitB.getFactor(factorB);

	czy::Fit fitO;
	fitO.polyfit(fit_Sigma, fit_AGO, 2);//fitting
	std::vector<double> factorO;
	fitO.getFactor(factorO);

	//compute root
	double optimal_adjust_parameterB = 0;
	double temp_para1B = ((-factorB[1]) + sqrt(pow(factorB[1], 2) - 4 * factorB[2] * (factorB[0] - dIdealBAG))) / (2 * factorB[2]);// 
	double temp_para2B = ((-factorB[1]) - sqrt(pow(factorB[1], 2) - 4 * factorB[2] * (factorB[0] - dIdealBAG))) / (2 * factorB[2]);
	if (temp_para1B > 0 && temp_para2B > 0)
	{
		optimal_adjust_parameterB = temp_para1B < temp_para2B ? temp_para1B : temp_para2B;
	}
	else
	{
		optimal_adjust_parameterB = temp_para1B > temp_para2B ? temp_para1B : temp_para2B;
	}

	double optimal_adjust_parameterO = 0;
	double temp_para1O = ((-factorO[1]) + sqrt(pow(factorO[1], 2) - 4 * factorO[2] * (factorO[0] - dIdealOAG))) / (2 * factorO[2]);
	double temp_para2O = ((-factorO[1]) - sqrt(pow(factorO[1], 2) - 4 * factorO[2] * (factorO[0] - dIdealOAG))) / (2 * factorO[2]);
	if (temp_para1O > 0 && temp_para2O > 0)
	{
		optimal_adjust_parameterO = temp_para1O < temp_para2O ? temp_para1O : temp_para2O;
	}
	else
	{
		optimal_adjust_parameterO = temp_para1O > temp_para2O ? temp_para1O : temp_para2O;
	}

	fprintf(txtfile, "sigmaB, sigmaO : %lf,%lf\n", optimal_adjust_parameterB, optimal_adjust_parameterO);
	fclose(txtfile);


	cv::Mat final_gauss_pan_dsB;
	cv::Mat final_gauss_pan_dsO;

	cv::GaussianBlur(d_src_pan_ds, final_gauss_pan_dsB, cv::Size(31, 31), optimal_adjust_parameterB, 
		optimal_adjust_parameterB);
	cv::GaussianBlur(d_src_pan_ds, final_gauss_pan_dsO, cv::Size(31, 31), optimal_adjust_parameterO,
		optimal_adjust_parameterO);

	d_src_pan_ds.release();


	//generate PAN' weighted by building factor

	cv::Mat combine_gauss_pan_ds = CombinePANByBuildingWeight(final_gauss_pan_dsB, final_gauss_pan_dsO,
		m_sBuildingWeightDownSampPath);

	cv::Mat gaussian_pan_resize;
	cv::resize(combine_gauss_pan_ds, gaussian_pan_resize, Pan.size());

	std::vector<cv::Mat> tempVec;
	tempVec.push_back(gaussian_pan_resize);


	return gaussian_pan_resize;
}

cv::Mat FusionLYH::CombinePANByBuildingWeight(cv::Mat PanB, cv::Mat PanO, std::string sBuildingWeightPath)
{
	GDALAllRegister();

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDataset *poSeg = (GDALDataset *)GDALOpen(sBuildingWeightPath.c_str(), GA_ReadOnly);
	int src_width = poSeg->GetRasterXSize();
	int src_height = poSeg->GetRasterYSize();
	float * pSegData = new float[src_width*src_height];
	poSeg->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, src_width, src_height, pSegData, src_width, src_height, GDT_Float32, 0, 0);

	int nImgSizeY = PanB.rows;
	int nImgSizeX = PanB.cols;

	GUInt16 *bufferB = new GUInt16[PanB.rows*PanB.cols];

#pragma omp parallel for
	for (int r = 0; r < PanB.rows; r++)
	{
		int tmpI = r*PanB.cols;
		GUInt16 *p = PanB.ptr<GUInt16>(r);
		for (int c = 0; c < PanB.cols; c++)
		{
			bufferB[tmpI + c] = p[c];
		}
	}


	GUInt16 *bufferO = new GUInt16[PanO.rows*PanO.cols];

#pragma omp parallel for
	for (int r = 0; r < PanO.rows; r++)
	{
		int tmpI = r*PanO.cols;
		GUInt16 *p = PanO.ptr<GUInt16>(r);
		for (int c = 0; c < PanO.cols; c++)
		{
			bufferO[tmpI + c] = p[c];
		}
	}


	GUInt16 * bufferCombine = new GUInt16[nImgSizeY*nImgSizeX];

	for (int i = 0; i<nImgSizeY; i++)
	{
		for (int j = 0; j<nImgSizeX; j++)
		{
			int nB = bufferB[i*nImgSizeX + j];
			int nO = bufferO[i*nImgSizeX + j];
			float weight = pSegData[i*nImgSizeX + j];
			double temp = 0;
			temp = weight*nB + (1 - weight)*nO;

			if (temp<0)
			{
				temp = 0;
			}

			bufferCombine[i*nImgSizeX + j] = unsigned short(temp);

		}
	}

	cv::Mat CombineResult = cv::Mat(nImgSizeY, nImgSizeX, CV_16UC1, bufferCombine);


	delete[] bufferB;
	delete[] bufferO;

	return CombineResult;


}

void FusionLYH::ComputeBOMeanByBuildingWeight(GUInt16 * pBuffer, std::string sBuildingWeightPath,
	int nWidth, int nHeight,
	double & BMean, double & OMean)
{

	GDALAllRegister();

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");


	GDALDataset *poSeg = (GDALDataset *)GDALOpen(sBuildingWeightPath.c_str(), GA_ReadOnly);
	int src_width = poSeg->GetRasterXSize();
	int src_height = poSeg->GetRasterYSize();
	float * pSegData = new float[src_width*src_height];
	poSeg->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, src_width, src_height, pSegData, src_width, src_height, GDT_Float32, 0, 0);


	double dTempMean_B = 0;
	double dTempMean_O = 0;

	int nBPixNum = 0;
	int nOPixNum = 0;

	for (int i = 0; i<(nHeight); i++)
	{
		for (int j = 0; j<(nWidth); j++)
		{
			double dThisPix = pBuffer[i*nWidth + j];

			float fMask = pSegData[i*nHeight + j];

			if (fMask>0.5)//threshold for determine wether it is built-up
			{
				nBPixNum = nBPixNum + 1;
				dTempMean_B = dTempMean_B + dThisPix;
			}
			else //non-built-up
			{
				nOPixNum = nOPixNum + 1;
				dTempMean_O = dTempMean_O + dThisPix;
			}

		}
	}

	dTempMean_B = dTempMean_B / nBPixNum;
	dTempMean_O = dTempMean_O / nOPixNum;

	BMean = dTempMean_B;
	OMean = dTempMean_O;
	GDALClose(poSeg);
}

void FusionLYH::ComputeBOAGByBuildingWeight(GUInt16 * pBuffer, std::string sBuildingWeightPath,
	int nWidth, int nHeight, 
	double & BAG, double & OAG)
{
	GDALAllRegister();

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDataset *poSeg = (GDALDataset *)GDALOpen(sBuildingWeightPath.c_str(), GA_ReadOnly);
	int src_width = poSeg->GetRasterXSize();
	int src_height = poSeg->GetRasterYSize();
	float * pSegData = new float[src_width*src_height];
	poSeg->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, src_width, src_height, pSegData, src_width, src_height, GDT_Float32, 0, 0);


	double dTempAG_B = 0;
	double dTempAG_O = 0;

	int nBPixNum = 0;
	int nOPixNum = 0;

	for (int i = 0; i<(src_height-1); i++)
	{
		for (int j = 0; j<(src_width-1); j++)
		{
			double dThisPix = pBuffer[i*src_width + j];
			double dRightPix = pBuffer[i*src_width + j + 1];
			double dDownPix = pBuffer[(i + 1)*src_width + j];

			double dx = dRightPix - dThisPix;
			double dy = dDownPix - dThisPix;
			double ds = std::sqrt((dx*dx + dy*dy) / 2);

			float fMask = pSegData[i*src_height+j];

			if (fMask>0.5)//threshold for determine whether it is building
			{
				nBPixNum = nBPixNum + 1;
				dTempAG_B = dTempAG_B + ds;
			}
			else 
			{
				nOPixNum = nOPixNum + 1;
				dTempAG_O = dTempAG_O + ds;
			}

		}
	}

	dTempAG_B = dTempAG_B / nBPixNum;
	dTempAG_O = dTempAG_O / nOPixNum;

	BAG = dTempAG_B;
	OAG = dTempAG_O;

	GDALClose(poSeg);
}





//compute AG 
double* FusionLYH::cal_mean_gradient(cv::Mat img) {
	double *_value = new double[3];
	cv::Mat mat_mean, mat_stddev;
	//GpuMat img_Mat_gpu(img);
	//Scalar _mean, _stddev;
	//gpu::meanStdDev(img_Mat_gpu, _mean, _stddev);
	meanStdDev(img, mat_mean, mat_stddev);
	double m, s;
	m = mat_mean.at<double>(0, 0);
	s = mat_stddev.at<double>(0, 0);
	//m = _mean.val[0];
	//s = _stddev.val[0];
	_value[0] = m;
	_value[1] = s;
	img.convertTo(img, CV_64FC1);
	double tmp = 0;

	double imageAvG = cal_gradient(img); //tmp / (rows*cols);
	_value[2] = imageAvG;
	mat_mean.release();
	mat_stddev.release();
	return _value;
}

double FusionLYH::cal_gradient(cv::Mat img)
{
	int src_height = 0;
	int src_width = 0;
	src_width = img.cols;
	src_height = img.rows;
	int InBands = img.channels();
	clock_t start, finish;
	start = clock();
	uint16_t src_temVal, value1, value2;
	double tmp = 0;
	for (int iBand = 0; iBand < InBands; iBand++)
	{
		double *srcData;//;= (double *)malloc(sizeof(double) *src_width * src_height * 1);
		srcData = (double *)img.data;
		for (int src_row = 0; src_row < src_height - 1; src_row++)
		{
			for (int src_col = 0; src_col < src_width - 1; src_col++)
			{
				src_temVal = *(srcData + src_row * src_width + src_col);
				value1 = *(srcData + src_row * src_width + src_col + 1);
				value2 = *(srcData + (src_row + 1) * src_width + src_col);
				double dx = value1 - src_temVal;
				double dy = value2 - src_temVal;
				double ds = std::sqrt((dx*dx + dy*dy) / 2);
				tmp += ds;
			}
		}
	}
	double imageAvG = tmp / (src_width*src_height);
	return imageAvG;
}




//convert cv::Mat to 16-bit array
bool FusionLYH::Mat2File_depth16(std::vector<cv::Mat> imgMat, char * fileName)
{
	if (imgMat.size() == 0)   
	{
		std::cout << "opencv mat is null！" << std::endl;
		return 0;
	}
	const int nBandCount = imgMat.size();
	const int nImgSizeX = imgMat[0].cols;
	const int nImgSizeY = imgMat[0].rows;

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALAllRegister();

	GDALDataset *poDataset;  
	GDALDriver *poDriver;     
	char *dst_driver_type = findImageTypeGDAL(fileName);
	poDriver = GetGDALDriverManager()->GetDriverByName(dst_driver_type);
	if (poDriver == NULL)
		return 0;
	GDALDataType type;
	type = GDT_UInt16;
	poDataset = poDriver->Create(fileName, nImgSizeX, nImgSizeY, nBandCount, type, NULL);

	GDALRasterBand *pBand = NULL;
	unsigned short *ppafScan = new unsigned short[nImgSizeX * nImgSizeY];
	cv::Mat tmpMat;
	int n1 = nImgSizeY;
	int nc = nImgSizeX;
	for (int i = 1; i <= nBandCount; i++)
	{
		pBand = poDataset->GetRasterBand(i);
		tmpMat = imgMat.at(i - 1);

#pragma omp parallel for
		for (int r = 0; r < nImgSizeY; r++)
		{
			int tmpI = r*nImgSizeX;
			unsigned short *p = tmpMat.ptr<unsigned short>(r);
			for (int c = 0; c < nImgSizeX; c++)
			{
				ppafScan[tmpI + c] = p[c];
			}
		}
		

		pBand->RasterIO(GF_Write, 0, 0, nImgSizeX, nImgSizeY, ppafScan,
			nImgSizeX, nImgSizeY, type, 0, 0);
		pBand->SetNoDataValue(0);

		
	}


	if (PROdata_Pan != NULL)
	{
		poDataset->SetProjection(PROdata_Pan);
		poDataset->SetGeoTransform(GEOpara_Pan);
	}
	GDALClose(poDataset);
	for (int i = 0; i < imgMat.size(); i++)
	{
		imgMat[i].release();
	}
	imgMat.clear();
	std::vector<cv::Mat>().swap(imgMat);

	delete[] ppafScan;


	return 1;
}

char* FusionLYH::findImageTypeGDAL(char *pDstImgFileName)
{
	char *dstExtension = _strlwr(strrchr(pDstImgFileName, '.') + 1);
	char *Gtype = NULL;
	if (0 == strcmp(dstExtension, "bmp")) Gtype = "BMP";
	else if (0 == strcmp(dstExtension, "jpg")) Gtype = "JPEG";
	else if (0 == strcmp(dstExtension, "png")) Gtype = "PNG";
	else if (0 == strcmp(dstExtension, "tif")) Gtype = "GTiff";
	else if (0 == strcmp(dstExtension, "gif")) Gtype = "GIF";
	else if (0 == strcmp(dstExtension, "pix")) Gtype = "PCIDSK";
	else if (0 == strcmp(dstExtension, "img")) Gtype = "HFA";
	else Gtype = NULL;
	return Gtype;
}


bool FusionLYH::FFT_LYH(double RESPONSE_VALUE, cv::Mat block_Mat1_cpu, cv::Mat block_Mat2_cpu,
	cv::Point2f &p01, cv::Point2f &p02, double &response_static,
	int start_x1, int start_y1, int start_x2, int start_y2)
{
	cv::Mat mul_src_8, pan_src_8;
	if (block_Mat1_cpu.channels() > 1)
	{
		cvtColor(block_Mat1_cpu, mul_src_8, CV_BGR2GRAY);
	}
	else
	{
		mul_src_8 = block_Mat1_cpu;
	}
	block_Mat1_cpu.release();

	if (block_Mat2_cpu.channels() > 1)
	{
		cvtColor(block_Mat2_cpu, pan_src_8, CV_BGR2GRAY);
	}
	else
	{
		pan_src_8 = block_Mat2_cpu;
	}

	block_Mat2_cpu.release();

	FFTML GOAL;
	cv::Point2d tr;
	double response;
	std::vector<cv::Point2f> one, two;
	cv::RotatedRect rr = GOAL.LogPolarFFTTemplateMatch(mul_src_8, pan_src_8, tr, response, 100, 200);

	int response_round;
	int aa, bb;
	if (response >= RESPONSE_VALUE)
	{
		cv::Point2d p(pan_src_8.cols / 2, pan_src_8.rows / 2);
		cv::Point2d p_true(pan_src_8.cols / 2 + start_x2, pan_src_8.rows / 2 + start_y2);

		cv::Point2d rr_true;
		rr_true.x = rr.center.x + start_x1;
		rr_true.y = rr.center.y + start_y1;


		if (int(rr.center.y) >= mul_src_8.rows)
		{
			return false;
		}
		if (int(rr.center.x) >= mul_src_8.cols)
		{
			return false;
		}

		if ((rr.center.y < 0) || (rr.center.x < 0))
		{
			return false;
		}

		aa = mul_src_8.at<uchar>(int(rr.center.y), int(rr.center.x));


		if ((pan_src_8.rows / 2) >= pan_src_8.rows)
		{
			return false;
		}
		if ((pan_src_8.cols / 2) >= pan_src_8.cols)
		{

			return false;
		}
		bb = pan_src_8.at<uchar>(pan_src_8.rows / 2, pan_src_8.cols / 2);


		if (mul_src_8.at<uchar>(int(rr.center.y), int(rr.center.x)) > 2 && pan_src_8.at<uchar>(pan_src_8.rows / 2, pan_src_8.cols / 2)>2)
		{
			p01 = rr_true;
			p02 = p_true;
			one.push_back(rr.center);
			two.push_back(p);
			response_static = response;
		}
		else
		{
			return false;
		}

	}
	else
	{
		if (response<m_dResponseThLow)
		{
			return false;
		}

		cv::Point2d p(pan_src_8.cols / 2, pan_src_8.rows / 2);
		cv::Point2d p_true(pan_src_8.cols / 2 + start_x2, pan_src_8.rows / 2 + start_y2);

		cv::Point2d rr_true;
		rr_true.x = rr.center.x + start_x1;
		rr_true.y = rr.center.y + start_y1;

		if (int(rr.center.y) >= mul_src_8.rows)
		{
			return false;
		}
		if (int(rr.center.x) >= mul_src_8.cols)
		{
			return false;
		}

		aa = mul_src_8.at<uchar>(int(rr.center.y), int(rr.center.x));


		if ((pan_src_8.rows / 2) >= pan_src_8.rows)
		{
			return false;
		}
		if ((pan_src_8.cols / 2) >= pan_src_8.cols)
		{
			return false;
		}
		bb = pan_src_8.at<uchar>(pan_src_8.rows / 2, pan_src_8.cols / 2);


		if (mul_src_8.at<uchar>(int(rr.center.y), int(rr.center.x)) > 2 && pan_src_8.at<uchar>(pan_src_8.rows / 2, pan_src_8.cols / 2)>2)
		{
			p01 = rr_true;
			p02 = p_true;
			one.push_back(rr.center);
			two.push_back(p);
			response_static = response;
			return false; 
		}
		else
		{
			return false;
		}
	}




	return true;

}



bool FusionLYH::MatchPointShwo(cv::Mat A, cv::Mat B, std::vector<cv::Point2f> one, std::vector<cv::Point2f> two,
	std::string name)
{
	IplImage *dst_big;
	CvRect rect1 = cvRect(0, 0, A.cols, A.rows);
	CvRect rect2 = cvRect(A.cols, 0, B.cols, B.rows);
	IplImage *img1 = cvCloneImage(&(IplImage)A);
	IplImage *img2 = cvCloneImage(&(IplImage)B);
	int cols_big, rows_big;
	rows_big = A.rows > B.rows ? A.rows : B.rows;
	cols_big = A.cols + B.cols;
	int numofchanne = A.channels();
	dst_big = cvCreateImage(cvSize(cols_big, rows_big), IPL_DEPTH_8U, numofchanne);
	cvSetImageROI(dst_big, rect1);
	cvCopy(img1, dst_big);
	cvSetImageROI(dst_big, rect2);
	cvCopy(img2, dst_big);
	cvResetImageROI(dst_big);
	cv::Mat showimg = dst_big;
	int line_num = one.size();
	cv::RNG rng(0xFFFFFFFF);
	for (int i = 0; i < line_num; i++)
	{
		int a, b, c;
		a = 255 - i * 30;
		if (a > 255)
			a = 255;
		b = i * 30;
		if (b > 255)
			b = 255;
		c = 100 + i * 30;
		if (c > 255)
			c = 255;
		cvLine(dst_big, cvPoint(one[i].x, one[i].y), cvPoint(two[i].x + A.cols, two[i].y), CV_RGB(a, b, c), 5, 2);

	}
	IplImage *temp_img = cvCreateImage(cvSize(cols_big / 2, rows_big / 2), dst_big->depth, numofchanne);
	cvResize(dst_big, temp_img);
	IplImage *dst = cvCreateImage(cvSize(cols_big / 2, rows_big / 2), temp_img->depth, numofchanne);
	cvCopy(temp_img, dst);
	cv::Mat lookmat(dst);
	cvNamedWindow(name.c_str(), 0);
	imshow(name.c_str(), lookmat);

	cvWaitKey();

	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	cvReleaseImage(&temp_img);
	cvReleaseImage(&dst);
	cvReleaseImage(&dst_big);

	return true;
}

