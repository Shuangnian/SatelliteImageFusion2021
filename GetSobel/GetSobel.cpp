


#include <math.h>
#include <iostream>
#include <fstream>
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

#include "gdal_priv.h"
#include "gdal.h"
#include "gdalwarper.h"
#include "cpl_conv.h"
#include "ogrsf_frmts.h"

#pragma warning(disable:4996)

//function to compute sobel
int Sobel0821(const char * img, const char * output_img)//img is the MS image to compute sobel (16-bit 4 bands)
{                                                       //output_img is the sobel result image
	const char* pszSrcFile = img;
	const char *pszDstFile = output_img;
	
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALAllRegister();


	GDALDataset *hSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
	if (NULL == hSrcDS)
	{
		std::cout << "image error" << std::endl;
		return 1;
	}

	GDALDataType eDataType = GDALGetRasterDataType(GDALGetRasterBand(hSrcDS, 1));

	int nBandCount = GDALGetRasterCount(hSrcDS);
	int nImgwidth = hSrcDS->GetRasterXSize();
	int nImgheight = hSrcDS->GetRasterYSize();

	std::vector<GByte*> MSS4BandBuf(4);

	int bufnum = nImgwidth* nImgheight;

	for (int iband = 0; iband < 4; iband++)
	{ 
		GUInt16 *pImgData = new GUInt16[bufnum];
		GDALRasterBand  *pBuf_src = hSrcDS->GetRasterBand(iband + 1);
		pBuf_src->RasterIO(GF_Read, 0, 0, nImgwidth, nImgheight, pImgData, nImgwidth, nImgheight, eDataType, 0, 0);

		MSS4BandBuf[iband] = new GByte[bufnum];
		stretch_percent_16to8_data(pImgData, MSS4BandBuf[iband], nImgwidth, nImgheight);//stretch 16-bit to 8-bit

		delete[] pImgData;
	}

	std::vector<cv::Mat> MSS4BandMat8(4);
	for (int iband = 0; iband < 4; iband++)
	{
		MSS4BandMat8[iband] = cv::Mat::zeros(nImgheight, nImgwidth, CV_8UC1);

		for (int j=0;j<nImgheight;j++)
		{
			uchar * pMat8Data = MSS4BandMat8[iband].ptr<uchar>(j);
			for (int i = 0; i<nImgwidth; i++)
			{
				pMat8Data[i] = MSS4BandBuf[iband][j*nImgwidth + i];
			}
		}
	}


	std::vector<cv::Mat> MSS4BandMatGrad(4);
	for (int iband = 0; iband < 4; iband++)
	{
		//compute graadient
		cv::Mat gradImgX;
		cv::Mat gradImgY;
		cv::Sobel(MSS4BandMat8[iband], gradImgX, CV_16S, 1, 0, 3, 1, 0);
		cv::Sobel(MSS4BandMat8[iband], gradImgY, CV_16S, 0, 1, 3, 1, 0);

		gradImgX = cv::abs(gradImgX);
		gradImgY = cv::abs(gradImgY);

		cv::addWeighted(gradImgX, 0.5, gradImgY, 0.5, 0, MSS4BandMatGrad[iband]);
	}


	cv::Mat finalGrad = cv::Mat::zeros(MSS4BandMatGrad[0].rows, MSS4BandMatGrad[0].cols, CV_8UC1);
	for (int j = 0; j<MSS4BandMatGrad[0].rows; j++)
	{

		short * pGradData1 = MSS4BandMatGrad[0].ptr<short>(j);
		short * pGradData2 = MSS4BandMatGrad[1].ptr<short>(j);
		short * pGradData3 = MSS4BandMatGrad[2].ptr<short>(j);
		short * pGradData4 = MSS4BandMatGrad[3].ptr<short>(j);

		uchar * pFinalData = finalGrad.ptr<uchar>(j);

		for (int i = 0; i<MSS4BandMatGrad[0].cols; i++)
		{
			short temp1 = pGradData1[i];
			short temp2 = pGradData2[i];
			short temp3 = pGradData3[i];
			short temp4 = pGradData4[i];

			uchar tempMax = 0;
			if (temp1>tempMax)
			{
				tempMax = temp1;
			}
			if (temp2>tempMax)
			{
				tempMax = temp2;
			}
			if (temp3>tempMax)
			{
				tempMax = temp3;
			}
			if (temp4>tempMax)
			{
				tempMax = temp4;
			}

			if (tempMax>255)
			{
				tempMax = 255;
			}
			if (tempMax<0)
			{
				tempMax = 0;
			}
			pFinalData[i] = uchar(tempMax);

		}
	}


	cv::imwrite(output_img, finalGrad);



	for (int iband = 0; iband < 4; iband++)
	{
		delete[] MSS4BandBuf[iband];
	}




	GDALClose(hSrcDS);


	std::cout << "done！" << std::endl;
	return 0;
}


int stretch_percent_16to8_data(unsigned short *srcdata, uchar *outdata, int wid, int height)
{
	for (int iBand = 0; iBand < 1; iBand++) 
	{
		int src_max = 0, src_min = 65500;
		
		//get max 
		for (int src_row = 0; src_row < height; src_row++)
		{
			for (int src_col = 0; src_col < wid; src_col++)
			{
				uint16_t src_temVal = *(srcdata + src_row * wid + src_col);
				if (src_temVal > src_max)
					src_max = src_temVal;
				if (src_temVal < src_min)
					src_min = src_temVal;
			}
		}

		double *numb_pix = (double *)malloc(sizeof(double)*(src_max + 1));      
		memset(numb_pix, 0, sizeof(double) * (src_max + 1));
		
		//get histogram
#pragma omp parallel for
		for (int src_row = 0; src_row < height; src_row++)
		{
			for (int src_col = 0; src_col < wid; src_col++)
			{
				uint16_t src_temVal = *(srcdata + src_row * wid + src_col);
				*(numb_pix + src_temVal) += 1;
			}
		}

		double *frequency_val = (double *)malloc(sizeof(double)*(src_max + 1));           //pixel value frequency
		memset(frequency_val, 0.0, sizeof(double)*(src_max + 1));
#pragma omp parallel for
		for (int val_i = 0; val_i <= src_max; val_i++)
		{
			*(frequency_val + val_i) = *(numb_pix + val_i) / double(wid * height);
		}

		double *accumlt_frequency_val = (double*)malloc(sizeof(double)*(src_max + 1));   //accumulated frequency
		memset(accumlt_frequency_val, 0.0, sizeof(double)*(src_max + 1));
#pragma omp parallel for
		for (int val_i = 0; val_i <= src_max; val_i++)
		{
			for (int val_j = 0; val_j < val_i; val_j++)
			{
				*(accumlt_frequency_val + val_i) += *(frequency_val + val_j);
			}
		}
		
		int minVal = 0, maxVal = 0;
		for (int val_i = 1; val_i < src_max; val_i++)
		{
			double acc_fre_temVal0 = *(frequency_val + 0);
			double acc_fre_temVal = *(accumlt_frequency_val + val_i);
			if ((acc_fre_temVal - acc_fre_temVal0) > 0.0015)
			{
				minVal = val_i;
				break;
			}
		}
		for (int val_i = src_max - 1; val_i > 0; val_i--)
		{
			double acc_fre_temVal0 = *(accumlt_frequency_val + src_max);
			double acc_fre_temVal = *(accumlt_frequency_val + val_i);
			if (acc_fre_temVal < (acc_fre_temVal0 - 0.00012))
			{
				maxVal = val_i;
				break;
			}
		}
		
#pragma omp parallel for
		for (int src_row = 0; src_row < height; src_row++)
		{
			for (int src_col = 0; src_col < wid; src_col++)
			{
				uint16_t src_temVal = *(srcdata + src_row * wid + src_col);
				double stre_temVal = (src_temVal - minVal) / double(maxVal - minVal);
				if (src_temVal < minVal)
				{
					*(outdata + src_row*wid + src_col) = (src_temVal) *(20.0 / double(minVal));
				}
				else if (src_temVal > maxVal)
				{
					stre_temVal = (src_temVal - src_min) / double(src_max - src_min);
					*(outdata + src_row*wid + src_col) = 254;
				}
				else
					*(outdata + src_row*wid + src_col) = pow(stre_temVal, 0.7) * 250;

			}

		}

		free(numb_pix);
		free(frequency_val);
		free(accumlt_frequency_val);


	}
	return 0;
}

