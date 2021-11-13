
#include <iostream>
#include <fstream>
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

#include "gdal_priv.h"
#include "gdal.h"
#include "gdalwarper.h"
#include "cpl_conv.h"
#include "ogrsf_frmts.h"
#include <math.h>

#pragma warning(disable:4996)



//change the input image's band index, RGB to BGR
int GeometricCorrectionLYH::ChangeBandIndex(const char * img, const char * output_img)
{
	const char* pszSrcFile = img;
	const char *pszDstFile = output_img;
	
	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");
	GDALAllRegister();


	GDALDataset *hSrcDS = (GDALDataset*)GDALOpen(pszSrcFile, GA_ReadOnly);
	if (NULL == hSrcDS)
	{
		std::cout << "error" << std::endl;
		return 1;
	}

	GDALDataType eDataType = GDALGetRasterDataType(GDALGetRasterBand(hSrcDS, 1));
	int nBandCount = GDALGetRasterCount(hSrcDS);
	int nImgwidth = hSrcDS->GetRasterXSize();
	int nImgheight = hSrcDS->GetRasterYSize();

	GDALDriver *pDsmDriver = NULL;
	pDsmDriver = GetGDALDriverManager()->GetDriverByName("GTiff");
	if (pDsmDriver == NULL)
	{
		std::cout << "error" << std::endl;
		return 5;
	}


	GDALDataset *hDstDS;
	hDstDS = pDsmDriver->Create(pszDstFile, nImgwidth, nImgheight, nBandCount, GDT_UInt16, NULL);
	if (hDstDS == NULL)
	{
		std::cout << "error" << std::endl;
		return 6;
	}

	const char * PROdata_Pan = hSrcDS->GetProjectionRef();
	double GEOpara_Pan[6];
	hSrcDS->GetGeoTransform(GEOpara_Pan);

	hDstDS->SetGeoTransform(GEOpara_Pan);
	hDstDS->SetProjection(PROdata_Pan);


	//process band by band
	for (int iband = 0; iband < nBandCount; iband++)
	{
		clock_t startTime, endTime;
		startTime = clock();

		std::cout << "compute band" << iband + 1 << std::endl;

		int srcIndex = 0;
		int dstIndex = 0;
		if (iband<3)
		{
			srcIndex = iband + 1;
			dstIndex = 3 - iband;
		}
		else
		{
			srcIndex = iband + 1;
			dstIndex = iband + 1;
		}

		GDALRasterBand  *pBuf_src = hSrcDS->GetRasterBand(srcIndex);
		GDALRasterBand  *pBuf_dst = hDstDS->GetRasterBand(dstIndex);

		int bufnum = nImgwidth* nImgheight;
		GUInt16 *pImgData = new GUInt16[bufnum];

		pBuf_src->RasterIO(GF_Read, 0, 0, nImgwidth, nImgheight, pImgData, nImgwidth, nImgheight, GDT_UInt16, 0, 0);


		pBuf_dst->RasterIO(GF_Write, 0, 0, nImgwidth, nImgheight, pImgData, nImgwidth, nImgheight, GDT_UInt16, 0, 0);


		if (pImgData) delete[]pImgData;




	}
	GDALClose(hSrcDS);
	GDALClose(hDstDS);

	GetGDALDriverManager()->DeregisterDriver(pDsmDriver);
	std::cout << "done" << std::endl;
	return 0;
}







































//通过底图匹配获取控制信息
int GeometricCorrectionLYH::stretch_percent_16to8_data(unsigned short *srcdata, uchar *outdata, int wid, int height)
{
	//cout << "16位数据指针降到8位数据指针..." << endl;
	for (int iBand = 0; iBand < 1; iBand++) //这个循环好像不需要吧？李一挥标注
	{
		//cout << "正在处理第 " << iBand + 1 << " 波段" << endl;
		//读取影像
		//uint16_t *srcData = (uint16_t *)malloc(sizeof(uint16_t) *wid * height * 1);
		//memset(srcData, 0, sizeof(uint16_t) * 1 * wid * height);
		int src_max = 0, src_min = 65500;
		//读取多光谱影像到缓存
		//poIn->GetRasterBand(iBand + 1)->RasterIO(GF_Read, 0, 0, wid, height, srcData + 0 * wid * height, wid, height, GDT_UInt16, 0, 0);
		//}
		//统计最大值
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

		double *numb_pix = (double *)malloc(sizeof(double)*(src_max + 1));      //存像素值直方图，即每个像素值的个数
		memset(numb_pix, 0, sizeof(double) * (src_max + 1));
		//                 -------  统计像素值直方图  ------------         //
#pragma omp parallel for
		for (int src_row = 0; src_row < height; src_row++)
		{
			for (int src_col = 0; src_col < wid; src_col++)
			{
				uint16_t src_temVal = *(srcdata + src_row * wid + src_col);
				*(numb_pix + src_temVal) += 1;
			}
		}

		double *frequency_val = (double *)malloc(sizeof(double)*(src_max + 1));           //像素值出现的频率
		memset(frequency_val, 0.0, sizeof(double)*(src_max + 1));
#pragma omp parallel for
		for (int val_i = 0; val_i <= src_max; val_i++)
		{
			*(frequency_val + val_i) = *(numb_pix + val_i) / double(wid * height);
		}

		double *accumlt_frequency_val = (double*)malloc(sizeof(double)*(src_max + 1));   //像素出现的累计频率
		memset(accumlt_frequency_val, 0.0, sizeof(double)*(src_max + 1));
#pragma omp parallel for
		for (int val_i = 0; val_i <= src_max; val_i++)
		{
			for (int val_j = 0; val_j < val_i; val_j++)
			{
				*(accumlt_frequency_val + val_i) += *(frequency_val + val_j);
			}
		}
		//统计像素两端截断值
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
			//uint8_t *dstData = (uint8_t*)malloc(sizeof(uint8_t)*wid);
			//memset(dstData, 0, sizeof(uint8_t)*wid);
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





std::string GeometricCorrectionLYH::get_date_time_now_here()
{
	time_t now;
	struct tm *fmt;
	time(&now);
	fmt = localtime(&now);

	std::stringstream data_now_ss;
	data_now_ss << "-"
		<< (fmt->tm_year + 1900)
		<< "-"
		<< (fmt->tm_mon + 1)
		<< "-"
		<< (fmt->tm_mday)
		<< "-"
		<< (fmt->tm_hour)
		<< "-"
		<< (fmt->tm_min)
		<< "-"
		<< (fmt->tm_sec)
		<< "-";
	return data_now_ss.str();

}

