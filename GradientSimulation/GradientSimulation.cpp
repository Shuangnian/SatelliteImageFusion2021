
#include <stdio.h>
#include <iostream>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"  
#include "gdalwarper.h"  

//
#define NODATA_VALUE 0

//this exe is used in cmd

int main(int argc, char* argv[])
{
	//get exe input parameters
	char MSSPath[1024];//Upsampled MSS image
	memset(MSSPath, '\0', sizeof(MSSPath));
	strcat_s(MSSPath, argv[1]);

	char PANPath[1024];//Pan image
	memset(PANPath, '\0', sizeof(PANPath));
	strcat_s(PANPath, argv[2]);

	char MaskPath[1024];// sobel results
	memset(MaskPath, '\0', sizeof(MaskPath));
	strcat_s(MaskPath, argv[3]);

	//output Pan
	char outputPanPath[1024];
	memset(outputPanPath, '\0', sizeof(outputPanPath));
	strcat_s(outputPanPath, argv[4]);

	GDALAllRegister();  //注册所有的驱动

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	GDALDataset *poCLASS1;   //GDAL数据集  已释放
	poCLASS1 = (GDALDataset *)GDALOpen(MSSPath, GA_ReadOnly);
	if (poCLASS1 == NULL)
	{
		std::cout << "error！" << std::endl;
		return 1;
	}

	//get image size
	int lImgSizeX = poCLASS1->GetRasterXSize();
	int lImgSizeY = poCLASS1->GetRasterYSize();
	long lPixelCount = lImgSizeX*lImgSizeY;

	//read MSS image   4 bands
	GDALRasterBand *poBand1;
	poBand1 = poCLASS1->GetRasterBand(1);
	unsigned short *buffer1;  
	buffer1 = new unsigned short[lPixelCount];
	poBand1->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, buffer1, lImgSizeX, lImgSizeY,
		GDALDataType(poBand1->GetRasterDataType()), 0, 0);

	GDALRasterBand *poBand2; 
	poBand2 = poCLASS1->GetRasterBand(2);
	unsigned short *buffer2;  
	buffer2 = new unsigned short[lPixelCount];
	poBand2->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, buffer2, lImgSizeX, lImgSizeY,
		GDALDataType(poBand2->GetRasterDataType()), 0, 0);

	GDALRasterBand *poBand3; 
	poBand3 = poCLASS1->GetRasterBand(3);
	unsigned short *buffer3;  
	buffer3 = new unsigned short[lPixelCount];
	poBand3->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, buffer3, lImgSizeX, lImgSizeY,
		GDALDataType(poBand3->GetRasterDataType()), 0, 0);

	GDALRasterBand *poBand4; 
	poBand4 = poCLASS1->GetRasterBand(4);
	unsigned short *buffer4;  
	buffer4 = new unsigned short[lPixelCount];
	poBand4->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, buffer4, lImgSizeX, lImgSizeY,
		GDALDataType(poBand4->GetRasterDataType()), 0, 0);



	//reaad Pan image
	GDALDataset *poCLASS2;  
	poCLASS2 = (GDALDataset *)GDALOpen(PANPath, GA_ReadOnly);
	if (poCLASS2 == NULL)
	{
		std::cout << "error！" << std::endl;
		return 1;
	}
	
	GDALRasterBand *poBandPan; 
	poBandPan = poCLASS2->GetRasterBand(1);

	unsigned short *bufferPan;  
	bufferPan = new unsigned short[lPixelCount];
	poBandPan->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, bufferPan, lImgSizeX, lImgSizeY,
		GDALDataType(poBandPan->GetRasterDataType()), 0, 0);

	unsigned short *bufferPanNew;  
	bufferPanNew = new unsigned short[lPixelCount];


	//read sobel as mask
	GDALDataset *poMask;  
	poMask = (GDALDataset *)GDALOpen(MaskPath, GA_ReadOnly);
	unsigned char * bufferMask = new unsigned char[lPixelCount];
	poMask->GetRasterBand(1)->RasterIO(GF_Read, 0, 0, lImgSizeX, lImgSizeY, bufferMask, lImgSizeX, lImgSizeY,
		GDT_Byte, 0, 0);

	int nGradMinThreshold = 27;//high gradient threshold

	for (int j = 1; j < lImgSizeY; j++)
	{
		for (int i = 1; i < lImgSizeX; i++)
		{
			unsigned char gradMask = bufferMask[j*lImgSizeX + i];//梯度

			if (gradMask<nGradMinThreshold)
			{
				continue;
			}

			// current pixel and left up pixels
			int panCurPix = bufferPan[j*lImgSizeX + i];
			int panCurPixLeft = bufferPan[j*lImgSizeX + i-1];
			int panCurPixUp = bufferPan[(j - 1)*lImgSizeX + i];

			if ((panCurPix==0) || (panCurPixLeft == 0) || (panCurPixUp == 0))
			{
				continue;
			}

			int dXp = int(bufferPan[j*lImgSizeX + i - 1]) - int(bufferPan[j*lImgSizeX + i]);
			int dYp = int(bufferPan[(j - 1)*lImgSizeX + i]) - int(bufferPan[j*lImgSizeX + i]);

			int dXm1 = int(buffer1[j*lImgSizeX + i - 1]) - int(buffer1[j*lImgSizeX + i]);
			int dYm1 = int(buffer1[(j-1)*lImgSizeX + i]) - int(buffer1[j*lImgSizeX + i]);

			int dXm2 = int(buffer2[j*lImgSizeX + i - 1]) - int(buffer2[j*lImgSizeX + i]);
			int dYm2 = int(buffer2[(j - 1)*lImgSizeX + i]) - int(buffer2[j*lImgSizeX + i]);

			int dXm3 = int(buffer3[j*lImgSizeX + i - 1]) - int(buffer3[j*lImgSizeX + i]);
			int dYm3 = int(buffer3[(j - 1)*lImgSizeX + i]) - int(buffer3[j*lImgSizeX + i]);

			int dXm4 = int(buffer4[j*lImgSizeX + i - 1]) - int(buffer4[j*lImgSizeX + i]);
			int dYm4 = int(buffer4[(j - 1)*lImgSizeX + i]) - int(buffer4[j*lImgSizeX + i]);


			int adjPix = ((((dXm1 + dXm2 + dXm3 + dXm4)*1.0 / 4.0)- dXp) + (((dYm1 + dYm2 + dYm3 + dYm4)*1.0 / 4.0) - dYp))/2;

			int newPanPix = int(bufferPan[j*lImgSizeX + i]) - adjPix;
			if (newPanPix<=0)
			{
				continue;
			}
			else
			{
				bufferPan[j*lImgSizeX + i] = unsigned short(newPanPix);
			}

		}
	}

	//output
	const char* imageFormat = "GTiff";
	GDALDriver* gdalDriver = GetGDALDriverManager()->GetDriverByName(imageFormat);
	if (gdalDriver == NULL)
	{
		std::cout << "error！" << '\n';
		return EXIT_FAILURE;
	}

	// geo ref
	double goeInformation[6];
	poCLASS1->GetGeoTransform(goeInformation);

	const char* gdalProjection = poCLASS1->GetProjectionRef();

	GDALDataset* outputDataset1;
	outputDataset1 = gdalDriver->Create(outputPanPath, lImgSizeX, lImgSizeY, 1, GDT_UInt16, NULL);
	outputDataset1->SetGeoTransform(goeInformation);
	outputDataset1->SetProjection(gdalProjection);
	GDALRasterBand* outputRasterBand1 = outputDataset1->GetRasterBand(1);
	outputRasterBand1->RasterIO(GF_Write, 0, 0, lImgSizeX, lImgSizeY, bufferPan, lImgSizeX, lImgSizeY, GDT_UInt16, 0, 0); // write result
	outputRasterBand1->SetNoDataValue(NODATA_VALUE);


	GDALClose(poCLASS1);
	GDALClose(poCLASS2);
	GDALClose(outputDataset1);
	GDALClose(poMask);

	GDALDestroyDriverManager();

	delete[] buffer1;
	delete[] buffer2;
	delete[] buffer3;
	delete[] buffer4;
	delete[] bufferPan;
	delete[] bufferPanNew;
	delete[] bufferMask;

	std::cout << "Done" << std::endl;

	return 0;
}

