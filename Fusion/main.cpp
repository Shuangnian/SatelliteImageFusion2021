
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <stdio.h>

#include "FusionLYH.h"


//fusion main func
int main(int argc, char* argv[])
{
	//get input param
	//original MS image path
	char pcMSSImgPath[1024];
	memset(pcMSSImgPath, '\0', sizeof(pcMSSImgPath));
	strcat_s(pcMSSImgPath, argv[1]);

	//original PAN image path
	char pcPANImgPath[1024];
	memset(pcPANImgPath, '\0', sizeof(pcPANImgPath));
	strcat_s(pcPANImgPath, argv[2]);

	//PAN image after gradient simulation
	char pcPANAdjImgPath[1024];
	memset(pcPANAdjImgPath, '\0', sizeof(pcPANAdjImgPath));
	strcat_s(pcPANAdjImgPath, argv[3]);

	//building factor image 
	char pcBuildingWeightPath[1024];
	memset(pcBuildingWeightPath, '\0', sizeof(pcBuildingWeightPath));
	strcat_s(pcBuildingWeightPath, argv[4]);

	//building factor image resized to the same size with the MS image
	char pcBuildingWeightDownSampPath[1024];
	memset(pcBuildingWeightDownSampPath, '\0', sizeof(pcBuildingWeightDownSampPath));
	strcat_s(pcBuildingWeightDownSampPath, argv[5]);

	//result report
	char pcTxtReportPath[1024];
	memset(pcTxtReportPath, '\0', sizeof(pcTxtReportPath));
	strcat_s(pcTxtReportPath, argv[6]);

	//fused image output path
	char pcFusionResultImgPath[1024];
	memset(pcFusionResultImgPath, '\0', sizeof(pcFusionResultImgPath));
	strcat_s(pcFusionResultImgPath, argv[7]);
	
	//GDAL data folder path
	char gdalDataFolder[1024];
	memset(gdalDataFolder, '\0', sizeof(gdalDataFolder));
	strcat_s(gdalDataFolder, argv[8]);

	clock_t startTime, endTime;
	startTime = clock();


	FusionLYH * pFusionLYH = new FusionLYH(pcMSSImgPath, pcPANImgPath, pcPANAdjImgPath,pcBuildingWeightPath,
		pcBuildingWeightDownSampPath, pcTxtReportPath,
		pcFusionResultImgPath, gdalDataFolder);
	pFusionLYH->Run();

	endTime = clock();
	double timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	std::cout << "  total time: " << timeCost << " s" << std::endl;

	return 0;
}

