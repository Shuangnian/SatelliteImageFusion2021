#include <stdio.h>
#include <iostream>
#include <math.h>
#include <ctime>

#include "gdal_priv.h"
#include "ogrsf_frmts.h"  
#include "gdalwarper.h"  

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/morphological_filter.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>

#include "opencv2\imgproc.hpp"
#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"



//compute building factor

/////////////////////////////////////////////////////////////////
//function

//delete buildings with invalid areas
cv::Mat DeleteSmallBigArea(cv::Mat segImg, int nMaxBuildingSegArea, int nMinBuildingSegArea);

//convert to point cloud to compute distance
void ContourToPtCloud(cv::Mat segImg, std::vector<double> & outputX, std::vector<double> & outputY);


/////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	/////////////////////////////////////////////////////////////////
	//
	char segBuildingDOMPath[1024];//building segmented results
	memset(segBuildingDOMPath, '\0', sizeof(segBuildingDOMPath));
	strcat_s(segBuildingDOMPath, argv[1]);

	char outputPath[1024];//building factor result output
	memset(outputPath, '\0', sizeof(outputPath));
	strcat_s(outputPath, argv[2]);

	/////////////////////////////////////////////////////////////////
	//parameters and thresholds

	double dMovingWindowSize = 400;// in meters

	double dDSMGSD = 0.8;//resolution

	double dDth = 50;//a constant, can change with different sensors

	int nMaxBuildingSegAreaPix = 220000;//building max area  in pixel
	
	int nMinBuildingSegAreaPix = 60;//building min area  in pixel
	
	//////////////////////////////////////////////////////////////////////////

	GDALAllRegister(); 

	CPLSetConfigOption("GDAL_FILENAME_IS_UTF8", "NO");

	//first remove buildings with invalid areas 
	cv::Mat segBuildingImg = cv::imread(segBuildingDOMPath, cv::IMREAD_GRAYSCALE); 

	segBuildingImg = DeleteSmallBigArea(segBuildingImg, nMaxBuildingSegAreaPix, nMinBuildingSegAreaPix);

	//convert to point cloud
	std::vector<double> buildingCenterX;
	std::vector<double> buildingCenterY;
	ContourToPtCloud(segBuildingImg, buildingCenterX, buildingCenterY);

	int nMovingWindowSizePix = int(dMovingWindowSize / dDSMGSD);//radius for moving window

	pcl::PointCloud<pcl::PointXYZ>::Ptr buildingCloud2D(new pcl::PointCloud<pcl::PointXYZ>);
	buildingCloud2D->width = buildingCenterX.size();
	buildingCloud2D->height = 1;
	buildingCloud2D->is_dense = true; 
	buildingCloud2D->points.resize(buildingCloud2D->width*buildingCloud2D->height);
	for (int i = 0; i < buildingCenterX.size(); i++)
	{
		buildingCloud2D->points[i].x = buildingCenterX[i];//pixel coordinate
		buildingCloud2D->points[i].y = buildingCenterY[i];
		buildingCloud2D->points[i].z = 0;
	}

	//create a KD Tree
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(buildingCloud2D);


	//building weight
	float *pBuildingWeight = new float[segBuildingImg.rows*segBuildingImg.cols];
	for (int i=0;i<segBuildingImg.rows;i++)
	{
		unsigned char * pSegRow = segBuildingImg.ptr<unsigned char>(i);
		for (int j = 0; j<segBuildingImg.cols; j++)
		{
			unsigned char temp = pSegRow[j];

			if (temp>0)//if belongs to built-up area
			{
				pBuildingWeight[i*segBuildingImg.cols + j] = 1;
			}
			else
			{
				//window center
				pcl::PointXYZ pt;
				pt.x = j;
				pt.y = i;
				pt.z = 0;

				//create kd tree for searching
				std::vector<int> pointIdxRadiusSearch;
				std::vector<float> pointRadiusSquaredDistance;
				kdtree.radiusSearch(pt, nMovingWindowSizePix, pointIdxRadiusSearch, pointRadiusSquaredDistance);

				if (pointIdxRadiusSearch.size()==0)//no result
				{
					pBuildingWeight[i*segBuildingImg.cols + j] = 0;
				}
				else
				{
					float fWeight = 0;

					for (int k=0;k<pointIdxRadiusSearch.size();k++)
					{
						float tempWeight = dDth / (sqrt(pointRadiusSquaredDistance[k])*dDSMGSD);
						if (tempWeight > fWeight)
						{
							fWeight = tempWeight;
						}
	
					}

					if (fWeight > 1)
					{
						fWeight = 1;
					}

					pBuildingWeight[i*segBuildingImg.cols + j] = fWeight;
				}


			}




		}
	}

	//output result
	const char* imageFormat = "GTiff";
	GDALDriver* gdalDriver = GetGDALDriverManager()->GetDriverByName(imageFormat);
	if (gdalDriver == NULL)
	{
		std::cout << "error！" << '\n';
		return EXIT_FAILURE;
	}
	GDALDataset* pOutput;
	pOutput = gdalDriver->Create(outputPath, segBuildingImg.cols,segBuildingImg.rows, 1,
		GDT_Float32, NULL);


	pOutput->GetRasterBand(1)->RasterIO(GF_Write, 0, 0,
		segBuildingImg.cols, segBuildingImg.rows, pBuildingWeight,
		segBuildingImg.cols, segBuildingImg.rows, GDT_Float32, 0, 0);


	delete[] pBuildingWeight;
	
	GDALClose(pOutput);
	GDALDestroyDriverManager();

	return 0;
}





cv::Mat DeleteSmallBigArea(cv::Mat segImg, int nMaxBuildingSegArea, int nMinBuildingSegArea)
{
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(segImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	for (int i=0;i<contours.size();i++)
	{
		double dContourArea = abs(cv::contourArea(contours[i]));

		if ((dContourArea<nMinBuildingSegArea))
		{
			cv::drawContours(segImg, contours, i, cv::Scalar(0), CV_FILLED);
		}
		if ((dContourArea>nMaxBuildingSegArea))
		{
			cv::drawContours(segImg, contours, i, cv::Scalar(0), CV_FILLED);
		}
	}
	
	return segImg;
}


void ContourToPtCloud(cv::Mat segImg, std::vector<double> & outputX, std::vector<double> & outputY)
{
	std::vector<std::vector<cv::Point>> contours;

	cv::findContours(segImg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++)
	{
		double dMeanX = 0;
		double dMeanY = 0;

		for (int j=0;j<contours[i].size();j++)
		{
			dMeanX = dMeanX + contours[i][j].x;
			dMeanY = dMeanY + contours[i][j].y;
		}

		dMeanX = dMeanX / contours[i].size();
		dMeanY = dMeanY / contours[i].size();

		outputX.push_back(dMeanX);
		outputY.push_back(dMeanY);


	}

}



