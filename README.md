# SatelliteImageFusion2021
Code reuqires:
OpenCV 2.4
GDAL 2.2
PCL 1.8

Development environments:
MicroSoft Visual Studio 2015 C++
Windows 10 x64
16G RAM

Datasets: 16 bit images, MS images have 4 bands: blue, green, red, near-infrared (B G R NIR)
By Jilin-1 Satellite (WGS 1984)

Note: 
1. Please note the size of upsampled MS image, whether it is the same with the PAN image. If not, can use our code "Resize" to change the size.
2. Please note the fused result of ArcGIS GS method is RGB, not BGR, which can be changed to BGR with our code "ChangeBandIndex". 
3. The building segmentation is a pre-trained exe file, the code is from YOLACT.
4. Code only support image in longitude-latitude coordinate system. It is easy to edit the code if you want to run in projected coordinates.


Chang Guang Satellite Technology Co., Ltd., No. 1299 Mingxi Rd., Changchun, Jilin, China.

Contact: yihui_li_whu@qq.com
