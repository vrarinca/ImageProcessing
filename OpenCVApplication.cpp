#include "stdafx.h"
#include "common.h"
#include <queue> 

using namespace std;
int factor;
void testOpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image", src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName) == 0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName, "bmp");
	while (fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(), src);
		if (waitKey() == 27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}
void negative_image() {
	Mat img = imread("Images/kids.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			img.at<uchar>(i, j) = 255 - img.at<uchar>(i, j);
		}
	}
	imshow("negative image", img);
	waitKey(0);
}
void testNegativeImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar neg = 255 - val;
				dst.at<uchar>(i, j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int)src.step; // no dword alignment is done !!!
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i, j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i, j) = (r + g + b) / 3;
			}
		}

		imshow("input image", src);
		imshow("gray image", dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1, dst2;
		//without interpolation
		resizeImg(src, dst1, 320, false);
		//with interpolation
		resizeImg(src, dst2, 320, true);
		imshow("input image", src);
		imshow("resized image (without interpolation)", dst1);
		imshow("resized image (with interpolation)", dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}

	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame, edges, 40, 100, 3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n");
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];

	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;

		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115) { //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess)
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

bool hasWhiteNeighbours(Mat img, int i, int j) {
	Vec3b white = Vec3b(255, 255, 255);
	if (img.at<Vec3b>(i, j) == white)
		return true;
	if (img.at<Vec3b>(i - 1, j - 1) == white)
		return true;
	if (img.at<Vec3b>(i - 1, j) == white)
		return true;
	if (img.at<Vec3b>(i - 1, j + 1) == white)
		return true;
	if (img.at<Vec3b>(i, j - 1) == white)
		return true;
	if (img.at<Vec3b>(i, j + 1) == white)
		return true;
	if (img.at<Vec3b>(i + 1, j - 1) == white)
		return true;
	if (img.at<Vec3b>(i + 1, j) == white)
		return true;
	if (img.at<Vec3b>(i + 1, j + 1) == white)
		return true;
	return false;
}


void lab41(Mat img, int B, int G, int R) {


	// b = pixel[0];   //b
	// g = pixel[1];   //g
	// r = pixel[2];   //r 
	int area = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == B &&
				img.at<Vec3b>(i, j)[1] == G &&
				img.at<Vec3b>(i, j)[2] == R) {
				area++;
			}
		}
	}
	printf("Area: %d", area);
	float r = 0;
	float c = 0;
	//center of mass
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == B &&
				img.at<Vec3b>(i, j)[1] == G &&
				img.at<Vec3b>(i, j)[2] == R) {
				r += i;
				c += j;
			}
		}
	}

	r = r / area;
	c = c / area;

	printf("Center of mass: %d %d", r, c);
	//axis of elongation
	int up = 0;
	int dC = 0;
	int dR = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img.at<Vec3b>(i, j)[0] == B &&
				img.at<Vec3b>(i, j)[1] == G &&
				img.at<Vec3b>(i, j)[2] == R) {
				up += 2 * (i - r) * (j - c);
				dC += (j - c) * (j - c);
				dR += (i - r) * (i - r);
			}
		}
	}

	float elong = (atan2(2 * up, (dC - dR)) / 2) / PI * 180 + 180;
	printf("Elongation : %f\d", elong);

	int per = 0;
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img. cols - 1; j++) {
			if (img.at<Vec3b>(i, j)[0] == B &&
				img.at<Vec3b>(i, j)[1] == G &&
				img.at<Vec3b>(i, j)[2] == R &&
				hasWhiteNeighbours(img, i, j))
				per++;

		}
	}

	printf("Perimeter: %d", per);

}
void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Mat img = imread("Images/oval_vert.bmp", CV_LOAD_IMAGE_COLOR);
		lab41(img, img.at<Vec3b>(y, x)[0], img.at<Vec3b>(y, x)[1], img.at<Vec3b>(y, x)[2]);
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
	}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
void additive_gray(int factor)
{

	Mat_<uchar> img = imread("Images/saturn.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) + factor < 255 && img.at<uchar>(i, j) + factor>0)
				img.at<uchar>(i, j) = img.at<uchar>(i, j) + factor;
			else if (img.at<uchar>(i, j) + factor > 255)
				img.at<uchar>(i, j) = 255;
			else
				img.at<uchar>(i, j) = 0;
		}
	}
	imshow("additive_grey", img);
	waitKey(0);
}
void create_squares()
{
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (i < img.rows / 2 && j < img.cols / 2)
				img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			if (i < img.rows / 2 && j >= img.cols / 2)
				img.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
			if (i > img.rows / 2 && j < img.cols / 2)
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 0);
			if (i > img.rows / 2 && j >= img.cols / 2)
				img.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
		}
	}
	imshow("create_squares", img);
	waitKey(0);
}
void inverse_matrix()
{
	Mat img(3, 3, CV_8UC3);
	img.inv();
}
void ex1()
{
	Mat_<Vec3b> img = imread("Images/stops.png", CV_LOAD_IMAGE_COLOR);
	Mat img1(img.rows, img.cols, CV_8UC1);
	Mat img2(img.rows, img.cols, CV_8UC1);
	Mat img3(img.rows, img.cols, CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b a;
			a = img.at<Vec3b>(i, j);
			img1.at<uchar>(i, j) = a[0];
			img2.at<uchar>(i, j) = a[1];
			img3.at<uchar>(i, j) = a[2];
		}
	}

	//	imwrite("newpic1.bmp", img1);
	imshow("ex1", img1);
	//imwrite("newpic1.bmp", img2);
	imshow("ex2", img2);
	//imwrite("newpic1.bmp", img3);
	imshow("ex3", img3);
	waitKey(0);
}
void ex2()
{
	Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat img1(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b a;
			a = img.at<Vec3b>(i, j);
			img1.at<uchar>(i, j) = (a[0] + a[1] + a[2]) / 3;

		}
	}

	//	imwrite("newpic1.bmp", img1);
	imshow("ex2", img1);
	imshow("a", img);

	waitKey(0);
}
void gray_to_binary()
{
	Mat_<uchar> img = imread("Images/Lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img1(img.rows, img.cols, CV_8UC1);

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) > 128)
				img1.at<uchar>(i, j) = 255;
			else
				img1.at<uchar>(i, j) = 0;
		}
	}

	//	imwrite("newpic1.bmp", img1);
	imshow("ex2", img1);
	imshow("a", img);

	waitKey(0);
}
void hsv_values()
{
	Mat_<Vec3b> img = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);
	Mat img1(img.rows, img.cols, CV_8UC1);
	Mat img2(img.rows, img.cols, CV_8UC1);
	Mat img3(img.rows, img.cols, CV_8UC1);
	float r, g, b, R, G, B, M, m, C, V, S, H;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b a;
			a = img.at<Vec3b>(i, j);
			B = a[0];
			G = a[1];
			R = a[2];
			r = R / 255; // r : the normalized R component
			g = G / 255; // g : the normalized G component
			b = B / 255; // b : the normalized B component
			// Attention: please declare all variables as float
			// If you have declared R as uchar, you have to use a cast: r = (float)R/255 !!!
			M = max(max(r, g), b);
			m = min(min(r, g), b);
			C = M - m;

			V = M;

			if (V != 0)
				S = C / V;
			else // grayscale
				S = 0;

			if (C != 0) {
				if (M == r) H = 60 * (g - b) / C;
				if (M == g) H = 120 + 60 * (b - r) / C;
				if (M == b) H = 240 + 60 * (r - g) / C;
			}
			else // grayscale
				H = 0;
			if (H < 0)
				H = H + 360;
			img1.at<uchar>(i, j) = H * 255 / 360;
			img2.at<uchar>(i, j) = S * 255;
			img3.at<uchar>(i, j) = V * 255;

		}
	}
	imshow("ex11", img);
	imshow("ex1", img1);
	//imwrite("newpic1.bmp", img2);
	imshow("ex2", img2);
	//imwrite("newpic1.bmp", img3);
	imshow("ex3", img3);
	waitKey(0);
}
bool isInside()
{
	int posi, posj;
	scanf("%d", &posi);
	scanf("%d", &posj);
	Mat_<uchar> img = imread("Images/Lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (posi < img.rows && posi > 0 && posj < img.cols && posj > 0)
		return 1;
	else return 0;
}
void multiplicative_gray(int factor)
{

	Mat_<uchar> img = imread("Images/Lena_gray.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.at<uchar>(i, j) * factor < 255)
				img.at<uchar>(i, j) = img.at<uchar>(i, j) + factor;
			else
				img.at<uchar>(i, j) = 255;
		}
	}
	imshow("multiplicative_grey", img);
	imwrite("newpic.bmp", img);
	waitKey(0);
}

/*
void showHistogram(const string& name, int* hist, const int hist_cols, const int hist_height)
{
Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255));
int max_hist = 0;
for (int i = 0; i < hist_cols; i++)
if (hist[i] > max_hist)
max_hist = hist[i];
double scale = 1.0;
scale = (double)hist_height / max_hist;
int baseline = hist_height - 1;
for (int x = 0; x < hist_cols; x++)
{
Point p1 = Point(x, baseline);
Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
line(imgHist, p1, p2, CV_RGB(255, 255, 255));
}
imshow(name, imgHist);
}
*/

void vecHist()
{
	Mat img = imread("Images/cameraman.bmp", 1);
	int vec[256] = { 0 };

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			vec[img.at<uchar>(i, j)]++;
		}
	printf("The vector is:");
	for (int i = 0; i < 256; i++)
	{
		printf(" %d", vec[i]);
	}
	printf("PDF:\n");
	for (int i = 0; i < 256; i++)
	{
		printf("%f", (float)vec[i] / (img.rows*img.cols));
	}
	showHistogram("Images/cameraman.bmp", vec, img.cols, img.rows);
	//system("pause");
	waitKey(0);
}

void multi_level_thresh() {
	char fname[MAX_PATH];
	int values[256] = { 0 };
	float pdf[256] = { 0 };
	int WH = 5;
	float TH = 0.0003;
	float v = 0.0;
	int lut[256] = { 0 };

	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int M = height * width;
		int maxs = 1;
		bool greater = true;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				values[src.at<uchar>(i, j)]++;
			}
		}
		for (int i = 0; i < 256; i++) {
			pdf[i] = values[i] * 1.0 / M;
		}
		for (int k = 0 + WH; k < 255 - WH; k++) {
			v = 0.0;
			greater = true;
			for (int i = k - WH; i < k + WH; i++) {
				v += pdf[i];
				if (pdf[k] < pdf[i]) {
					greater = false;
				}
			}
			v /= 2 * WH + 1;
			if ((pdf[k] > v + TH) && greater) {
				lut[maxs++] = k;
			}
		}
		lut[0] = 0;
		lut[maxs] = 255;

		for (int i = 0; i <= maxs; i++) {
			printf("%d ", lut[i]);
		}
		int goodi = 0;
		Mat finalm = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				int mindif = 255;
				for (int k = 0; k <= maxs; k++) {
					if (abs(src.at<uchar>(i, j) - lut[k]) < mindif) {
						mindif = abs(src.at<uchar>(i, j) - lut[k]);
						goodi = k;
					}
				}
				finalm.at<uchar>(i, j) = lut[goodi];
			}
		}
		//showHistogram("Histogram", values, width, height);
		imshow("img", finalm);
		//imshow("img", src);
		waitKey();
	}
}

int d8i[8] = { 0, -1, -1, -1,  0,  1, 1, 1 };
int d8j[8] = { 1,  1,  0, -1, -1, -1, 0, 1 };

Mat convertToCol(Mat img) {
	Mat color(img.rows, img.cols, CV_8UC3);
	Vec3b colVec[600];

	for (int i = 0; i < 600; i++) {			//go through vector and generate random colors
		int R = rand() % 256;				//and remember them in a vector
		int G = rand() % 256;
		int B = rand() % 256;
		colVec[i][0] = R;					
		colVec[i][1] = G;
		colVec[i][2] = B;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int label = img.at<uchar>(i, j);
			if (label == 0) {
				color.at<Vec3b>(i, j) = Vec3b(255, 255, 255);	//point takes the color white because it is the background
			}
			else
				color.at<Vec3b>(i, j) = colVec[label];			//takes color which is at the label position in the color vector 
		}
	}
	return color;												//returns colored vector
}

void BFSlabeling() {
	Mat img = imread("Images/labeling1.bmp", CV_LOAD_IMAGE_GRAYSCALE);			
	int label = 0;

	int height = img.rows;
	int width = img.cols;

	cv::Mat labels = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0 && labels.at<uchar>(i, j) == 0){		//if i encounter an unidentified object i start labeling him
				label++;														//label gets upgraded
				//printf("%d", label);
				queue <Point2i> Q;												//FIFO to store the object's values
				labels.at<uchar>(i, j) = label;									//point stores the value of the label
				Q.push({ i, j });												//store value further extension
				while (!Q.empty()) {											//go through fifo through all the points
					Point2i q = Q.front();										//store value before pop
					Q.pop();													//pop the point because we already used it
					for (int k = 0; k < 8; k++) {								//go through all the neighbors
						if (img.at<uchar>(d8i[k] + q.x, d8j[k] + q.y) == 0 && labels.at<uchar>(d8i[k] + q.x, d8j[k] + q.y) == 0) { //verify that the neighbors are a part of the object
							labels.at<uchar>(d8i[k] + q.x, d8j[k] + q.y) = label;				//store the value of label in the point
							Q.push({ d8i[k] + q.x, d8j[k] + q.y });								//push to find its neighbors in next iteration
							
						}
					}
				}
			}
		}
	}
	imshow("first", img);
	imshow("resolved", convertToCol(labels));
	waitKey(0);
}


void borderTracing() {
	std::vector<Point> border;
	std::vector<int> codes;

	int cRow;
	int cCol;

	Mat img = imread("Images/triangle_up.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat res = Mat::zeros(img.size(), CV_8UC1);

	int height = img.rows;
	int width = img.cols;


	bool found = false;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (img.at<uchar>(i, j) == 0) {			//spot the first black pixel in the image
				found = true;						
				cRow = i;
				cCol = j;
				break;								//we don't care about the others exit first "for"
			}
		}

		if (found == true)							//exit the second "for"
			break;
	}

	int dir = 7;
	border.push_back(Point(cCol, cRow));			//push first pixel
	int i;

	do {
		if (dir % 2 == 0) {
			dir = (dir + 7) % 8;
		}
		else {
			dir = (dir + 6) % 8;
		}

		i = dir;

		do {
			if (img.at<uchar>(cRow + d8i[i], cCol + d8j[i]) == img.at<uchar>(cRow, cCol)) {		//check same pixel color as last pixel
				dir = i;
				cRow = cRow + d8i[i];
				cCol = cCol + d8j[i];
				border.push_back(Point(cCol, cRow));					//push as border
				codes.push_back(dir);
				break;
			}
			i = (i + 1) % 8;
		} while (i != dir);
	} while (!(border[border.size() - 1] == border[1] && border[border.size() - 2] == border[0] && border.size() != 2));	//stop when first two pixels are last two pixels

	
	for (int i = 0; i < border.size() - 1; i++) {
		res.at<uchar>(border[i].y, border[i].x) = 255;
	}
	for (int i = 0; i < codes.size() - 2; i++) {
		std::cout << codes[i] << " ";					//normal codes
	}
	std::cout << "\n\n Derivative Chain Code\n\n";		
	for (int i = 1; i < codes.size() - 1; i++) {		
		int code = codes[i] - codes[i - 1];		//derivative chain code
		code = code < 0 ? code + 8 : code;
		std::cout << code << " ";
	}

	int code = codes[0] - codes[codes.size() - 3];
	code = code < 0 ? code + 8 : code;
	std::cout << code << " ";

	imshow("original", img);
	imshow("contour", res);
	waitKey(0);
}

bool isInside2(Mat img, int i, int j) {
	int width = img.cols;
	int height = img.rows;
	if (i >= height || j >= width || j < 0 || i < 0) {
		return false;
	}
	return true;
}

//dilation alg 8 neighbours
Mat dilation(Mat m1) {
	Mat m2(m1.rows, m1.cols, CV_8UC1);			//create new matrix

	for (int i = 0; i < m1.rows; i++) {					
		for (int j = 0; j < m1.cols; j++) {
			m2.at<uchar>(i, j) = m1.at<uchar>(i, j);	//copy the matrix m1 into m2
		}
	}

	for (int i = 0; i < m1.rows; i++) {					
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uchar>(i, j) == 0) {				//if original matrix has a black pixel then we go through all its neighbours 
				for (int k = 0; k < 8; k++) {			//but we only change the pixels that are on the second matrix to not cause a recursive coloring
					if (isInside2(m2, i + d8i[k], j + d8j[k])) {		//by coloring the new generated black pixels
						m2.at<uchar>(i + d8i[k], j + d8j[k]) = 0;
					}
				}
			}
		}
	}
	return m2;

}

//erosion alg 8 neighbours 
Mat erosion(Mat m1) {
	Mat m2(m1.rows, m1.cols, CV_8UC1);			//create new matrix

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			m2.at<uchar>(i, j) = m1.at<uchar>(i, j);	//copy the matrix m1 into m2
		}
	}

	for (int i = 0; i < m1.rows; i++) {
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uchar>(i, j) == 255) {				//if original matrix has a white pixel then we go through all its neighbours 
				for (int k = 0; k < 8; k++) {			//but we only change the pixels that are on the second matrix to not cause a recursive coloring
					if (isInside2(m2, i + d8i[k], j + d8j[k])) {		//by coloring the new generated white pixels
						m2.at<uchar>(i + d8i[k], j + d8j[k]) = 255;
					}
				}
			}
		}
	}
	return m2;
}

int countBlack(Mat m1) {
	int nr = 0;
	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uchar>(i, j) == 0) {
				nr++;
			}
		}
	return nr;
}

Mat opening(Mat m1) {
	Mat m2 = erosion(m1);		//call first erosion
	Mat m3 = dilation(m2);		//call second dilation on erosion matrix
	return m3;
}

Mat closing(Mat m1) {
	Mat m2 = dilation(m1);		//call first dilation
	Mat m3 = erosion(m2);		//call second erosion on dilation matrix
	
	return m3;
}

Mat openingN(Mat m1, int n) {
	if (n <= 0) return m1;
	Mat m2 = m1;
	for (int i = 0; i < n; i++)	
		m2 = opening(m2);

	return m2;
}

Mat closingN(Mat m1, int n) {
	if (n <= 0) return m1;
	Mat m2 = m1;
	for (int i = 0; i < n; i++)
		m2 = closing(m2);

	return m2;
}

void morphof1() {
	Mat src = imread("Images/Morphological_Op_Images/1_Dilate/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat m1 = dilation(src);
	Mat m2 = erosion(src);

	printf("Black pixels original: %d \n",countBlack(src));
	printf("Black pixels dilation: %d \n", countBlack(m1));
	printf("Black pixels erosion: %d \n", countBlack(m2));


	imshow("Original", src);
	imshow("Dilation", m1);
	imshow("Erosion", m2);
	waitKey(0);
}

void morphof2() {
	Mat src = imread("Images/Morphological_Op_Images/3_Open/cel4thr3_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat m3 = opening(src);

	int n;
	printf("\nEnter n: ");
	scanf("%d", &n);
	Mat m3 = openingN(src, n);

	printf("Black pixels original: %d \n", countBlack(src));
	printf("Black pixels opening: %d \n", countBlack(m3));
	
	imshow("Original", src);
	imshow("Opening N", m3);
	waitKey(0);
}

void morphof3() {
	Mat src = imread("Images/Morphological_Op_Images/4_Close/mon1thr1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat m3 = opening(src);

	int n;
	printf("\nEnter n: ");
	scanf("%d", &n);
	Mat m3 = closingN(src, n);

	printf("Black pixels original: %d \n", countBlack(src));
	printf("Black pixels closing: %d \n", countBlack(m3));

	imshow("Original", src);
	imshow("Closing N", m3);
	waitKey(0);
}

void boundaryExtraction() {
	Mat src = imread("Images/Morphological_Op_Images/5_BoundaryExtraction/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	Mat m2 = erosion(src);		//it first computes the erosion of the matrix
	Mat m3(src.rows, src.cols, CV_8UC1);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			m3.at<uchar>(i, j) = src.at<uchar>(i, j);
			if (src.at<uchar>(i, j) == 0 && m2.at<uchar>(i, j) == 0) {	//every pixel that is black in the original
				m3.at<uchar>(i, j) = 255;								//image AND the second image becomes white
			}
		}
	}
	imshow("Original", src);
	imshow("Boundary", m3);
	waitKey(0);
}

Mat complement(Mat m1) {
	Mat m2(m1.rows, m1.cols, CV_8UC1);
	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++) {//loops through each pixel, if white it becomes black
			if (m1.at<uchar>(i, j) == 0)
				m2.at<uchar>(i, j) = 255;
			else
				m2.at<uchar>(i, j) = 0;			//else it is set as white
		}

	return m2;
}

Mat intersection(Mat m1, Mat m2) {
	Mat m3(m1.rows, m1.cols, CV_8UC1);

	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++)
			m3.at<uchar>(i, j) = 255;


	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uchar>(i, j) == 0 && m2.at<uchar>(i, j) == 0)	//both must be black
				m3.at<uchar>(i, j) = 0;								//becomes black if both are black
			else
				m3.at<uchar>(i, j) = 255;						//if not, becomes white
		}

	return m3;
}

Mat reunion(Mat m1, Mat m2) {
	Mat m3(m1.rows, m1.cols, CV_8UC1);

	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++)
			m3.at<uchar>(i, j) = 255;


	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++) {
			if (m1.at<uchar>(i, j) == 0 || m2.at<uchar>(i, j) == 0)		//check each pixel position if it is black
				m3.at<uchar>(i, j) = 0;									//it becomes black
			else
				m3.at<uchar>(i, j) = 255;								//if both are white, it becomes white
		}

	return m3;
}

void regionFilling() {
	Mat m1 = imread("Images/Morphological_Op_Images/6_RegionFilling/reg1neg1_bw.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int x = m1.rows / 2;			//start from the middle
	int y = m1.cols / 2;			//of the image

	Mat m2 = complement(m1);		//compute the complement 
	Mat m3(m1.rows, m1.cols, CV_8UC1);	//create new matrix like m1

	for (int i = 0; i < m1.rows; i++){
		for (int j = 0; j < m1.cols; j++) {
			m3.at<uchar>(i, j) = 255;		//make the new matrix all white
		}
	}

	m3.at<uchar>(y, x) = 0;						//make starting pixel black, the rest are white
	int nr1 = countBlack(m3);					//it will be 1 at first
	int nr2;

	while (1) {
		Mat m4 = intersection(dilation(m3), m2);	//compute intersection between the dilation of m3 and m2
		nr2 = countBlack(m4);						//count number of black pixels
		m3 = m4;
		if (nr2 == nr1) {							//cpmpare the two images by comparing the number of 
			break;									//black pixels
		}
		nr1 = nr2;
	}
	Mat m5 = reunion(m1, m3);

	imshow("Original", m1);
	imshow("RegionFill", m5);
	waitKey(0);
}

void meanDev() {
	Mat m1 = imread("Images/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int x = m1.rows;
	int y = m1.cols;
	int aux = 0;
	double sum = 0;
	double sum2 = 0;

	for (int i = 0; i < x; i++)						//we apply the formula, by computing the intensity values
		for (int j = 0; j < y; j++)
			sum = sum + m1.at<uchar>(i, j);
	sum = sum / (x*y);								//divide by the area

	for (int i = 0; i < x; i++)
		for (int j = 0; j < y; j++) {
			aux = (m1.at<uchar>(i, j) - sum) * (m1.at<uchar>(i, j) - sum);	//apply formula, 
			sum2 = sum2 + aux;												//add to sum 
		}
	sum2 = sum2 / (x*y);													//divide by area
	sum2 = sqrt(sum2);														

	printf("Mean = %f\n", sum);												//print mean value
	printf("Deviation = %f\n", sum2);										//print deviation
	waitKey();
	system("pause");
}

void bglobalThreshAlg() {
	Mat m2 = imread("Images/eight.bmp");

	int imax = 0;							//max intensity
	int imin = 255;							//min intensity

	double T = 0;							//current threshold value
	double Tk = 0;							//previous threshold value

	double miu1 = 0;						//miu low
	double miu2=0;							//miu high
	int aux1 = 0;
	int aux2 = 0;
	double error = 0.1;						//error

	for(int i = 0; i < m2.rows; i++)
		for (int j = 0; j < m2.cols; j++) {
			if (m2.at<uchar>(i, j) > imax)		//compute imax by searching highest intensity
				imax = m2.at<uchar>(i, j);
			if (m2.at<uchar>(i, j) < imin)		//compute imin by searching lowest intensity
				imin = m2.at<uchar>(i, j);
		}

	T = (imax + imin) / 2;						//average intensity

	while (abs(Tk - T) > error){				//repeat until lower than error
		aux1 = 0;
		aux2 = 0;
		Tk = T;									//remember the previus threshold

		for (int i = 0; i < m2.rows; i++)
			for (int j = 0; j < m2.cols; j++) {
				if (m2.at<uchar>(i, j) <= T) {
					aux1++;
					miu1 = miu1 + m2.at<uchar>(i, j);	//lower threshold
				}
				if (m2.at<uchar>(i, j) > T) {
					aux2++;	
					miu2 = miu2 + m2.at<uchar>(i, j);	//higher threshold
				}
			}

		miu1 = miu1 / (aux1);
		miu2 = miu2 / (aux2);
		T = (miu1 + miu2) / 2;		//current threshold
	}
	imshow("Original", m2);			//show original image
	for (int i = 0; i < m2.rows; i++) {
		for (int j = 0; j < m2.cols+600; j++) {
			if (m2.at<uchar>(i, j) <= T)
				m2.at<uchar>(i, j) = 0;			//lower than threshold then white
			else
				m2.at<uchar>(i, j) = 255;		//higher than threshold then black
		}
	}

	imshow("Basic global thresholding algorithm", m2);	//show image after threshold
	waitKey();
}

void histoTrans() {
	int off;
	Mat m2 = imread("Images/balloons.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	printf("Insert offset: ");
	scanf("%d", &off);

	for (int i = 0; i < m2.rows; i++)
		for (int j = 0; j < m2.cols; j++)
			if (m2.at<uchar>(i, j) + off < 0)					//if overflow then if smaller than 0
				m2.at<uchar>(i, j) = 0;							//turn it 0 and if bigger than 255
			else
				if (m2.at<uchar>(i, j) + off > 255)				//turn it 255 so it won't turn other values
					m2.at<uchar>(i, j) = 255;
				else
					m2.at<uchar>(i, j) = m2.at<uchar>(i, j) + off;
	imshow("After", m2);
	waitKey();
}

void histoStretShri() {
	Mat img = imread("Images/eight.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int gimin = 10;
	int gimax = 250;
	int del = 50;

	int gomax = 200;
	int gomin = 100;
	int h = img.rows;
	int w = img.cols;

	int vect[256];
	float s2 = 0;
	memset(vect, 0, sizeof(vect));
	int histSize;
	int vect1[256];
	memset(vect1, 0, sizeof(vect1));

	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			vect[img.at<uchar>(i, j)]++;
		}
	}

	showHistogram("hist", vect, 255, 200);
	for (int i = 0; i < 256; i++) {
		vect1[i] = gomin + (vect[i] - gimin) * (gomax - gomin) / (gimax - gimin);
	}
	showHistogram("hist1", vect1, 255, 200);
	waitKey(0);
}

void convolution(Mat_<int> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

	output.create(img.size());
	memcpy(output.data, img.data, img.rows * img.cols * sizeof(uchar));

	int scalingCoeff = 1;
	int additionFactor = 0;
	bool ok = 1;
	//TODO: decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
	// low pass if all elements >= 0
	// high pass has elements < 0
	for (int i = 0; i < filter.rows; i++) {
		for (int j = 0; j < filter.cols; j++) {
			if (filter.at<int>(i, j) < 0) {
				ok = 0;
			}
		}
	}
	// compute scaling coefficient and addition factor for low pass and high pass
	// low pass: additionFactor = 0, scalingCoeff = sum of all elements
	// high pass: formula 9.20
	//if ok == 1 the it is a low pass filter, s. coeff = sum of elements
	if (ok == 1) {
		printf("LP filter\n");
		int sum = 0;
		for (int i = 0; i < filter.rows; i++) {
			for (int j = 0; j < filter.cols; j++) {
				sum += filter.at<int>(i, j);
			}
		}
		scalingCoeff = sum;
	}
	else {
		//apply 9.2 formula
		int sumPlus = 0;
		int sumMinus = 0;
		printf("HP filter\n");
		for (int i = 0; i < filter.rows; i++) {
			for (int j = 0; j < filter.cols; j++) {
				if (filter[i][j] >= 0) {
					sumPlus += filter.at<int>(i, j);   //the sum of positive elements 
				}
				else {
					sumMinus -= filter.at<int>(i, j);  //the sum of the magnitude of the negative elements
				}
			}
		}
		if (sumMinus == 0) {
			additionFactor = 0;
			scalingCoeff = sumPlus;
		}
		else {
			additionFactor = 255 / 2;
			scalingCoeff = 2 * max(sumPlus, sumMinus);
		}
	}
	// TODO: implement convolution operation (formula 9.2)
	// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]
	int k = (filter.rows - 1) / 2;
	for (int i = k; i < img.rows - k; i++) {
		for (int j = k; j < img.cols - k; j++) {
			int val = 0;
			for (int u = 0; u < filter.rows; u++) {
				for (int v = 0; v < filter.cols; v++) {
					val += filter.at<int>(u, v) * img.at<uchar>(i + u - k, j + v - k);
				}
			}
			val /= scalingCoeff;
			val += additionFactor;
			output.at<uchar>(i, j) = val;
		}
	}


}

void logOfMagnitude(Mat_<float> img, Mat_<float> mag) {
	int max = 0;				//lowest value max
	int min = 255;				//highest value min
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			img(i, j) = log(1 + mag.at<float>(i, j));
			if (img(i, j) < min) {	//finding minimum
				min = img(i, j);
			}
			if (img(i, j) > max) {	//finding maximum
				max = img(i, j);
			}
		}
	}
	for (int i = 0; i < mag.rows; i++) {
		for (int j = 0; j < mag.cols; j++) {
			img(i, j) = (img(i, j) - min) / (max - min);	//applying formula
		}
	}
}

void Lab9() {


	// PART 1: convolution in the spatial domain
	Mat_<uchar> img = imread("cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat_<uchar> outputImage;

	// LOW PASS
	// mean filter 5x5
	int meanFilterData5x5[25];
	fill_n(meanFilterData5x5, 25, 1);
	Mat_<int> meanFilter5x5(5, 5, meanFilterData5x5);

	// mean filter 3x3
	Mat_<int> meanFilter3x3(3, 3, meanFilterData5x5);

	// gaussian filter
	int gaussianFilterData[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	Mat_<int> gaussianFilter(3, 3, gaussianFilterData);

	// HIGH PASS
	// laplace filter 3x3
	int laplaceFilterData[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
	Mat_<int> laplaceFilter(3, 3, laplaceFilterData);

	int highpassFilterData[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	Mat_<int> highpassFilter(3, 3, highpassFilterData);

	//TODO: convolution with the mean filter 5 x 5
	convolution(meanFilter5x5, img, outputImage);
	//TODO: convolution with the mean filter 3 x 3
	//TODO: convolution with the gaussian filter
	//TODO: convolution with the laplacian filter
	//TODO: convolution with the highpass filter


	// PART 2: convolution in the frequency domain
	// use the generic_frequency_domain_filter() function


	// TODO: convolution with the ideal low pass filter (formula 9.16) take R = 20 - set the real and imaginary part to 0 for locations that are located at a distance larger than R from the image center 
	// TODO: convolution with the ideal high pass filter (formula 9.17) take R = 20 - set the real and imaginary part to 0 for locations that are located at a distance smaller than R from the image center 
	// TODO: convolution with the Gaussian low pass filter (formula 9.18) take A = 10 - multiply both the the real and imaginary part with e to the power as in formula 9.18
	// TODO: convolution with the Gaussian high pass filter (formula 9.19) take A = 10 -  multiply both the the real and imaginary part with e to the power as in formula 9.19

}

/*  in the frequency domain, the process of convolution simplifies to multiplication => faster than in the spatial domain
	the output is simply given by F(u,v)ÄÂG(u,v) where F(u,v) and G(u,v) are the Fourier transforms of their respective functions
	The frequency-domain representation of a signal carries information about the signal's magnitude and phase at each frequency*/

	/*
	The algorithm for filtering in the frequency domain is:
		a) Perform the image centering transform on the original image (9.15)
		b) Perform the DFT transform
		c) Alter the Fourier coefficients according to the required filtering
		d) Perform the IDFT transform
		e) Perform the image centering transform again (this undoes the first centering transform)
	 */

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src)
{

	// Discrete Fourier Transform: https://docs.opencv.org/4.2.0/d8/d01/tutorial_discrete_fourier_transform.html
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	// Centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	// the frequency is represented by its real and imaginary parts called frequency coefficients
	// split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

	//calculate magnitude and phase of the frequency by transforming it from cartesian to polar coordinates
	// the magnitude is useful for visualization

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga6d3b097586bca4409873d64a90fe64c3
	phase(channels[0], channels[1], phi); // https://docs.opencv.org/master/d2/de8/group__core__array.html#ga9db9ca9b4d81c3bde5677b8f64dc0137


	// TODO: Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
	// do not forget to normalize

	// TODO: Insert filtering operations here ( channels[0] = Re(DFT(I), channels[1] = Im(DFT(I) )


	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

	// Inverse Centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

void spNoiseMedian()
{
	Mat src = imread("Images/balloons_Salt&Pepper.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int w = 5;

	int height = src.rows;
	int width = src.cols;

	Mat dst = Mat(height - w / 2 * 2, width - w / 2 * 2, CV_8UC1);	//compute destination matrix
	for (int i = w / 2; i < height - w / 2; i++)					//start from center of window so we need to adjust starting point in the image
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			std::vector <uchar> vals;							//here we save vals
			for (int y = 0; y < w; y++)
			{
				for (int x = 0; x < w; x++)
				{
					vals.push_back(src.at<uchar>(i + y - w / 2, j + x - w / 2));	//save elements in vector
				}
			}
			std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end()); //take the middle element
			dst.at<uchar>(i - w / 2, j - w / 2) = vals.at(vals.size() / 2);				//save it in the image 
		}
	}

	imshow("Gaussian", dst);
	imshow("Initial", src);
	waitKey();
}

void gaussian2DNoise(int w)
{
	Mat src = imread("Images/balloons_Gauss.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	int width = src.cols;
	int height = src.rows;

	Mat gMat = Mat(w, w, CV_32F);
	Mat dst = Mat(height - w / 2 * 2, width - w / 2 * 2, CV_8UC1);	//make destination matrix

	float sigm = w * 1.0f / 6;						//compute sigma
	int med = w / 2;								//central 
	for (int i = 0; i < w; i++)
		for (int j = 0; j < w; j++)					//using the equation to construct the elements
		{
			gMat.at<float>(i, j) = 1 / (2 * CV_PI * sigm * sigm) * exp(-((i - med) * (i - med) + (j - med) * (j - med)) / (2 * sigm * sigm));
			printf("%f", gMat.at<float>(i, j));
		}

	for (int i = w / 2; i < height - w / 2; i++)
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0;
			for (int y = 0; y < w; y++)
			{
				for (int x = 0; x < w; x++)
				{
					sum += src.at<uchar>(i + y - w / 2, j + x - w / 2) * gMat.at<float>(y, x);	//computing sum
				}
			}
			dst.at<uchar>(i - w / 2, j - w / 2) = sum;		//destination takes final sum
		}

	imshow("Gaussian", dst);
	imshow("Initial", src);
	waitKey();
}

void gaussian1DNoise(int w)
{
	Mat src = imread("Images/portrait_Gauss2.bmp", CV_LOAD_IMAGE_GRAYSCALE);

	int width = src.cols;
	int height = src.rows;
	Mat dst = Mat(height - w / 2 * 2, width - w / 2 * 2, CV_8UC1);		//create destination matrix

	float* gx = (float*)malloc(w * sizeof(float));						
	float* sums = (float*)malloc(w * sizeof(float));

	float sigm = w * 1.0f / 6;							
	int med = w / 2;
	for (int i = 0; i < w; i++)
	{
		gx[i] = 1 / sqrt((2 * CV_PI * sigm * sigm)) * exp(-(i - med) * (i - med) / (2 * sigm * sigm));	//apply formula
	}

	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			float sum = 0;
			memset(sums, 0, w * sizeof(float));
			for (int y = 0; y < w; y++)
			{
				for (int x = 0; x < w; x++)
				{
					sums[y] += src.at<uchar>(i + x - w / 2, j + y - w / 2) * gx[x];
				}
			}
			for (int x = 0; x < w; x++)
			{
				sum += sums[x] * gx[x];
			}
			dst.at<uchar>(i - w / 2, j - w / 2) = sum;
		}
	}

	imshow("Gaussian", dst);
	imshow("Initial", src);
	waitKey();
}

void sobelConvY() {

	//int matX[3][3] = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	int matY[3][3] = { {1,2,1},{0,0,0}, {-1,-2,-1} };
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float suum = 0;
	float fx = 0;
	float fy = 0;

	Mat destX = img.clone();
	Mat destY = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			destX.at<uchar>(i, j) = 0;
			destY.at<uchar>(i, j) = 0;
		}
	}

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			float s = 0;
			float s2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {
					s2 += (float)matY[u][v] * img.at<uchar>(i + u - 1, j + v - 1);
				}
			}

			if (s2 > 255) {
				s2 = 255;
			}
			else if (s2 < 0) {
				s2 = 0;
			}
			destY.at<uchar>(i, j) = s2;
		}
	}

	imshow("destY", destY);
	imshow("source", img);
	waitKey(0);
	//return destY;
}

void sobelConvX() {

	int matX[3][3] = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	int matY[3][3] = { {1,2,1},{0,0,0}, {-1,-2,-1} };
	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float suum = 0;
	float fx = 0;
	float fy = 0;

	Mat destX = img.clone();
	Mat destY = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			destX.at<uchar>(i, j) = 0;
			destY.at<uchar>(i, j) = 0;
		}
	}

	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			float s = 0;
			float s2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {

					s += (float)matX[u][v] * img.at<uchar>(i + u - 1, j + v - 1);
					s2 += (float)matY[u][v] * img.at<uchar>(i + u - 1, j + v - 1);

				}
			}
			if (s > 255) {
				s = 255;
			}
			else if (s < 0) {
				s = 0;
			}
			if (s2 > 255) {
				s2 = 255;
			}
			else if (s2 < 0) {
				s2 = 0;
			}
			destX.at<uchar>(i, j) = s;
			destY.at<uchar>(i, j) = s2;
		}
	}

	imshow("destX", destX);
	imshow("source", img);
	waitKey(0);
	//return destX;
}

void magnitudeSob() {

	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float suum = 0;

	float fx = 0;
	float fy = 0;

	Mat destX = img.clone();
	Mat destY = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			destX.at<uchar>(i, j) = 0;
			destY.at<uchar>(i, j) = 0;
		}
	}
	Mat mag(img.rows, img.cols, CV_8UC1);
	Mat dir(img.rows, img.cols, CV_32F);


	int matX[3][3] = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	int matY[3][3] = { {1,2,1},{0,0,0}, {-1,-2,-1} };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			mag.at<uchar>(i, j) = 0;
			dir.at<float>(i, j) = 0;
		}
	}
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			float s = 0;
			float s2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {

					s += (float)matX[u][v] * img.at<uchar>(i + u - 1, j + v - 1);
					s2 += (float)matY[u][v] * img.at<uchar>(i + u - 1, j + v - 1);



				}
			}


			mag.at<uchar>(i, j) = (sqrt(pow(s, 2) + pow(s2, 2))) / (4 * sqrt(2));
			dir.at<float>(i, j) = std::atan2(s2, s);
		}
	}

	imshow("magnitude", mag);
	waitKey(0);
}

void dirSob() {

	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float suum = 0;

	float fx = 0;
	float fy = 0;

	Mat destX = img.clone();
	Mat destY = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			destX.at<uchar>(i, j) = 0;
			destY.at<uchar>(i, j) = 0;
		}
	}
	Mat dir(img.rows, img.cols, CV_32F);


	int matX[3][3] = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	int matY[3][3] = { {1,2,1},{0,0,0}, {-1,-2,-1} };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			dir.at<float>(i, j) = 0;
		}
	}
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			float s = 0;
			float s2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {

					s += (float)matX[u][v] * img.at<uchar>(i + u - 1, j + v - 1);
					s2 += (float)matY[u][v] * img.at<uchar>(i + u - 1, j + v - 1);



				}
			}
			dir.at<float>(i, j) = std::atan2(s2, s);
		}
	}
	imshow("dir", dir);
	waitKey(0);
}

void magThreshSob() {

	Mat img = imread("Images/cameraman.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	float suum = 0;

	float fx = 0;
	float fy = 0;

	Mat destX = img.clone();
	Mat destY = img.clone();

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			destX.at<uchar>(i, j) = 0;
			destY.at<uchar>(i, j) = 0;
		}
	}
	Mat mag(img.rows, img.cols, CV_8UC1);
	Mat dir(img.rows, img.cols, CV_32F);


	int matX[3][3] = { {-1,0,1} , {-2,0,2} , {-1,0,1} };
	int matY[3][3] = { {1,2,1},{0,0,0}, {-1,-2,-1} };

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			mag.at<uchar>(i, j) = 0;
			dir.at<float>(i, j) = 0;
		}
	}
	for (int i = 1; i < img.rows - 1; i++) {
		for (int j = 1; j < img.cols - 1; j++) {
			float s = 0;
			float s2 = 0;
			for (int u = 0; u < 3; u++) {
				for (int v = 0; v < 3; v++) {

					s += (float)matX[u][v] * img.at<uchar>(i + u - 1, j + v - 1);
					s2 += (float)matY[u][v] * img.at<uchar>(i + u - 1, j + v - 1);



				}
			}
			int magg = (sqrt(pow(s, 2) + pow(s2, 2))) / (4 * sqrt(2));
			if (magg < 50) {
				mag.at<uchar>(i, j) = 0;

			}
			else mag.at<uchar>(i, j) = 255;
			dir.at<float>(i, j) = std::atan2(s2, s);
		}
	}

	imshow("magnitude", mag);
	waitKey(0);
}

void tryCanny()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src, dst, gauss;
		src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int)k * pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss, dst, pL, pH, 3);
		imshow("input image", src);
		imshow("canny", dst);
		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Negative image\n");
		printf(" 11 - Additive gray\n");
		printf(" 12 - Multiplicative gray\n");
		printf(" 13 - Squares\n");
		printf(" 14 - Squares\n");
		printf(" 15 - 3 channels\n");
		printf(" 16 - Gray\n");
		printf(" 17 - Binary\n");
		printf(" 18 - HSV\n");
		printf(" 19 - Is inside\n");
		printf(" 20 - vecHist\n");
		printf(" 21 - multiThresh\n");
		printf(" 22 - testClick\n");
		printf(" 23 - BFS labeling\n");
		printf(" 24 - Border tracing Algorithm\n");
		printf(" 25 - Test dilution/erosion operations\n");
		printf(" 26 - Test opening operation\n");
		printf(" 27 - Test closing operation\n");
		printf(" 28 - Boundary extraction\n");
		printf(" 29 - Region filling\n");
		printf(" 30 - Mean value and standard deviation\n");
		printf(" 31 - Basic global thresholding algorithm\n");
		printf(" 32 - Analytical histogram transformation functions\n");
		printf(" 33 - Stretch shrink histogram\n");
		printf(" 34 - Convolution\n");
		printf(" 35 - Salt & pepper noise removal with median filter\n");
		printf(" 36 - Gaussian 2D\n");
		printf(" 37 - Gaussian 1D\n");
		printf(" 38 - Vertical Conv\n");
		printf(" 39 - Horizontal Conv\n");
		printf(" 40 - Gradient Magnitude\n");
		printf(" 41 - Direction\n");
		printf(" 42 - Threshold arbitrary+fixed\n");
		printf(" 43 - Canny edge detection algorithm\n");
		printf(" 0 - Exit\n\n");

		printf("Option: ");
		scanf("%d", &op);
		switch (op)
		{
		case 1:
			testOpenImage();
			break;
		case 2:
			testOpenImagesFld();
			break;
		case 3:
			testParcurgereSimplaDiblookStyle(); //diblook style
			break;
		case 4:
			//testColor2Gray();
			testBGR2HSV();
			break;
		case 5:
			testResize();
			break;
		case 6:
			testCanny();
			break;
		case 7:
			testVideoSequence();
			break;
		case 8:
			testSnap();
			break;
		case 9:
			testMouseClick();
			break;
		case 10:
			negative_image();
			break;
		case 11:
			scanf("%d", &factor);
			additive_gray(factor);
			break;
		case 12:
			scanf("%d", &factor);
			multiplicative_gray(factor);
			break;
		case 13:
			create_squares();
			break;
		case 14:
			inverse_matrix();
			break;
		case 15:
			ex1();
			break;
		case 16:
			ex2();
			break;
		case 17:
			gray_to_binary();
			break;
		case 18:
			hsv_values();
			break;
		case 19:
			printf("%d", isInside());
			break;
		case 20:
			vecHist();
			break;
		case 21:
			//scanf("%d", &factor);
			multi_level_thresh();
			break;
		case 22:
			//scanf("%d", &factor);
			testMouseClick();
			break;
		case 23:
			BFSlabeling();
			break;
		case 24:
			borderTracing();
			break;
		case 25:
			morphof1();
			break;
		case 26:
			morphof2();
			break;
		case 27:
			morphof3();
			break;
		case 28:
			boundaryExtraction();
			break;
		case 29:
			regionFilling();
			break;
		case 30:
			meanDev();
			break;
		case 31:
			bglobalThreshAlg();
			break;
		case 32:
			histoTrans();
			break;
		case 33:
			histoStretShri();
			break;
		case 34:
			Lab9();
			break;
		case 35:
			spNoiseMedian();
			break;
		case 36:
			gaussian2DNoise(5);
			break;
		case 37:
			gaussian1DNoise(3);
			break;
		case 38:
			sobelConvY();
			break;
		case 39:
			sobelConvX();
			break;
		case 40:
			magnitudeSob();
			break;
		case 41:
			dirSob();
			break;
		case 42:
			magThreshSob();
			break;
		case 43:
			tryCanny();
			break;
		}
	} while (op != 0);
	return 0;
}