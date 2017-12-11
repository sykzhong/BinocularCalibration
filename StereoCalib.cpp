//
// Created by syk on 17-10-23.
//

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "tinydir.h"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>

using namespace std;
using namespace cv;

static void StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true)
{
    if(imagelist.size()%2 != 0)
    {
        cout << "Error: the image list contains odd(non-even) number of elements\n";
        return;
    }
    bool displayCorners = true;
    const int maxScale = 2;
    const float squareSize = 1.f;   //Set this to your actual square size

    vector<vector<Point2f> > imagePoints[2];
    vector<vector<Point3f> > objectPoints;
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;

    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    vector<string> goodImageList;

//    sykfix: Get the current path, but is redundancy
    char current_absolute_path[200];
    if (NULL == getcwd(current_absolute_path, 200))
    {
        printf("***Error***");
        exit(-1);
    }


    for(i = j = 0; i < nimages; i++)
    {
        for(k = 0; k < 2; k++)
        {
            //sykdebug: k is the flag to tell whether success, k == 2
            const string& filename = string(current_absolute_path)+imagelist[i*2 + k];
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if(imageSize == Size())
                imageSize = img.size();
            else if(img.size() != imageSize)
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for(int scale = 1; scale <= maxScale; scale++)
            {
                Mat timg;
                if(scale == 1)
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
                found = findChessboardCorners(timg, boardSize, corners,
                    CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
                if(found)
                {
                    if(scale > 1)       //sykdebug: for debug? add imshow?
                    {
                        Mat cornerMat(corners);
                        cornerMat *= 1./scale;
                    }
                    break;
                }
            }
            if(!found)
                break;
            cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),
                         TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 30, 0.01));
            if(displayCorners)
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(cimg, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if(c == 27 || c == 'q' || c == 'Q')
                    exit(-1);
            }
            else
                putchar('.');

        }
        if(k == 2)
        {
            goodImageList.push_back(imagelist[i*2]);
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if(nimages < 2)
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }
    imagePoints[0].resize(nimages);
    imagePoints[1].resize(nimages);
    objectPoints.resize(nimages);
    for(i = 0; i < nimages; i++)
    {
        for(j = 0; j < boardSize.height; j++)
            for(k = 0; k < boardSize.width; k++)
                objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";
    Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        imageSize, R, T, E, F,
        CV_CALIB_FIX_ASPECT_RATIO +
        CV_CALIB_ZERO_TANGENT_DIST +
        CV_CALIB_SAME_FOCAL_LENGTH +
        CV_CALIB_RATIONAL_MODEL +
        CV_CALIB_FIX_K3 + CV_CALIB_FIX_K4 + CV_CALIB_FIX_K5,
        TermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 100, 1e-5)
    );
    cout << "done with RMS error=" << rms << endl;

    /*
     * CALIBRATION QUALITY CHECK
     * because the output fundermental matrix implicitly
     * includes all the output information,
     * we can check the quality of calibration using the
     * epipolar geometry constraint: m2^t*F*m1=0
     * */
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for(i = 0; i < nimages; i++)
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for(k = 0; k < 2; k++)
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            // undistort the corner point
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for(j = 0; j < npt; j++)
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                imagePoints[0][i][j].y*lines[1][j][1] +lines[1][j][2]) +
                fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " << err / npoints << endl;

    //save intrinsic parameters
    FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
    if (fs.isOpened())
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
           "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";
    // OpenCv can handle left-right
    // or up-down camer arrangements

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]
    );

    fs.open("extrinsics.yml", CV_STORAGE_WRITE);
    if (fs.isOpened())
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

    if (!showRectified)
        return;
    Mat rmap[2][2];
    // if by calibrated(bouguet's method)
    if (useCalibrated)
    {
        // already computed everything
    }
        // or else hartley's method
    else
        // use intrinsic parameters of each camera, but
        // compute the rectification transformation directly
        // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for (k = 0; k < 2; k++)
        {
            for (i = 0; i < nimages; i++)
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);
        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }
    // Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);
    Mat canvas;
    double sf;
    int w, h;
    if (!isVerticalStereo)
    {
        sf = 600. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w * 2, CV_8UC3);
    }
    else
    {
        sf = 300. / MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h * 2, w, CV_8UC3);
    }
    for (i = 0; i < nimages; i++)
    {
        for (k = 0; k < 2; k++)
        {
            Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
            imshow("单目相机校正结果", rimg);
            waitKey();
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
            if (useCalibrated)
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[j].height*sf));
                rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
            }
        }
        if (!isVerticalStereo)
            for (j = 0; j < canvas.rows; j += 16)
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for (j = 0; j < canvas.cols; j += 16)
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("双目校正结果", canvas);
        waitKey();
        char c = (char)waitKey();
        if (c == 27 || c == 'q' || c == 'Q')
            break;
    }
}

static bool readStringList(const string& filename, vector<string>& l)
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if(n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}

//This function only works in clion, and should define the work dirctory in clion first
int getImagelist(vector<string> &imagelist, string Prefix)
{
    imagelist.clear();
    //Get current path
//    char *current_absolute_path = "/home/syk/CLionProjects/BinocularCalibration";
    char current_absolute_path[200];
    if (NULL == getcwd(current_absolute_path, 200))
    {
        printf("***Error***");
        exit(-1);
    }

    tinydir_dir dir;
    size_t i;
    if (tinydir_open_sorted(&dir, TINYDIR_STRING(current_absolute_path)) == -1)
    {
        perror("Error opening file");
        tinydir_close(&dir);
    }
    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        if (tinydir_readfile_n(&dir, &file, i) == -1)
        {
            perror("Error getting file");
            return 0;
        }
        if (file.is_dir && file.name == Prefix)
        {
            if (tinydir_open_subdir_n(&dir, i) == -1)
            {
                perror("Error opening subdirectory");
                return 0;
            }
            break;
        }
    }
    for (i = 0; i < dir.n_files; i++)
    {
        tinydir_file file;
        if (tinydir_readfile_n(&dir, &file, i) == -1)
        {
            perror("Error getting file");
            return 0;
        }
        if(string(file.name) != "." && string(file.name) != "..")
            imagelist.push_back(string(current_absolute_path) + "/" + Prefix + "/" + (file.name));
    }
    return 1;
}

#define TEST2

#ifdef TEST1

int main(int argc, char** argv)
{
    Size boardSize;
    string imagelistfn;
    bool showRectified = true;
    for(int i = 1; i < argc; i++)
    {
        if(string(argv[i]) == "-w")
        {
            if(sscanf(argv[++i], "%d", &boardSize.width) != 1 || boardSize.width <= 0)
            {
                cout << "invalid board width" << endl;
                return -1;
            }
        }
        else if(string(argv[i]) == "-h")
        {
            if(sscanf(argv[++i], "%d", &boardSize.height) != 1 || boardSize.height <= 0)
            {
                cout << "invalid board height" << endl;
                return -1;
            }
        }
        else if(string(argv[i]) == "-nr")
            showRectified = false;
        else if(string(argv[i]) == "--help")
            return -1;
        else if(argv[i][0] = '-')
        {
            cout << "invalid option " << argv[i] << endl;
            return 0;
        }
        else
            imagelistfn = argv[i];
    }
    if(imagelistfn == "")
    {
        imagelistfn = "stereo_calib.xml";
        boardSize = Size(9, 6);
    }
    else if(boardSize.width <= 0 || boardSize.height <= 0)
    {
        cout << "if you specified XML file with chessboards, you should also specify the board width and height" << endl;
        return 0;
    }
    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
        return -1;
    }
    StereoCalib(imagelist, boardSize, true, showRectified);
    return 0;
}

#elif defined TEST2
int main()
{
    Size boardSize = Size(9, 6);
    bool showRectified = true;
    string leftprefix = "leftimages", rightprefix = "rightimages";
    if(boardSize.width <= 0 || boardSize.height <= 0)
    {
        cout << "if you specified XML file with chessboards, you should also specify the board width and height" << endl;
        return 0;
    }
    vector<string> leftimagelist;
    vector<string> rightimagelist;
    vector<string> imagelist;
    getImagelist(leftimagelist, leftprefix);
    getImagelist(rightimagelist, rightprefix);
//    mixing two imagelist into one imagelist :<left, right, left, right...>
    if(leftimagelist.size() != rightimagelist.size())
    {
        cerr << "left images number don't equal to right images number" << endl;
        return 0;
    }
    else
    {
        for(int i = 0; i < leftimagelist.size(); i++)
        {
            imagelist.push_back(leftimagelist[i]);
            imagelist.push_back(rightimagelist[i]);
        }

    }
    StereoCalib(imagelist, boardSize, true, showRectified);
    return 0;
}

#endif
