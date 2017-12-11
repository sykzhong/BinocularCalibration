//
// Created by syk on 17-12-11.
//

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv_contrib.h"

using namespace cv;

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
                continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

int main(int argc, char** argv) {
//    const char *algorithm_opt = "--algorithm=";
//    const char *maxdisp_opt = "--max-disparity=";
//    const char *blocksize_opt = "--blocksize=";
//    const char *nodisplay_opt = "--no-display";
//    const char *scale_opt = "--scale=";
//
//    if (argc < 3) {
//        return 0;
//    }
    const char *img1_filename = "left.jpg";
    const char *img2_filename = "right.jpg";
    const char *intrinsic_filename = 0;
    const char *extrinsic_filename = 0;
    const char *disparity_filename = 0;
    const char *point_cloud_filename = 0;

    enum {
        STEREO_BM = 0,
        STEREO_SGBM = 1,
        STEREO_HH = 2,
        STEREO_VAR = 3
    };
    int alg = STEREO_SGBM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

//    StereoBM bm;
//    StereoSGBM sgbm;
//    StereoVar var;

    Ptr<StereoBM> bm = StereoBM::create();
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, numberOfDisparities, SADWindowSize);

    img1_filename = "left.jpg";
    img2_filename = "right.jpg";

    int color_mode = alg == STEREO_BM ? 0 : -1;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    Size img_size = img1.size();
    Rect roi1, roi2;
    Mat Q;
//    sykdebug: ?
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

//    bm.state->roi1 = roi1;
//    bm.state->roi2 = roi2;
//    bm.state->preFilterCap = 31;
//    bm.state->SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 9;
//    bm.state->minDisparity = 0;
//    bm.state->numberOfDisparities = numberOfDisparities;
//    bm.state->textureThreshold = 10;
//    bm.state->uniquenessRatio = 15;
//    bm.state->speckleWindowSize = 100;
//    bm.state->speckleRange = 32;
//    bm.state->disp12MaxDiff = 1;

    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    SADWindowSize > 0 ? bm->setBlockSize(SADWindowSize) : bm->setBlockSize(9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    SADWindowSize > 0 ? sgbm->setBlockSize(SADWindowSize) : sgbm->setBlockSize(9);

    int cn = img1.channels();

    sgbm->setP1(8 * cn*sgbm->getBlockSize()*sgbm->getBlockSize());
    sgbm->setP2(32 * cn*sgbm->getBlockSize()*sgbm->getBlockSize());
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(bm->getSpeckleWindowSize());
    sgbm->setSpeckleRange(bm->getSpeckleRange());
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(alg == STEREO_HH);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if (alg == STEREO_BM)
        bm->compute(img1, img2, disp);
    else if (alg == STEREO_SGBM || alg == STEREO_HH)
        sgbm->compute(img1, img2, disp);//------

    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());
    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if (alg != STEREO_VAR)
        disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);

    if (!no_display)
    {
        namedWindow("left", 1);
        imshow("left", img1);

        namedWindow("right", 1);
        imshow("right", img2);

        namedWindow("disparity", 0);
        imshow("disparity", disp8);

        imwrite("result.bmp", disp8);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }

    return 0;
}