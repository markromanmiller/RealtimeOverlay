#include <jni.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <vector>
#include <time.h>
#include <sys/time.h>

using namespace std;
using namespace cv;

extern "C" {

const double ransac_thresh = 2.5f; // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio

Mat mRgbPrev;
bool prevCreated = false;

/* Feature Extraction */
Ptr<Feature2D> detector;
Ptr<DescriptorMatcher> matcher;
Ptr<ORB> orb;
vector<Point2f> object_bb (4);
Mat ref_descriptors;
vector<KeyPoint> ref_keypoints;

/* Optical Flow */
Mat gray, prevGray, image, frame;
vector<Point2f> baselinePoints;
vector<Point2f> framePoints;
TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
Size subPixWinSize(10,10), winSize(31,31);
const int MAX_COUNT = 200;
bool needToInit = true;
double flength;
Mat keypoints3d;
Mat baseOrientation(3, 3, CV_64F);
double downsample_factor = 0.25;

/* Profiling Techniques */

Scalar colors[10];

/* Timing functions*/

long long getTimeNsec()
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec * 1000000LL + (now.tv_nsec / 1000);
}

long long currentTimeInMicroseconds()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((tv.tv_sec * 1000000LL) + tv.tv_usec);
}

long long ticTime;

void tic() {
    ticTime = getTimeNsec();
}
long long toc() {
    long long tocTime = getTimeNsec();
    return tocTime - ticTime;
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_Initialize(JNIEnv *, jobject)
{
    // feature homography
    detector = BRISK::create();
    matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // make colors
    colors[0] = Scalar(0, 127, 0); // green
    colors[1] = Scalar(255, 0, 0); // red
    colors[2] = Scalar(0, 0, 255); // blue
    colors[3] = Scalar(255, 255, 0); // yellow
    colors[4] = Scalar(255, 127, 0); // orange
    colors[5] = Scalar(127, 0, 127); // purple
    colors[6] = Scalar(255, 255, 255); // white
    colors[7] = Scalar(0, 0, 0); // black
    colors[8] = Scalar(0, 255, 0); // bright green
    colors[9] = Scalar(255, 0, 255); // pink
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_FramewiseImgDiff(JNIEnv *env, jobject obj, jlong addrGray, jlong addrRgba)
{
    Mat& mGr  = *(Mat*)addrGray;
    Mat& mRgb = *(Mat*)addrRgba;
    vector<KeyPoint> v;

    /*Ptr<FeatureDetector> detector = FastFeatureDetector::create(50);
    detector->detect(mGr, v);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
     const KeyPoint& kp = v[i];
     circle(mRgb, Point(kp.pt.x, kp.pt.y), 10, Scalar(255,0,0,255));
    }*/
    Mat mLol;
    if (prevCreated)
    {
        absdiff(mRgb, mRgbPrev, mLol);
    }
    else
    {
        mLol = mRgb;
    }
    mRgbPrev = Mat(mRgb);
    prevCreated = true;
    mRgb = mLol;
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_LKHomography
    (JNIEnv * env, jobject, jlong addrGray, jlong addrRgba, jboolean sharpMotion,
        jfloatArray orientationArray, jboolean displayDebug)
{
    if (!prevCreated)
    {
        return;
    }
    tic();
    // -1; Load up the timing process
    int size = 7;
    long long times[size];
    times[0] = 0; // always start at zero

    // 0: Access as C++ arrays
    Mat full_gray  = *(Mat*)addrGray;
    Mat full_image = *(Mat*)addrRgba;

    Mat gray, image;
    resize(full_gray, gray, Size(), downsample_factor, downsample_factor);
    resize(full_image, image, Size(), downsample_factor, downsample_factor);

    times[1] = toc();

    // 1: Grab orientation array
    Mat orientation(3, 3, CV_64F);
    jfloat* oa = env->GetFloatArrayElements(orientationArray, 0);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            orientation.at<double>(i, j) = oa[i*3 + j];
        }
    }
    times[2] = toc();

    Mat wc2px = (Mat_<double>(3, 3) << flength, 0, image.cols/2, 0, flength, image.rows/2, 0, 0, 1);
    Mat temp_gray, grayrgb;

    Mat hmat = wc2px * orientation.t() * baseOrientation * wc2px.inv();
    //cvtColor(image, temp_gray, CV_RGBA2GRAY);
    warpPerspective(prevGray, temp_gray, hmat, Size(temp_gray.cols, temp_gray.rows));
    //cvtColor(temp_gray, grayrgb, CV_GRAY2RGBA);
    //grayrgb.copyTo(image);

    Mat transformed_keypoints = orientation.t() * keypoints3d;

    double height = (double) gray.rows;
    double width = (double) gray.cols;

    framePoints.resize(transformed_keypoints.cols);
    for (int i = 0; i < transformed_keypoints.cols; i++)
    {
        double w = transformed_keypoints.at<double>(2, i);
        framePoints[i].x = transformed_keypoints.at<double>(0, i) * flength / w + width / 2;
        framePoints[i].y = transformed_keypoints.at<double>(1, i) * flength / w + height / 2;
    }

    vector<Point2f> filtered_baseline_points, filtered_frame_points, point_guesses;
    if (true/*!sharpMotion*/) {
        // filter all points that we don't expect to see


        Rect r = Rect(0, 0, gray.cols, gray.rows);
        for (int i = 0; i < framePoints.size(); i++)
        {
            if (r.contains(framePoints[i]))
            {
                filtered_baseline_points.push_back(baselinePoints[i]);
                filtered_frame_points.push_back(framePoints[i]);
            }
        }
        point_guesses = filtered_frame_points;

        times[3] = toc();

        // 2:
        vector<uchar> status;
        vector<float> err;
        calcOpticalFlowPyrLK(temp_gray, gray, point_guesses, filtered_frame_points, status, err, winSize,
                             5, termcrit, 0, 0.001);

        Mat grayrgba;
        Mat overlaid = temp_gray + gray;
        cvtColor(overlaid, grayrgba, CV_GRAY2RGBA);
        grayrgba.copyTo(image);

        times[4] = toc();
    }
    else
    {
        times[3] = toc();
        filtered_baseline_points = baselinePoints;
        filtered_frame_points = framePoints;
        point_guesses = filtered_frame_points;
        times[4] = toc();
    }

    Mat inlier_mask, homography;
    homography = findHomography(filtered_baseline_points, filtered_frame_points,
                                    RANSAC, ransac_thresh, inlier_mask);
    times[5] = toc();
    if(!homography.empty()) {

        // 4: Draw bounding box
        vector<Point2f> new_bb, scaled_object_bb(4);
        for (int i = 0; i < object_bb.size(); i++) {scaled_object_bb[i] = object_bb[i] * downsample_factor;}
        perspectiveTransform(scaled_object_bb, new_bb, homography);
        for (int i = 0; i < new_bb.size(); i++) {new_bb[i] = new_bb[i] / downsample_factor;}

        line( full_image, new_bb[0], new_bb[1], Scalar( 0, 0, 255), 4 );
        line( full_image, new_bb[1], new_bb[2], Scalar( 0, 0, 255), 4 );
        line( full_image, new_bb[2], new_bb[3], Scalar( 0, 0, 255), 4 );
        line( full_image, new_bb[3], new_bb[0], Scalar( 0, 0, 255), 4 );
    }

    if (displayDebug) {
        for( int i = 0; i < filtered_frame_points.size(); i++ )
        {
            circle( full_image, filtered_frame_points[i] / downsample_factor, 3, Scalar(0,255,0), -1, 8);
            circle( full_image, point_guesses[i] / downsample_factor, 3, Scalar(255, 0, 0), -1, 8);
        }

        // 5: Compute all timings
        //(*env)->SetIntArrayRegion(env, result, 0, size, fill);
        times[6] = toc();

        // 5: Draw timings
        for (int i = 0; i < size-1; i++)
        {
            ellipse(full_image, Point(110, 110), Size (100, 100), 0, // mat, center, axes, ellipse_angle
                    (360 * times[i]) / times[size-1], // start_angle
                    (360 * times[i+1]) / times[size-1], // end_angle
                    colors[i], -1); // color, thickness [fill]
        }
    }
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_FramewiseHomography
    (JNIEnv *, jobject, jlong addrGray, jlong addrRgba)
{
    if (!prevCreated)
    {
        return;
    }
    tic();
    // -1: Load up the timing process
    int size = 7;
    long long times[size];
    times[0] = 0; // always start at zero

    // 0: Access as C++ arrays
    Mat& grayscale  = *(Mat*)addrGray;
    Mat& color = *(Mat*)addrRgba;
    times[1] = toc();

    // 1: compute keypoints
    Mat query_descriptors;
    vector<KeyPoint> query_keypoints;
    detector->detectAndCompute(grayscale, noArray(), query_keypoints, query_descriptors);
    times[2] = toc();

    // 2: Match and align keypoints
    vector< vector<DMatch> > matches;
    vector<Point2f> matched1, matched2;
    matcher->knnMatch(ref_descriptors, query_descriptors, matches, 2);
    times[3] = toc();
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            matched1.push_back(  ref_keypoints[matches[i][0].queryIdx].pt);
            matched2.push_back(query_keypoints[matches[i][0].trainIdx].pt);
        }
    }
    times[4] = toc();

    // 3: Compute homography
    Mat inlier_mask, homography;
    vector<Point2f> inliers1, inliers2;
    if(matched1.size() >= 4) {
        homography = findHomography(matched1, matched2,
                                    RANSAC, ransac_thresh, inlier_mask);
    }
    if(matched1.size() < 4 || homography.empty()) {
        return;
    }
    times[5] = toc();

    // 4: Draw bounding box
    vector<Point2f> new_bb;
    perspectiveTransform(object_bb, new_bb, homography);
    line( color, new_bb[0], new_bb[1], Scalar( 0, 255, 0), 4 );
    line( color, new_bb[1], new_bb[2], Scalar( 0, 255, 0), 4 );
    line( color, new_bb[2], new_bb[3], Scalar( 0, 255, 0), 4 );
    line( color, new_bb[3], new_bb[0], Scalar( 0, 255, 0), 4 );

    // 5: Compute all timings
    //(*env)->SetIntArrayRegion(env, result, 0, size, fill);
    times[6] = toc();

    // 5: Draw timings
    for (int i = 0; i < size-1; i++)
    {
        ellipse(color, Point(110, 110), Size (100, 100), 0, // mat, center, axes, ellipse_angle
                (360 * times[i]) / times[size-1], // start_angle
                (360 * times[i+1]) / times[size-1], // end_angle
                colors[i], -1); // color, thickness [fill]
    }
    //return result;
}



JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_SetBoundingBox
    (JNIEnv *, jobject, jfloat x1, jfloat y1, jfloat x2, jfloat y2)
{
        object_bb[0] = Point2f(x1, y1);
        object_bb[1] = Point2f(x1, y2);
        object_bb[2] = Point2f(x2, y2);
        object_bb[3] = Point2f(x2, y1);
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_ClearBoundingBox
    (JNIEnv *, jobject)
{
    prevCreated = false;
}

JNIEXPORT void JNICALL Java_com_mrmallironmaker_realtimeoverlay_MainActivity_SetBaseFrame
 (JNIEnv * env, jobject, jlong addrGray, jdouble f, jfloatArray baseOrientationArray)
{
    flength = f * downsample_factor;
    // 0: Access as C++ array
    Mat& full_grayscale = *(Mat*)addrGray;
    Mat grayscale;
    resize(full_grayscale, grayscale, Size(), downsample_factor, downsample_factor);

    // Compute keypoints
    //detector->detectAndCompute(grayscale, noArray(), ref_keypoints, ref_descriptors);

    // automatic initialization
    goodFeaturesToTrack(grayscale, baselinePoints, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
    cornerSubPix(grayscale, baselinePoints, subPixWinSize, Size(-1,-1), termcrit);
    grayscale.copyTo(prevGray);

    // get height and width
    double height = (double) grayscale.rows;
    double width = (double) grayscale.cols;

    // initialize points matrix
    Mat temp_keypoints(3, baselinePoints.size(), CV_64F);
    for (int i = 0; i < baselinePoints.size(); i++)
    {
        temp_keypoints.at<double>(0, i) = (baselinePoints[i].x - width / 2) / flength;
        temp_keypoints.at<double>(1, i) = (baselinePoints[i].y - height / 2) / flength;
        temp_keypoints.at<double>(2, i) = 1.0;
    }

    // convert to world coordinates
    jfloat* bo = env->GetFloatArrayElements(baseOrientationArray, 0);
    //Mat baseOrientation(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            baseOrientation.at<double>(i, j) = bo[i*3 + j];
        }
    }
    Mat temp = baseOrientation * temp_keypoints;
    keypoints3d = temp.clone();

    prevCreated = true;
}
}
