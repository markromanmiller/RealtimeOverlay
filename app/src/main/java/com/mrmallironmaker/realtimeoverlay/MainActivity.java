package com.mrmallironmaker.realtimeoverlay;

import com.mrmallironmaker.realtimeoverlay.util.SystemUiHider;

import android.annotation.TargetApi;
import android.app.Activity;
import android.content.Context;
import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.Nullable;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.Timer;
import java.util.TimerTask;


/**
 * An example full-screen activity that shows and hides the system UI (i.e.
 * status bar and navigation/system bar) with user interaction.
 *
 * @see SystemUiHider
 */
public class MainActivity extends Activity implements CvCameraViewListener2, View.OnTouchListener {
    private final String TAG = "RealtimeOverlay";
    private SensorManager sensorManager;
    private SensorEventListener sensorEventListener;
    private Timer orientationUpdate = new Timer();

    private Mat mRgba;
    private Mat mGray;
    private Mat mIntermediateMat;
    private boolean openCVIsLoaded = false;
    private boolean differentialColor = true;
    private float[] prevPhoneOrientation;
    private boolean prevSharpMotion;
    private boolean displayDebug = false;
    private boolean boxMaking = false;
    private boolean boxMakingPrimed = false;
    private float touchDownX, touchDownY;
    private float touchCurrentX, touchCurrentY;
    private float mStretch = .75f;

    private double flength, thetaW, thetaH;

    private MenuItem mItemClear;
    private MenuItem mItemDraw;
    private MenuItem mItemDRModality;

    private CameraBridgeViewBase   mOpenCvCameraView;

    static {
        System.loadLibrary("helloworld");
    }

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("helloworld");
                    Initialize();
                    openCVIsLoaded = true;

                    mOpenCvCameraView.enableView();
                    prevPhoneOrientation = sensorEventListener.getPhoneOrientation();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
        super.onPause();
        sensorManager.unregisterListener(sensorEventListener);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.e(TAG, "Beginning.");
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);

        sensorManager = (SensorManager)getSystemService(SENSOR_SERVICE);
        sensorEventListener = new SensorEventListener(sensorManager);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setOnTouchListener(this);

        Camera c = null;
        try {
            c = Camera.open(); // attempt to get a Camera instance
        }
        catch (Exception e){
            // Camera is not available (in use or does not exist)
        }
        Camera.Parameters p = c.getParameters();
        thetaW = Math.toRadians(p.getHorizontalViewAngle());
        thetaH = Math.toRadians(p.getVerticalViewAngle());
        c.release();
        c = null;

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mItemDraw = menu.add("Draw Rectangle");
        mItemClear = menu.add("Clear");
        mItemDRModality = menu.add("Show/Hide Info");
        return true;
    }

    public boolean onTouch(View v, MotionEvent m)
    {
        if (!boxMakingPrimed)
        {
            return false;
        }
        switch (m.getAction())
        {
        case MotionEvent.ACTION_DOWN:
            onDrawClicked();
            touchDownX = m.getX() * mStretch;
            touchDownY = m.getY() * mStretch;
            boxMaking = true;
            return true;

        case MotionEvent.ACTION_MOVE:
            touchCurrentX = m.getX() * mStretch;
            touchCurrentY = m.getY() * mStretch;
            return true;

        case MotionEvent.ACTION_UP:
            touchCurrentX = m.getX() * mStretch;
            touchCurrentY = m.getY() * mStretch;
            SetBoundingBox(touchCurrentX, touchCurrentY, touchDownX, touchDownY);
            boxMaking = false;
            boxMakingPrimed = false;
            return true;
        }
        return false;
    }

    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);

        if (item == mItemDraw) {
            boxMakingPrimed = true;
        } else if (item == mItemClear) {
            ClearBoundingBox();
        } else if (item == mItemDRModality) {
            if (displayDebug)
            {
                mOpenCvCameraView.disableFpsMeter();
                displayDebug = false;
            }
            else
            {
                mOpenCvCameraView.enableFpsMeter();
                displayDebug = true;
            }
        }

        return true;
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        sensorManager.registerListener(sensorEventListener, sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(sensorEventListener, sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_FASTEST);
        sensorManager.registerListener(sensorEventListener, sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_FASTEST);
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDrawClicked() {
        //differentialColor = !differentialColor;
        SetBaseFrame(mGray.getNativeObjAddr(), flength, sensorEventListener.getPhoneOrientation());
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        Mat mReturn;
        if (boxMaking) {
            if (flength == 0.0) {
                flength = Math.sqrt( mRgba.rows() * mRgba.cols()
                                / (4 * Math.tan(thetaH / 2) * Math.tan(thetaW / 2))
                ) / mStretch;
            }
            mReturn = mRgba.clone();
            Imgproc.rectangle(mReturn, new Point(touchDownX, touchDownY),
                    new Point(touchCurrentX, touchCurrentY), new Scalar(0, 0, 255), 4);

        }
        else {
            // input frame has RGBA format
            mRgba = inputFrame.rgba();
            mGray = inputFrame.gray();
            //if (openCVIsLoaded && differentialColor) {
            //FramewiseImgDiff(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
            //}
            if (openCVIsLoaded) {
                LKHomography(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr(), prevSharpMotion,
                        prevPhoneOrientation, displayDebug);
            }
            mReturn = mRgba;
        }
        prevPhoneOrientation = sensorEventListener.getPhoneOrientation();
        prevSharpMotion = sensorEventListener.getSharpMotion();
        return mReturn;
    }

    private native void FramewiseImgDiff(long matAddrGr, long matAddrRgba);
    private native void FramewiseHomography(long matAddrGr, long matAddrRgba);
    private native void LKHomography(long matAddrGr, long matAddrRgba, boolean sharpMotion,
                                     float[] phoneOrientation, boolean displayDebug);
    private native void Initialize();
    private native void SetBaseFrame(long matAddrBase, double flength, float[] phoneOrientation);
    private native void ClearBoundingBox();
    private native void SetBoundingBox(float x1, float y1, float x2, float y2);
}
