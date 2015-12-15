package com.mrmallironmaker.realtimeoverlay;

import android.hardware.Camera;

/**
 * Created by mark on 11/12/15.
 */
public class PreviewEditor implements Camera.PreviewCallback{
    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        for (int i = 0; i < data.length; i += 3)
        {
            // do something that changes some public value clearly.
            data[i] = 0;
        }
    }
}
