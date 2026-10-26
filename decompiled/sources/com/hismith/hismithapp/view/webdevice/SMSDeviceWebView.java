package com.hismith.hismithapp.view.webdevice;

import android.content.Context;
import android.util.AttributeSet;
import com.hismith.hismithapp.model.SocketSpeedModel;
import com.hismith.hismithapp.view.CustomWebView;

/* loaded from: classes3.dex */
public class SMSDeviceWebView extends CustomWebView {
    public SMSDeviceWebView(Context context) {
        super(context);
    }

    public SMSDeviceWebView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
    }

    public SMSDeviceWebView(Context context, AttributeSet attributeSet, int i) {
        super(context, attributeSet, i);
    }

    public void setSocketSpeedData(SocketSpeedModel socketSpeedModel) {
        if (socketSpeedModel != null) {
            try {
                "setStrokeNow".equals(socketSpeedModel.getAction());
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
