package com.hismith.hismithapp.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.hardware.usb.UsbDevice;
import com.facebook.devicerequests.internal.DeviceRequestsHelper;
import com.hismith.hismithapp.utils.LogUtil2;

/* loaded from: classes3.dex */
public class UsbReceiver extends BroadcastReceiver {
    public static final String ACTION_USB_PERMISSION = "com.android.example.USB_PERMISSION";

    @Override // android.content.BroadcastReceiver
    public void onReceive(Context context, Intent intent) {
        String action = intent.getAction();
        if (ACTION_USB_PERMISSION.equals(action)) {
            synchronized (this) {
                UsbDevice usbDevice = (UsbDevice) intent.getParcelableExtra(DeviceRequestsHelper.DEVICE_INFO_DEVICE);
                if (usbDevice != null) {
                    if (intent.getBooleanExtra("permission", false)) {
                        LogUtil2.e("USB 获取权限成功：" + usbDevice.getDeviceName());
                    } else {
                        LogUtil2.e("USB 获取权限失败：" + usbDevice.getDeviceName());
                    }
                }
            }
            return;
        }
        if ("android.hardware.usb.action.USB_DEVICE_ATTACHED".equals(action)) {
            LogUtil2.e("USB usb插入" + intent.getExtras().toString());
        } else if ("android.hardware.usb.action.USB_DEVICE_DETACHED".equals(action)) {
            LogUtil2.e("USB usb拔出" + intent.getExtras().toString());
        }
    }
}
