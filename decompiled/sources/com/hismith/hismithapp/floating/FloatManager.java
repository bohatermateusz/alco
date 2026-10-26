package com.hismith.hismithapp.floating;

import android.content.Context;
import android.view.View;
import com.hismith.hismithapp.utils.Constans;

/* loaded from: classes3.dex */
public class FloatManager {
    public static FloatBase getFloat(Context context, String str, String str2, View.OnClickListener onClickListener) {
        return Constans.DEVICE_TYPE_SM.equals(str) ? new FloatSM(context, str2, onClickListener) : Constans.DEVICE_TYPE_SM2.equals(str) ? new FloatSM2(context, str2, onClickListener) : Constans.DEVICE_TYPE_SMS.equals(str) ? new FloatCCSMS(context, str2, onClickListener) : Constans.DEVICE_TYPE_SMV.equals(str) ? new FloatCCSMV(context, str2, onClickListener) : new FloatSM(context, str2, onClickListener);
    }
}
