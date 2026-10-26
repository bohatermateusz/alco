package com.hismith.hismithapp.floating;

import android.content.Context;
import android.view.View;
import com.hismith.hismithapp.utils.LogUtil;
import com.yhao.floatwindow.FloatWindow;

/* loaded from: classes3.dex */
public abstract class FloatBase {
    private Context mContext;
    private String mTag;

    public FloatBase(Context context, String str, View.OnClickListener onClickListener) {
        this.mContext = context;
        this.mTag = str;
    }

    public void destroy() {
        try {
            if (FloatWindow.get(this.mTag) != null) {
                LogUtil.e("清除悬浮窗：" + this.mTag);
                FloatWindow.destroy(this.mTag);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    abstract void initView(View.OnClickListener onClickListener);

    public abstract void refreshFloatClimaxUI(int i);

    public abstract void refreshFloatModeUI(int i);

    public abstract void refreshFloatMusicUI(int i);

    public abstract void refreshFloatPosition(int i);

    public abstract void refreshFloatReversal(int i);

    public abstract void refreshFloatRunMode(int i);

    public abstract void refreshFloatShutdown();

    public abstract void refreshFloatSmooth(int i);

    public abstract void refreshFloatSpeedUI(int i);

    public abstract void refreshFloatStartPosition(int i);

    public abstract void refreshFloatStopPosition(int i);

    public abstract void refreshFloatVibrateModeUI(int i);

    public abstract void refreshFloatVibrateUI(int i);

    public abstract void show();
}
