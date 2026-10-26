package com.hismith.hismithapp.utils;

import android.app.Activity;
import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/* loaded from: classes3.dex */
public class StatusBarUtil {
    static final /* synthetic */ boolean $assertionsDisabled = false;

    private static void MIUISetStatusBarLightMode(Object obj, boolean z) {
        Window window = obj instanceof Activity ? ((Activity) obj).getWindow() : obj instanceof Window ? (Window) obj : null;
        if (window != null) {
            Class<?> cls = window.getClass();
            try {
                Class<?> cls2 = Class.forName("android.view.MiuiWindowManager$LayoutParams");
                int i = cls2.getField("EXTRA_FLAG_STATUS_BAR_DARK_MODE").getInt(cls2);
                Method method = cls.getMethod("setExtraFlags", Integer.TYPE, Integer.TYPE);
                if (z) {
                    method.invoke(window, Integer.valueOf(i), Integer.valueOf(i));
                } else {
                    method.invoke(window, 0, Integer.valueOf(i));
                }
                if (RomUtils.isMiUIV7OrAbove()) {
                    if (z) {
                        window.getDecorView().setSystemUiVisibility(9216);
                    } else {
                        window.getDecorView().setSystemUiVisibility(1280);
                    }
                }
            } catch (Exception unused) {
            }
        }
    }

    public static void fitTitleBar(final Activity activity, final View view) {
        final ViewGroup.LayoutParams layoutParams = view.getLayoutParams();
        if (layoutParams.height == -2 || layoutParams.height == -1) {
            view.post(new Runnable() { // from class: com.hismith.hismithapp.utils.StatusBarUtil.1
                @Override // java.lang.Runnable
                public void run() {
                    layoutParams.height = view.getHeight() + StatusBarUtil.getStatusBarHeight(activity);
                    View view2 = view;
                    view2.setPadding(view2.getPaddingLeft(), view.getPaddingTop() + StatusBarUtil.getStatusBarHeight(activity), view.getPaddingRight(), view.getPaddingBottom());
                }
            });
        } else {
            layoutParams.height += getStatusBarHeight(activity);
            view.setPadding(view.getPaddingLeft(), view.getPaddingTop() + getStatusBarHeight(activity), view.getPaddingRight(), view.getPaddingBottom());
        }
    }

    public static int getStatusBarHeight(Context context) {
        int statusBarHeight = MeasureUtil.getStatusBarHeight(context);
        return statusBarHeight == 0 ? MeasureUtil.dip2px(context, 25.0f) : statusBarHeight;
    }

    private static void setAndroidNativeLightStatusBar(Object obj, boolean z, boolean z2) {
        View decorView = obj instanceof Activity ? ((Activity) obj).getWindow().getDecorView() : obj instanceof Window ? ((Window) obj).getDecorView() : null;
        if (decorView == null) {
            return;
        }
        if (z) {
            if (z2) {
                decorView.setSystemUiVisibility(9216);
                return;
            } else {
                decorView.setSystemUiVisibility(8192);
                return;
            }
        }
        if (z2) {
            decorView.setSystemUiVisibility(1280);
        } else {
            decorView.setSystemUiVisibility(256);
        }
    }

    private static boolean setFlymeLightStatusBar(Object obj, boolean z) {
        Window window = obj instanceof Activity ? ((Activity) obj).getWindow() : obj instanceof Window ? (Window) obj : null;
        if (window == null) {
            return false;
        }
        try {
            WindowManager.LayoutParams attributes = window.getAttributes();
            Field declaredField = WindowManager.LayoutParams.class.getDeclaredField("MEIZU_FLAG_DARK_STATUS_BAR_ICON");
            Field declaredField2 = WindowManager.LayoutParams.class.getDeclaredField("meizuFlags");
            declaredField.setAccessible(true);
            declaredField2.setAccessible(true);
            int i = declaredField.getInt(null);
            int i2 = declaredField2.getInt(attributes);
            declaredField2.setInt(attributes, z ? i2 | i : (~i) & i2);
            window.setAttributes(attributes);
            return true;
        } catch (Exception unused) {
            return false;
        }
    }

    public static void setLightStatusBar(Activity activity, boolean z, boolean z2) {
        if (Constans.DEVICE_FIRM == -1) {
            Constans.DEVICE_FIRM = RomUtils.getLightStatusBarAvailableRomType();
        }
        int i = Constans.DEVICE_FIRM;
        if (i == 1) {
            MIUISetStatusBarLightMode(activity, z);
        } else if (i == 2) {
            setAndroidNativeLightStatusBar(activity, z, z2);
        } else {
            if (i != 3) {
                return;
            }
            setAndroidNativeLightStatusBar(activity, z, z2);
        }
    }

    public static void setLightStatusBar(Window window, boolean z, boolean z2) {
        int lightStatusBarAvailableRomType = RomUtils.getLightStatusBarAvailableRomType();
        if (lightStatusBarAvailableRomType == 1) {
            MIUISetStatusBarLightMode(window, z);
        } else if (lightStatusBarAvailableRomType == 2) {
            setAndroidNativeLightStatusBar(window, z, z2);
        } else {
            if (lightStatusBarAvailableRomType != 3) {
                return;
            }
            setAndroidNativeLightStatusBar(window, z, z2);
        }
    }

    public static void setStatusBarColor(Activity activity, int i) {
        Window window = activity.getWindow();
        window.clearFlags(67108864);
        window.addFlags(Integer.MIN_VALUE);
        window.setStatusBarColor(activity.getResources().getColor(i));
    }

    public static void transparencyBar(Activity activity) {
        Window window = activity.getWindow();
        window.clearFlags(67108864);
        window.addFlags(Integer.MIN_VALUE);
        window.setStatusBarColor(0);
        window.getDecorView().setSystemUiVisibility(1280);
    }
}
