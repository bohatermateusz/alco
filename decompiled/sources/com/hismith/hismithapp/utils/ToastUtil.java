package com.hismith.hismithapp.utils;

import android.content.Context;
import android.text.TextUtils;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class ToastUtil {
    private static Toast toast;

    public static void cancelToast() {
        Toast toast2 = toast;
        if (toast2 != null) {
            toast2.cancel();
        }
    }

    public static void showLongToast(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 1);
            toast = makeText;
            makeText.show();
        }
    }

    public static void showLongToastCenter(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 1);
            toast = makeText;
            makeText.setGravity(17, 0, 0);
            toast.show();
        }
    }

    public static void showLongToastTop(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 1);
            toast = makeText;
            makeText.setGravity(48, 0, 0);
            toast.show();
        }
    }

    public static void showShortToast(Context context, int i) {
        showShortToast(context, context.getResources().getString(i));
    }

    public static void showShortToast(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 0);
            toast = makeText;
            makeText.show();
        }
    }

    public static void showShortToastCenter(Context context, int i) {
        showShortToastCenter(context, context.getResources().getString(i));
    }

    public static void showShortToastCenter(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 0);
            toast = makeText;
            makeText.setGravity(17, 0, 0);
            toast.show();
        }
    }

    public static void showShortToastTop(Context context, String str) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            Toast makeText = Toast.makeText(context, str, 0);
            toast = makeText;
            makeText.setGravity(48, 0, 0);
            toast.show();
        }
    }

    public static void showWarnLongToast(Context context, String str) {
        showWarnToast(context, 0, str, 1);
    }

    public static void showWarnShortToast(Context context, String str) {
        showWarnToast(context, 0, str, 0);
    }

    public static void showWarnToast(Context context, int i, String str, int i2) {
        if (context != null) {
            Toast toast2 = toast;
            if (toast2 != null) {
                toast2.cancel();
            }
            View inflate = LayoutInflater.from(context).inflate(R.layout.toast_warn, (ViewGroup) null);
            ImageView imageView = (ImageView) inflate.findViewById(R.id.img_icon);
            if (i != 0) {
                imageView.setImageResource(i);
            }
            TextView textView = (TextView) inflate.findViewById(R.id.txt_content);
            if (!TextUtils.isEmpty(str)) {
                LanguageUtil.setText(context, textView, str);
            }
            Toast toast3 = new Toast(context);
            toast = toast3;
            toast3.setGravity(17, 0, 0);
            toast.setDuration(i2);
            toast.setView(inflate);
            toast.show();
        }
    }
}
