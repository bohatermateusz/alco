package com.hismith.hismithapp.utils;

import android.content.Context;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import java.util.Iterator;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;

/* loaded from: classes3.dex */
public class SoftKeyboardUtil {
    public static void hideSoftKeyboard(Context context, View view) {
        if (view == null) {
            return;
        }
        ((InputMethodManager) context.getSystemService("input_method")).hideSoftInputFromWindow(view.getWindowToken(), 2);
    }

    public static void hideSoftKeyboard(Context context, List<View> list) {
        if (list == null) {
            return;
        }
        InputMethodManager inputMethodManager = (InputMethodManager) context.getSystemService("input_method");
        Iterator<View> it = list.iterator();
        while (it.hasNext()) {
            inputMethodManager.hideSoftInputFromWindow(it.next().getWindowToken(), 2);
        }
    }

    public static void openKeyBoard(final Context context, final EditText editText) {
        try {
            editText.setFocusable(true);
            editText.setFocusableInTouchMode(true);
            editText.requestFocus();
            new Timer().schedule(new TimerTask() { // from class: com.hismith.hismithapp.utils.SoftKeyboardUtil.1
                @Override // java.util.TimerTask, java.lang.Runnable
                public void run() {
                    ((InputMethodManager) context.getSystemService("input_method")).showSoftInput(editText, 1);
                    EditText editText2 = editText;
                    editText2.setSelection(editText2.getText().length());
                }
            }, 500L);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void showInput(Context context, EditText editText) {
        editText.requestFocus();
        ((InputMethodManager) context.getSystemService("input_method")).showSoftInput(editText, 1);
    }
}
