package com.hismith.hismithapp.dialog;

import android.app.Dialog;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public class ProgressDialog extends Dialog {
    public ProgressDialog(Context context) {
        super(context);
        addContentView(LayoutInflater.from(context).inflate(R.layout.dialog_progress, (ViewGroup) null, false), new ViewGroup.LayoutParams(-1, -2));
    }

    public ProgressDialog(Context context, int i) {
        super(context, i);
        addContentView(LayoutInflater.from(context).inflate(R.layout.dialog_progress, (ViewGroup) null, false), new ViewGroup.LayoutParams(-1, -2));
    }
}
