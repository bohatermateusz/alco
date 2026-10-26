package com.hismith.hismithapp.dialog;

import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.LanguageUtil;

/* loaded from: classes3.dex */
public class NetErrorDialog {

    public interface OnResetClickListener {
        void onResetClick();
    }

    public static void dissErrorDialog(Context context, ViewGroup viewGroup) {
        viewGroup.removeView(LayoutInflater.from(context).inflate(R.layout.dialog_net_error, viewGroup, false));
    }

    public static void showErrorDialog(Context context, final ViewGroup viewGroup, final OnResetClickListener onResetClickListener) {
        dissErrorDialog(context, viewGroup);
        final View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_net_error, viewGroup, false);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_net_txt_label);
        Button button = (Button) inflate.findViewById(R.id.dialog_net_btn_reset);
        LanguageUtil.setText(context, textView, "networkError");
        LanguageUtil.setText(context, button, "clickToRetry");
        button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.dialog.NetErrorDialog.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                viewGroup.removeView(inflate);
                OnResetClickListener onResetClickListener2 = onResetClickListener;
                if (onResetClickListener2 != null) {
                    onResetClickListener2.onResetClick();
                }
            }
        });
        viewGroup.addView(inflate);
    }
}
