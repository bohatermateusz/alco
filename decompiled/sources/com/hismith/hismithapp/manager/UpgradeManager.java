package com.hismith.hismithapp.manager;

import android.app.Dialog;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.utils.AppManager;
import com.hismith.hismithapp.utils.AppUtil;
import com.hismith.hismithapp.utils.Constans;
import com.hismith.hismithapp.utils.LanguageUtil;
import com.hismith.hismithapp.utils.MeasureUtil;
import com.hismith.hismithapp.utils.SharedPreferencesUtil;

/* loaded from: classes3.dex */
public class UpgradeManager {
    private static boolean isShow = false;
    private static Dialog mUpgradeDialog;

    public static void dismissDialog() {
        Dialog dialog = mUpgradeDialog;
        if (dialog == null || !dialog.isShowing()) {
            return;
        }
        mUpgradeDialog.dismiss();
    }

    public static void initUpgrade(final Context context) {
        if (isShow) {
            return;
        }
        String str = (String) SharedPreferencesUtil.get(context, Constans.newVersion, "");
        final int intValue = ((Integer) SharedPreferencesUtil.get(context, Constans.newVersionForcibly, 0)).intValue();
        int intValue2 = ((Integer) SharedPreferencesUtil.get(context, Constans.newVersionPrompt, 0)).intValue();
        String str2 = (String) SharedPreferencesUtil.get(context, Constans.newVersionDesc, "");
        if (AppUtil.getAppVersionName(context).compareTo(str) >= 0 || intValue2 != 1) {
            return;
        }
        showUpgradeDialog(context, str, intValue, str2, new View.OnClickListener() { // from class: com.hismith.hismithapp.manager.UpgradeManager.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                int i = intValue;
                if (i == 0) {
                    UpgradeManager.dismissDialog();
                } else if (i == 1) {
                    AppManager.getInstance().AppExit();
                }
            }
        }, new View.OnClickListener() { // from class: com.hismith.hismithapp.manager.UpgradeManager.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                Context context2 = context;
                AppUtil.shareAppDl(context2, AppUtil.getPackageName(context2));
                if (intValue == 0) {
                    UpgradeManager.dismissDialog();
                }
            }
        });
    }

    public static void showUpgradeDialog(Context context, String str, int i, String str2, View.OnClickListener onClickListener, View.OnClickListener onClickListener2) {
        Dialog dialog = mUpgradeDialog;
        if (dialog == null || !dialog.isShowing()) {
            mUpgradeDialog = new Dialog(context, R.style.MyDialog);
            View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_upgrade, (ViewGroup) null, false);
            TextView textView = (TextView) inflate.findViewById(R.id.dialog_upgrade_txt_title);
            TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_upgrade_txt_version);
            TextView textView3 = (TextView) inflate.findViewById(R.id.dialog_upgrade_txt_desc);
            TextView textView4 = (TextView) inflate.findViewById(R.id.dialog_upgrade_txt_cancel);
            TextView textView5 = (TextView) inflate.findViewById(R.id.dialog_upgrade_txt_upgrade);
            LanguageUtil.setText(context, textView, "versioUpdate");
            textView2.setText(str);
            if (i == 1) {
                LanguageUtil.setText(context, textView4, "exitButton");
            } else if (i == 0) {
                LanguageUtil.setText(context, textView4, "DontUpdate");
            }
            LanguageUtil.setText(context, textView5, "updateNow");
            textView3.setText(str2);
            textView4.setOnClickListener(onClickListener);
            textView5.setOnClickListener(onClickListener2);
            mUpgradeDialog.setContentView(inflate);
            mUpgradeDialog.setCanceledOnTouchOutside(false);
            if (i == 1) {
                mUpgradeDialog.setCancelable(false);
            }
            Window window = mUpgradeDialog.getWindow();
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.8d);
            window.setAttributes(attributes);
            mUpgradeDialog.show();
            isShow = true;
        }
    }
}
