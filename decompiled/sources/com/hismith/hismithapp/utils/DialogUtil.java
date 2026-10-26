package com.hismith.hismithapp.utils;

import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.text.InputFilter;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.hismith.hismithapp.R;
import com.hismith.hismithapp.filter.BytesLengthFilter;

/* loaded from: classes3.dex */
public class DialogUtil {
    private static int dialogType;
    private static Dialog mDialog;

    public interface OnEditContentCallback {
        void onEditContent(String str);
    }

    public interface OnItemClickListener {
        void onItemClick(int i);
    }

    public interface onCheckListCallback {
        void onCheckList(String str);
    }

    public static void dismissDialog() {
        try {
            Dialog dialog = mDialog;
            if (dialog == null || !dialog.isShowing()) {
                return;
            }
            mDialog.dismiss();
            dialogType = 0;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static boolean isShowing() {
        try {
            Dialog dialog = mDialog;
            if (dialog != null) {
                return dialog.isShowing();
            }
            return false;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public static boolean isShowing(int i) {
        try {
            Dialog dialog = mDialog;
            if (dialog == null || dialogType != i) {
                return false;
            }
            return dialog.isShowing();
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    public static void showConfirmDialog(Context context, String str, String str2, String str3, String str4, View.OnClickListener onClickListener) {
        showConfirmDialog(context, str, str2, str3, str4, onClickListener, 17);
    }

    public static void showConfirmDialog(Context context, String str, String str2, String str3, String str4, View.OnClickListener onClickListener, int i) {
        dismissDialog();
        final Dialog dialog = new Dialog(context, R.style.MyDialog);
        View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_confirm, (ViewGroup) null, false);
        LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_title);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_title);
        LinearLayout linearLayout2 = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_content);
        TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_content);
        if (i > 0) {
            textView2.setGravity(i);
        }
        Button button = (Button) inflate.findViewById(R.id.dialog_confirm_btn_cancel);
        LanguageUtil.setText(context, button, "cancel");
        Button button2 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm);
        LanguageUtil.setText(context, button2, "confirm");
        if (str != null) {
            LanguageUtil.setText(context, textView, str);
        } else {
            linearLayout.setVisibility(8);
        }
        if (str2 != null) {
            LanguageUtil.setText(context, textView2, str2);
        } else {
            linearLayout2.setVisibility(8);
        }
        if (str3 != null) {
            LanguageUtil.setText(context, button, str3);
        }
        if (str4 != null) {
            LanguageUtil.setText(context, button2, str4);
        }
        button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.2
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        button2.setOnClickListener(onClickListener);
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 2;
    }

    public static void showConfirmDialog(Context context, String str, String str2, String str3, String str4, View.OnClickListener onClickListener, View.OnClickListener onClickListener2) {
        dismissDialog();
        Dialog dialog = new Dialog(context, R.style.MyDialog);
        View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_confirm, (ViewGroup) null, false);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_title);
        LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_content);
        TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_content);
        Button button = (Button) inflate.findViewById(R.id.dialog_confirm_btn_cancel);
        LanguageUtil.setText(context, button, "cancel");
        Button button2 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm);
        LanguageUtil.setText(context, button2, "confirm");
        if (str != null) {
            LanguageUtil.setText(context, textView, str);
        }
        if (str2 != null) {
            LanguageUtil.setText(context, textView2, str2);
        } else {
            linearLayout.setVisibility(8);
        }
        if (str3 != null) {
            LanguageUtil.setText(context, button, str3);
        }
        if (str4 != null) {
            LanguageUtil.setText(context, button2, str4);
        }
        button.setOnClickListener(onClickListener);
        button2.setOnClickListener(onClickListener2);
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.setCanceledOnTouchOutside(true);
        dialog.show();
        mDialog = dialog;
        dialogType = 2;
    }

    public static void showEditDialog2(Activity activity, String str, String str2, String str3, String str4, OnEditContentCallback onEditContentCallback) {
        showEditDialog2(activity, str, str2, str3, str4, onEditContentCallback, 0);
    }

    public static void showEditDialog2(Activity activity, String str, String str2, String str3, String str4, final OnEditContentCallback onEditContentCallback, int i) {
        dismissDialog();
        final Dialog dialog = new Dialog(activity, R.style.MyDialog);
        View inflate = LayoutInflater.from(activity).inflate(R.layout.dialog_edit2, (ViewGroup) null, false);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_edit_txt_title);
        final EditText editText = (EditText) inflate.findViewById(R.id.dialog_edit_edit_name);
        TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_edit_txt_cancel);
        LanguageUtil.setText(activity, textView2, "cancel");
        TextView textView3 = (TextView) inflate.findViewById(R.id.dialog_edit_txt_confirm);
        LanguageUtil.setText(activity, textView3, "confirm");
        if (str != null) {
            LanguageUtil.setText(activity, textView, str);
        }
        if (str2 != null) {
            LanguageUtil.setText(activity, editText, str2);
        }
        if (str3 != null) {
            LanguageUtil.setText(activity, textView2, str3);
        }
        if (str4 != null) {
            LanguageUtil.setText(activity, textView3, str4);
        }
        textView2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.3
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        textView3.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.4
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                String obj = editText.getText().toString();
                OnEditContentCallback onEditContentCallback2 = onEditContentCallback;
                if (onEditContentCallback2 != null) {
                    onEditContentCallback2.onEditContent(obj);
                }
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        if (i > 0) {
            editText.setFilters(new InputFilter[]{new BytesLengthFilter(i)});
        }
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (activity.getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 3;
    }

    public static void showInviteDialog(Context context, String str, String str2, String str3, final View.OnClickListener onClickListener) {
        dismissDialog();
        final Dialog dialog = new Dialog(context, R.style.MyDialog);
        View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_invite_join, (ViewGroup) null, false);
        LanguageUtil.setText(context, (TextView) inflate.findViewById(R.id.txt_content), str);
        TextView textView = (TextView) inflate.findViewById(R.id.txt_cancel);
        LanguageUtil.setText(context, textView, str2);
        textView.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.6
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        TextView textView2 = (TextView) inflate.findViewById(R.id.txt_confirm);
        LanguageUtil.setText(context, textView2, str3);
        textView2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.7
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                View.OnClickListener onClickListener2 = onClickListener;
                if (onClickListener2 != null) {
                    onClickListener2.onClick(view);
                }
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        dialog.setCanceledOnTouchOutside(true);
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 5;
    }

    public static void showMultipleConfirmDialog(Context context, String str, String str2, String str3, String str4, String str5, View.OnClickListener onClickListener, View.OnClickListener onClickListener2) {
        dismissDialog();
        final Dialog dialog = new Dialog(context, R.style.MyDialog);
        View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_multiple_confirm, (ViewGroup) null, false);
        LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_title);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_title);
        LinearLayout linearLayout2 = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_content);
        TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_content);
        Button button = (Button) inflate.findViewById(R.id.dialog_confirm_btn_cancel);
        LanguageUtil.setText(context, button, "cancel");
        Button button2 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm);
        LanguageUtil.setText(context, button2, "confirm");
        Button button3 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm_1);
        if (str != null) {
            LanguageUtil.setText(context, textView, str);
        } else {
            linearLayout.setVisibility(8);
        }
        if (str2 != null) {
            LanguageUtil.setText(context, textView2, str2);
        } else {
            linearLayout2.setVisibility(8);
        }
        if (str3 != null) {
            LanguageUtil.setText(context, button, str3);
        }
        if (str4 != null) {
            LanguageUtil.setText(context, button2, str4);
        }
        if (str5 != null) {
            LanguageUtil.setText(context, button3, str5);
        } else {
            button3.setVisibility(8);
        }
        button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.8
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        button2.setOnClickListener(onClickListener);
        button3.setOnClickListener(onClickListener2);
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 7;
    }

    public static void showSelectDialog(Activity activity, String str, String[] strArr, String str2, String str3, final OnItemClickListener onItemClickListener) {
        dismissDialog();
        final Dialog dialog = new Dialog(activity, R.style.MyDialog);
        View inflate = LayoutInflater.from(activity).inflate(R.layout.dialog_select, (ViewGroup) null, false);
        LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.dialog_select_layout_content);
        for (final int i = 0; i < strArr.length; i++) {
            Button button = new Button(activity);
            button.setText(strArr[i]);
            button.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.5
                @Override // android.view.View.OnClickListener
                public void onClick(View view) {
                    OnItemClickListener onItemClickListener2 = OnItemClickListener.this;
                    if (onItemClickListener2 != null) {
                        onItemClickListener2.onItemClick(i);
                    }
                    if (dialog.isShowing()) {
                        dialog.dismiss();
                    }
                }
            });
            linearLayout.addView(button);
        }
        dialog.setContentView(inflate);
        dialog.setCanceledOnTouchOutside(true);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (activity.getWindowManager().getDefaultDisplay().getWidth() * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 4;
    }

    public static void showTipsDialog(Context context, String str, String str2, String str3) {
        dismissDialog();
        final Dialog dialog = new Dialog(context, R.style.MyDialog);
        View inflate = LayoutInflater.from(context).inflate(R.layout.dialog_confirm, (ViewGroup) null, false);
        TextView textView = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_title);
        LinearLayout linearLayout = (LinearLayout) inflate.findViewById(R.id.dialog_confirm_layout_content);
        TextView textView2 = (TextView) inflate.findViewById(R.id.dialog_confirm_txt_content);
        Button button = (Button) inflate.findViewById(R.id.dialog_confirm_btn_cancel);
        button.setVisibility(8);
        LanguageUtil.setText(context, button, "cancel");
        Button button2 = (Button) inflate.findViewById(R.id.dialog_confirm_btn_confirm);
        LanguageUtil.setText(context, button2, "confirm");
        if (str != null) {
            LanguageUtil.setText(context, textView, str);
        } else {
            LanguageUtil.setText(context, textView, "prompt");
        }
        if (str2 != null) {
            LanguageUtil.setText(context, textView2, str2);
        } else {
            linearLayout.setVisibility(8);
        }
        if (str3 != null) {
            LanguageUtil.setText(context, button2, str3);
        }
        button2.setOnClickListener(new View.OnClickListener() { // from class: com.hismith.hismithapp.utils.DialogUtil.1
            @Override // android.view.View.OnClickListener
            public void onClick(View view) {
                if (dialog.isShowing()) {
                    dialog.dismiss();
                }
            }
        });
        dialog.setContentView(inflate);
        Window window = dialog.getWindow();
        if (window != null) {
            WindowManager.LayoutParams attributes = window.getAttributes();
            attributes.width = (int) (MeasureUtil.getScreenSize(context).x * 0.9d);
            window.setAttributes(attributes);
        }
        dialog.show();
        mDialog = dialog;
        dialogType = 1;
    }
}
