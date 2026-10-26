package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class DialogLineCreateBinding implements ViewBinding {
    public final Button dialogLineCreateBtnCancel;
    public final Button dialogLineCreateBtnComplete;
    public final EditText dialogLineCreateEditRoomName;
    public final LinearLayout dialogLineCreateLayoutRoomType;
    public final TextView dialogLineCreateTxtRoomName;
    public final TextView dialogLineCreateTxtRoomPrivate;
    public final TextView dialogLineCreateTxtRoomPublic;
    public final TextView dialogLineCreateTxtRoomType;
    public final TextView dialogLineCreateTxtTitle;
    public final TextView dialogLineCreateTxtUserLimit;
    private final LinearLayout rootView;
    public final TextView txtLimit2;
    public final TextView txtLimit3;
    public final TextView txtLimit4;
    public final TextView txtLimit5;

    private DialogLineCreateBinding(LinearLayout linearLayout, Button button, Button button2, EditText editText, LinearLayout linearLayout2, TextView textView, TextView textView2, TextView textView3, TextView textView4, TextView textView5, TextView textView6, TextView textView7, TextView textView8, TextView textView9, TextView textView10) {
        this.rootView = linearLayout;
        this.dialogLineCreateBtnCancel = button;
        this.dialogLineCreateBtnComplete = button2;
        this.dialogLineCreateEditRoomName = editText;
        this.dialogLineCreateLayoutRoomType = linearLayout2;
        this.dialogLineCreateTxtRoomName = textView;
        this.dialogLineCreateTxtRoomPrivate = textView2;
        this.dialogLineCreateTxtRoomPublic = textView3;
        this.dialogLineCreateTxtRoomType = textView4;
        this.dialogLineCreateTxtTitle = textView5;
        this.dialogLineCreateTxtUserLimit = textView6;
        this.txtLimit2 = textView7;
        this.txtLimit3 = textView8;
        this.txtLimit4 = textView9;
        this.txtLimit5 = textView10;
    }

    public static DialogLineCreateBinding bind(View view) {
        int i = R.id.dialog_line_create_btn_cancel;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.dialog_line_create_btn_complete;
            Button button2 = (Button) ViewBindings.findChildViewById(view, i);
            if (button2 != null) {
                i = R.id.dialog_line_create_edit_room_name;
                EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText != null) {
                    i = R.id.dialog_line_create_layout_room_type;
                    LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                    if (linearLayout != null) {
                        i = R.id.dialog_line_create_txt_room_name;
                        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                        if (textView != null) {
                            i = R.id.dialog_line_create_txt_room_private;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.dialog_line_create_txt_room_public;
                                TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                if (textView3 != null) {
                                    i = R.id.dialog_line_create_txt_room_type;
                                    TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                    if (textView4 != null) {
                                        i = R.id.dialog_line_create_txt_title;
                                        TextView textView5 = (TextView) ViewBindings.findChildViewById(view, i);
                                        if (textView5 != null) {
                                            i = R.id.dialog_line_create_txt_user_limit;
                                            TextView textView6 = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView6 != null) {
                                                i = R.id.txt_limit2;
                                                TextView textView7 = (TextView) ViewBindings.findChildViewById(view, i);
                                                if (textView7 != null) {
                                                    i = R.id.txt_limit3;
                                                    TextView textView8 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView8 != null) {
                                                        i = R.id.txt_limit4;
                                                        TextView textView9 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView9 != null) {
                                                            i = R.id.txt_limit5;
                                                            TextView textView10 = (TextView) ViewBindings.findChildViewById(view, i);
                                                            if (textView10 != null) {
                                                                return new DialogLineCreateBinding((LinearLayout) view, button, button2, editText, linearLayout, textView, textView2, textView3, textView4, textView5, textView6, textView7, textView8, textView9, textView10);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        throw new NullPointerException("Missing required view with ID: ".concat(view.getResources().getResourceName(i)));
    }

    public static DialogLineCreateBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static DialogLineCreateBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.dialog_line_create, viewGroup, false);
        if (z) {
            viewGroup.addView(inflate);
        }
        return bind(inflate);
    }

    @Override // androidx.viewbinding.ViewBinding
    public LinearLayout getRoot() {
        return this.rootView;
    }
}
