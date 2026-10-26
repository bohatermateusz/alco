package com.hismith.hismithapp.databinding;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import androidx.viewbinding.ViewBinding;
import androidx.viewbinding.ViewBindings;
import com.hismith.hismithapp.R;

/* loaded from: classes3.dex */
public final class ActivityUnsubscribeBinding implements ViewBinding {
    private final LinearLayout rootView;
    public final TextView titleTxtName;
    public final Button unsubscribeBtnCode;
    public final Button unsubscribeBtnError;
    public final Button unsubscribeBtnSubmit;
    public final EditText unsubscribeEditCode;
    public final TextView unsubscribeEditEmail;
    public final ImageView unsubscribeImgCode;
    public final ImageView unsubscribeImgEmail;
    public final ImageView unsubscribeImgError;
    public final LinearLayout unsubscribeLayoutEdit;
    public final LinearLayout unsubscribeLayoutEmpty;
    public final TextView unsubscribeTxtError;
    public final TextView unsubscribeTxtRemoveAccountNote;

    private ActivityUnsubscribeBinding(LinearLayout linearLayout, TextView textView, Button button, Button button2, Button button3, EditText editText, TextView textView2, ImageView imageView, ImageView imageView2, ImageView imageView3, LinearLayout linearLayout2, LinearLayout linearLayout3, TextView textView3, TextView textView4) {
        this.rootView = linearLayout;
        this.titleTxtName = textView;
        this.unsubscribeBtnCode = button;
        this.unsubscribeBtnError = button2;
        this.unsubscribeBtnSubmit = button3;
        this.unsubscribeEditCode = editText;
        this.unsubscribeEditEmail = textView2;
        this.unsubscribeImgCode = imageView;
        this.unsubscribeImgEmail = imageView2;
        this.unsubscribeImgError = imageView3;
        this.unsubscribeLayoutEdit = linearLayout2;
        this.unsubscribeLayoutEmpty = linearLayout3;
        this.unsubscribeTxtError = textView3;
        this.unsubscribeTxtRemoveAccountNote = textView4;
    }

    public static ActivityUnsubscribeBinding bind(View view) {
        int i = R.id.title_txt_name;
        TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
        if (textView != null) {
            i = R.id.unsubscribe_btn_code;
            Button button = (Button) ViewBindings.findChildViewById(view, i);
            if (button != null) {
                i = R.id.unsubscribe_btn_error;
                Button button2 = (Button) ViewBindings.findChildViewById(view, i);
                if (button2 != null) {
                    i = R.id.unsubscribe_btn_submit;
                    Button button3 = (Button) ViewBindings.findChildViewById(view, i);
                    if (button3 != null) {
                        i = R.id.unsubscribe_edit_code;
                        EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText != null) {
                            i = R.id.unsubscribe_edit_email;
                            TextView textView2 = (TextView) ViewBindings.findChildViewById(view, i);
                            if (textView2 != null) {
                                i = R.id.unsubscribe_img_code;
                                ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView != null) {
                                    i = R.id.unsubscribe_img_email;
                                    ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView2 != null) {
                                        i = R.id.unsubscribe_img_error;
                                        ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView3 != null) {
                                            i = R.id.unsubscribe_layout_edit;
                                            LinearLayout linearLayout = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                            if (linearLayout != null) {
                                                i = R.id.unsubscribe_layout_empty;
                                                LinearLayout linearLayout2 = (LinearLayout) ViewBindings.findChildViewById(view, i);
                                                if (linearLayout2 != null) {
                                                    i = R.id.unsubscribe_txt_error;
                                                    TextView textView3 = (TextView) ViewBindings.findChildViewById(view, i);
                                                    if (textView3 != null) {
                                                        i = R.id.unsubscribe_txt_removeAccountNote;
                                                        TextView textView4 = (TextView) ViewBindings.findChildViewById(view, i);
                                                        if (textView4 != null) {
                                                            return new ActivityUnsubscribeBinding((LinearLayout) view, textView, button, button2, button3, editText, textView2, imageView, imageView2, imageView3, linearLayout, linearLayout2, textView3, textView4);
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

    public static ActivityUnsubscribeBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityUnsubscribeBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_unsubscribe, viewGroup, false);
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
