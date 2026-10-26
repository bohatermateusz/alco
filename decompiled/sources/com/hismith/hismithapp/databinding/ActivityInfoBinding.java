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
public final class ActivityInfoBinding implements ViewBinding {
    public final Button infoBtnSubmit;
    public final EditText infoEditBirthdate;
    public final EditText infoEditCountry;
    public final EditText infoEditName;
    public final EditText infoEditPhone;
    public final ImageView infoImgHead;
    public final ImageView infoRbFemale;
    public final ImageView infoRbMale;
    public final ImageView infoRbTransgender;
    private final LinearLayout rootView;
    public final TextView titleTxtName;

    private ActivityInfoBinding(LinearLayout linearLayout, Button button, EditText editText, EditText editText2, EditText editText3, EditText editText4, ImageView imageView, ImageView imageView2, ImageView imageView3, ImageView imageView4, TextView textView) {
        this.rootView = linearLayout;
        this.infoBtnSubmit = button;
        this.infoEditBirthdate = editText;
        this.infoEditCountry = editText2;
        this.infoEditName = editText3;
        this.infoEditPhone = editText4;
        this.infoImgHead = imageView;
        this.infoRbFemale = imageView2;
        this.infoRbMale = imageView3;
        this.infoRbTransgender = imageView4;
        this.titleTxtName = textView;
    }

    public static ActivityInfoBinding bind(View view) {
        int i = R.id.info_btn_submit;
        Button button = (Button) ViewBindings.findChildViewById(view, i);
        if (button != null) {
            i = R.id.info_edit_birthdate;
            EditText editText = (EditText) ViewBindings.findChildViewById(view, i);
            if (editText != null) {
                i = R.id.info_edit_country;
                EditText editText2 = (EditText) ViewBindings.findChildViewById(view, i);
                if (editText2 != null) {
                    i = R.id.info_edit_name;
                    EditText editText3 = (EditText) ViewBindings.findChildViewById(view, i);
                    if (editText3 != null) {
                        i = R.id.info_edit_phone;
                        EditText editText4 = (EditText) ViewBindings.findChildViewById(view, i);
                        if (editText4 != null) {
                            i = R.id.info_img_head;
                            ImageView imageView = (ImageView) ViewBindings.findChildViewById(view, i);
                            if (imageView != null) {
                                i = R.id.info_rb_female;
                                ImageView imageView2 = (ImageView) ViewBindings.findChildViewById(view, i);
                                if (imageView2 != null) {
                                    i = R.id.info_rb_male;
                                    ImageView imageView3 = (ImageView) ViewBindings.findChildViewById(view, i);
                                    if (imageView3 != null) {
                                        i = R.id.info_rb_transgender;
                                        ImageView imageView4 = (ImageView) ViewBindings.findChildViewById(view, i);
                                        if (imageView4 != null) {
                                            i = R.id.title_txt_name;
                                            TextView textView = (TextView) ViewBindings.findChildViewById(view, i);
                                            if (textView != null) {
                                                return new ActivityInfoBinding((LinearLayout) view, button, editText, editText2, editText3, editText4, imageView, imageView2, imageView3, imageView4, textView);
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

    public static ActivityInfoBinding inflate(LayoutInflater layoutInflater) {
        return inflate(layoutInflater, null, false);
    }

    public static ActivityInfoBinding inflate(LayoutInflater layoutInflater, ViewGroup viewGroup, boolean z) {
        View inflate = layoutInflater.inflate(R.layout.activity_info, viewGroup, false);
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
